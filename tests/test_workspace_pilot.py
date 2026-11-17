"""Rehearse pilot provisioning and rollback with actual SCRAM authentication."""

from contextlib import ExitStack
from pathlib import Path
from secrets import token_urlsafe
from time import time
from types import SimpleNamespace
from uuid import UUID, uuid4

import pandas as pd
import pytest
import streamlit as st
from psycopg2.extensions import encrypt_password
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.pool import NullPool
from streamlit.testing.v1 import AppTest

from db.demo_catalog import DEMO_READER
from db.workspace_datasets import (
    WorkspaceAccess, WorkspaceDatasets, create_analysis_results, create_upload_receipts, version_catalog,
)
from db.workspace_memberships import (
    MEMBERSHIP_READER, MembershipRegistry, create_membership_registry, grant_membership, revoke_membership,
)
from scripts.stage_demo_catalog import grant_demo_access, stage_demo_catalog


def _password_verifier(connection, role, password):
    return encrypt_password(password, role, connection.connection.driver_connection, algorithm="scram-sha-256")


def _provision(connection, schema, credentials):
    for role, password in credentials.items():
        connection.exec_driver_sql(
            f"CREATE ROLE {role} LOGIN NOINHERIT NOSUPERUSER NOBYPASSRLS NOCREATEROLE NOCREATEDB NOREPLICATION PASSWORD %s",
            (_password_verifier(connection, role, password),),
        )
    stage_demo_catalog(connection)
    grant_demo_access(connection)
    create_membership_registry(connection)
    connection.exec_driver_sql(f"GRANT USAGE ON SCHEMA workspace_auth TO {MEMBERSHIP_READER}")
    connection.exec_driver_sql(f"GRANT SELECT ON workspace_auth.memberships TO {MEMBERSHIP_READER}")
    connection.exec_driver_sql(f"CREATE SCHEMA {schema}")
    version_catalog(schema).create(connection)
    connection.exec_driver_sql(f"GRANT USAGE ON SCHEMA {schema} TO {schema}_reader")
    connection.exec_driver_sql(f"GRANT USAGE, CREATE ON SCHEMA {schema} TO {schema}_writer")
    connection.exec_driver_sql(f"GRANT SELECT ON {schema}.dataset_versions TO {schema}_reader")
    connection.exec_driver_sql(f"GRANT SELECT, INSERT, DELETE ON {schema}.dataset_versions TO {schema}_writer")
    create_upload_receipts(connection, UUID(hex=schema[2:]))
    connection.exec_driver_sql("CREATE TABLE public.pilot_legacy_sample (id integer PRIMARY KEY)")
    connection.exec_driver_sql("INSERT INTO public.pilot_legacy_sample VALUES (42)")


def _drain(owner, roles):
    with owner.begin() as connection:
        terminated = connection.execute(text("""
            SELECT pid, pg_catalog.pg_terminate_backend(pid, 5000)
            FROM pg_catalog.pg_stat_activity
            WHERE datname = current_database() AND backend_type = 'client backend'
              AND usename = ANY(:roles)
        """), {"roles": list(roles)}).all()
        assert all(stopped for _, stopped in terminated)
    return {pid for pid, _ in terminated}


_PILOT_APP = Path(__file__).resolve().parents[1] / "app.py"
_TEST_AUTH = {
    "redirect_uri": "http://localhost:8501/oauth2callback",
    "cookie_secret": "test-only-cookie",
    "client_id": "test-only-client",
    "client_secret": "test-only-secret",
    "server_metadata_url": "https://issuer.example/.well-known/openid-configuration",
}


def test_pilot_requires_explicit_scoped_credentials(monkeypatch):
    monkeypatch.setattr(st, "user", SimpleNamespace(
        is_logged_in=True,
        to_dict=lambda: {"is_logged_in": True, "iss": "https://issuer.example", "sub": "test", "exp": int(time()) + 60},
    ))
    workspace = uuid4()
    role_prefix = f"w_{workspace.hex}"
    app = AppTest.from_file(str(_PILOT_APP))
    app.secrets["auth"] = _TEST_AUTH
    app.secrets["DATABASE_URL"] = "postgresql+psycopg2://postgres:legacy-sentinel@localhost/postgres"
    for registry, writer_host in (
        ("postgresql+psycopg2://postgres:registry-sentinel@localhost/postgres", "localhost"),
        (f"postgresql+psycopg2://{MEMBERSHIP_READER}:registry-sentinel@localhost/postgres", "127.0.0.1"),
        (f"postgresql+psycopg2://{MEMBERSHIP_READER}:registry-sentinel@db.example/postgres", "db.example"),
    ):
        app.secrets["private_pilot"] = {
            "enabled": True,
            "issuer": "https://issuer.example",
            "registry_url": registry,
            "workspaces": {str(workspace): {
                "reader_url": f"postgresql+psycopg2://{role_prefix}_reader:secret@localhost/postgres",
                "writer_url": f"postgresql+psycopg2://{role_prefix}_writer:secret@{writer_host}/postgres",
            }},
        }
        app.run()
        assert not app.exception and not app.selectbox and not app.get("file_uploader")
        assert len(app.error) == 1 and "temporarily unavailable" in app.error[0].value
        assert "sentinel" not in app.error[0].value


def test_pilot_scram_rotation_and_demo_rollback(scram_postgres_owner, monkeypatch, synthetic_demo_files):
    owner = scram_postgres_owner
    workspace_id = uuid4()
    schema = f"w_{workspace_id.hex}"
    reader_role, writer_role = f"{schema}_reader", f"{schema}_writer"
    pilot_roles = (MEMBERSHIP_READER, reader_role, writer_role)
    passwords = {role: token_urlsafe(32) for role in (*pilot_roles, DEMO_READER)}
    claims = {"iss": "https://issuer.example", "sub": "invitee", "is_logged_in": True, "exp": int(time()) + 300}
    identity = (claims["iss"], claims["sub"])
    monkeypatch.setattr(st, "user", SimpleNamespace(to_dict=lambda: dict(claims), is_logged_in=True))

    with owner.begin() as connection:
        _provision(connection, schema, passwords)
        grant_membership(connection, *identity, workspace_id, can_manage=True)

    with ExitStack() as stack:
        def connect_as(role, password):
            engine = create_engine(owner.url.set(username=role, password=password), poolclass=NullPool, hide_parameters=True)
            stack.callback(engine.dispose)
            return engine

        old = {role: connect_as(role, passwords[role]) for role in pilot_roles}
        demo = connect_as(DEMO_READER, passwords[DEMO_READER])

        def guarded_demo():
            app = AppTest.from_file(str(_PILOT_APP))
            app.secrets["private_pilot"] = {"enabled": False}
            app.secrets["demo_reader"] = {"url": demo.url.render_as_string(hide_password=False)}
            app.run(timeout=15)
            assert not app.exception and not app.error
            assert [tab.label for tab in app.tabs] == ["Explorer", "Chat"]
            assert "891 matching rows" in [caption.value for caption in app.caption]
            assert not app.get("file_uploader")

        guarded_demo()
        with owner.begin() as connection:
            create_analysis_results(connection, workspace_id)
        guarded_demo()

        def workspace_app(engines):
            app = AppTest.from_file(str(_PILOT_APP))
            app.secrets["auth"] = _TEST_AUTH
            app.secrets["private_pilot"] = {
                "enabled": True,
                "issuer": identity[0],
                "registry_url": engines[MEMBERSHIP_READER].url.render_as_string(hide_password=False),
                "workspaces": {str(workspace_id): {
                    "reader_url": engines[reader_role].url.render_as_string(hide_password=False),
                    "writer_url": engines[writer_role].url.render_as_string(hide_password=False),
                }},
            }
            return app.run(timeout=15)

        membership = MembershipRegistry(old[MEMBERSHIP_READER]).lookup(*identity)
        access = WorkspaceAccess(membership.principal_id, uuid4(), workspace_id, can_manage=True)
        service = WorkspaceDatasets(access, lambda: access, old[reader_role], old[writer_role])
        dataset = service.upload(pd.DataFrame({"value": [17]}), "pilot.csv")
        private_table = f"{schema}.v_{dataset.version_id.hex}"
        before = workspace_app(old)
        assert not before.exception
        before.selectbox[0].set_value(dataset.version_id).run()
        assert "<td>17</td>" in before.get("html")[0].proto.body

        for engine, statement in (
            (old[MEMBERSHIP_READER], f"SELECT * FROM {private_table}"),
            (old[reader_role], f"DELETE FROM {private_table}"),
            (old[writer_role], "SELECT * FROM workspace_auth.memberships"),
            (old[writer_role], "SELECT * FROM public.pilot_legacy_sample"),
            (demo, f"SELECT * FROM {private_table}"),
            (demo, f"SELECT * FROM {schema}.analysis_results"),
            (old[MEMBERSHIP_READER], f"SELECT * FROM {schema}.analysis_results"),
            (old[reader_role], f"INSERT INTO {schema}.analysis_results DEFAULT VALUES"),
        ):
            with engine.connect() as connection, pytest.raises(ProgrammingError) as denied:
                connection.exec_driver_sql(statement)
            assert denied.value.orig.pgcode == "42501"

        established = {role: stack.enter_context(engine.connect()) for role, engine in old.items()}
        established_pids = {connection.exec_driver_sql("SELECT pg_backend_pid()").scalar_one()
                            for connection in established.values()}
        demo_session = stack.enter_context(demo.connect())
        demo_count = "SELECT count(*) FROM demo.titanic"
        assert demo_session.exec_driver_sql(demo_count).scalar_one() == 891

        rotated = {role: token_urlsafe(32) for role in pilot_roles}
        with owner.begin() as connection:
            for role, password in rotated.items():
                connection.exec_driver_sql(
                    f"ALTER ROLE {role} PASSWORD %s", (_password_verifier(connection, role, password),),
                )
        for role, connection in established.items():
            assert connection.exec_driver_sql("SELECT current_user").scalar_one() == role
            with pytest.raises(OperationalError, match="password authentication failed"):
                old[role].connect()
        assert _drain(owner, pilot_roles) == established_pids
        for connection in established.values():
            with pytest.raises(OperationalError):
                connection.exec_driver_sql("SELECT 1")

        stale_app = workspace_app(old)
        assert stale_app.error and not stale_app.selectbox and not stale_app.get("html")
        fresh = {role: connect_as(role, password) for role, password in rotated.items()}
        after = workspace_app(fresh)
        assert not after.exception and not after.error
        after.selectbox[0].set_value(dataset.version_id).run()
        assert "<td>17</td>" in after.get("html")[0].proto.body
        rotated_service = WorkspaceDatasets(access, lambda: access, fresh[reader_role], fresh[writer_role])
        probe = rotated_service.upload(pd.DataFrame({"value": [18]}), "rotation-probe.csv")
        rotated_service.delete(probe.version_id)

        second_session = workspace_app(fresh)
        second_session.selectbox[0].set_value(dataset.version_id).run()
        assert "<td>17</td>" in second_session.get("html")[0].proto.body
        next(button for button in after.button if button.label == "Run analysis").click().run()
        original_csv, = after.get("iframe")
        original_payload = original_csv.proto.srcdoc
        next(button for button in after.button if button.label == "Save result").click().run()
        assert not after.exception
        result_id = UUID(after.success[0].value.removeprefix("Saved analysis result ").removesuffix("."))
        assert after.get("iframe")[0].proto.srcdoc == original_payload
        next(button for button in second_session.button if button.label == "Prepare preview CSV").click().run()
        assert second_session.get("iframe")

        with owner.begin() as connection:
            revoke_membership(connection, *identity)
        for open_session in (after, second_session):
            open_session.run()
            assert not open_session.exception and not open_session.error
            assert not open_session.selectbox and not open_session.get("file_uploader")
            assert not open_session.get("html") and not open_session.get("iframe")
            assert any("account that has workspace access" in item.value for item in open_session.info)

        rollback_session = stack.enter_context(fresh[writer_role].connect())
        rollback_pid = rollback_session.exec_driver_sql("SELECT pg_backend_pid()").scalar_one()
        with owner.begin() as connection:
            for role in pilot_roles:
                connection.exec_driver_sql(f"ALTER ROLE {role} NOLOGIN")
        assert _drain(owner, pilot_roles) == {rollback_pid}
        with pytest.raises(OperationalError):
            rollback_session.exec_driver_sql("SELECT 1")
        for engine in fresh.values():
            with pytest.raises(OperationalError, match="not permitted to log in"):
                engine.connect()
        assert demo_session.exec_driver_sql(demo_count).scalar_one() == 891

        guarded_demo()
        with owner.connect() as connection:
            assert connection.exec_driver_sql(f"SELECT value FROM {private_table}").scalar_one() == 17
            assert connection.exec_driver_sql("SELECT id FROM public.pilot_legacy_sample").scalar_one() == 42
            assert connection.exec_driver_sql("SELECT active FROM workspace_auth.memberships").scalar_one() is False
            saved = connection.exec_driver_sql(
                f"SELECT result_id, version_id FROM {schema}.analysis_results"
            ).one()
            assert saved == (result_id, dataset.version_id)

        with owner.begin() as connection:
            for role in pilot_roles:
                connection.exec_driver_sql(f"ALTER ROLE {role} LOGIN")
            grant_membership(connection, *identity, workspace_id, can_manage=False)
        recovered = workspace_app(fresh)
        assert not recovered.exception and not recovered.error
        recovered.selectbox[0].set_value(dataset.version_id).run()
        next(widget for widget in recovered.text_input if widget.label == "Saved result ID").set_value(str(result_id))
        next(button for button in recovered.button if button.label == "Restore result").click().run()
        assert not recovered.exception and not recovered.error
        assert [item.value for item in recovered.text] == ["Rows: 1"]
        assert recovered.get("iframe")[0].proto.srcdoc == original_payload
        guarded_demo()
