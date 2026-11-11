"""Exercise private dataset operations with actual PostgreSQL roles."""

import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from io import StringIO
from threading import Barrier
from time import time
from types import SimpleNamespace
from uuid import uuid4

import pandas as pd
import pytest
import streamlit as st
from pydantic import ValidationError
from sqlalchemy import create_engine, event, inspect
from sqlalchemy.exc import DBAPIError, IntegrityError, ProgrammingError
from sqlalchemy.pool import NullPool
from streamlit.testing.v1 import AppTest

from db.demo_catalog import DEMO_READER
from db.workspace_datasets import (
    MAX_UPLOAD_ROWS, ROW_KEY, WORKSPACE_UPLOADS_PER_DAY, UploadLimitReached, UploadRequestConflict,
    WorkspaceAccess, WorkspaceColumnSummary, WorkspaceDatasets, WorkspaceRowCount,
    create_upload_receipts, version_catalog,
)
from scripts.stage_demo_catalog import grant_demo_access, stage_demo_catalog
from db.workspace_memberships import (
    MEMBERSHIP_READER, MembershipRegistry, create_membership_registry, grant_membership, revoke_membership,
)


@pytest.fixture(scope="module")
def workspace_database(postgres_owner):
    with postgres_owner.begin() as connection:
        connection.exec_driver_sql("REVOKE TEMPORARY ON DATABASE postgres FROM PUBLIC")
        connection.exec_driver_sql("REVOKE ALL ON SCHEMA public FROM PUBLIC")
        connection.exec_driver_sql(f"CREATE ROLE {DEMO_READER} LOGIN NOINHERIT")
        stage_demo_catalog(connection)
        grant_demo_access(connection)
    return postgres_owner


@pytest.fixture
def make_workspace(workspace_database):
    owner = workspace_database
    engines = []

    def create(principal):
        access = WorkspaceAccess(principal, uuid4(), uuid4(), can_manage=True)
        schema = f"w_{access.workspace_id.hex}"
        reader_role, writer_role = f"{schema}_reader", f"{schema}_writer"
        with owner.begin() as connection:
            connection.exec_driver_sql(f"CREATE ROLE {reader_role} LOGIN NOINHERIT")
            connection.exec_driver_sql(f"CREATE ROLE {writer_role} LOGIN NOINHERIT")
            connection.exec_driver_sql(f"CREATE SCHEMA {schema}")
            version_catalog(schema).create(connection)
            connection.exec_driver_sql(f"GRANT USAGE ON SCHEMA {schema} TO {reader_role}")
            connection.exec_driver_sql(f"GRANT USAGE, CREATE ON SCHEMA {schema} TO {writer_role}")
            connection.exec_driver_sql(f"GRANT SELECT ON {schema}.dataset_versions TO {reader_role}")
            connection.exec_driver_sql(f"GRANT SELECT, INSERT, DELETE ON {schema}.dataset_versions TO {writer_role}")
            create_upload_receipts(connection, access.workspace_id)
        reader = create_engine(owner.url.set(username=reader_role), poolclass=NullPool)
        writer = create_engine(owner.url.set(username=writer_role), poolclass=NullPool)
        engines.extend((reader, writer))
        state = SimpleNamespace(current=access)
        service = WorkspaceDatasets(access, lambda: state.current, reader, writer)
        return service, access, state, schema, reader, writer

    yield create
    for engine in engines:
        engine.dispose()


@pytest.fixture(scope="module")
def membership_reader(workspace_database):
    with workspace_database.begin() as connection:
        create_membership_registry(connection)
        connection.exec_driver_sql(f"CREATE ROLE {MEMBERSHIP_READER} LOGIN NOINHERIT")
        connection.exec_driver_sql(f"GRANT USAGE ON SCHEMA workspace_auth TO {MEMBERSHIP_READER}")
        connection.exec_driver_sql(f"GRANT SELECT ON workspace_auth.memberships TO {MEMBERSHIP_READER}")
    reader = create_engine(workspace_database.url.set(username=MEMBERSHIP_READER), poolclass=NullPool)
    try:
        yield reader
    finally:
        reader.dispose()


def test_membership_grants_and_revocation(workspace_database, make_workspace, membership_reader):
    _, access, _, schema, workspace_reader, _ = make_workspace("member")
    registry = MembershipRegistry(membership_reader)
    identity = ("https://issuer.example", "member")
    other_identity = ("https://other-issuer.example", "member")
    assert registry.lookup(*identity) is None
    with workspace_database.begin() as connection:
        grant_membership(connection, *identity, access.workspace_id, can_manage=True)
        grant_membership(connection, *other_identity, uuid4(), can_manage=True)
    original = registry.lookup(*identity)
    other = registry.lookup(*other_identity)
    assert original.workspace_id == access.workspace_id and original.can_manage
    assert other.principal_id != original.principal_id

    with workspace_database.begin() as connection:
        grant_membership(connection, *identity, access.workspace_id, can_manage=True)
    assert registry.lookup(*identity) == original
    with workspace_database.begin() as connection:
        grant_membership(connection, *identity, access.workspace_id, can_manage=False)
    downgraded = registry.lookup(*identity)
    assert downgraded.principal_id == original.principal_id
    assert not downgraded.can_manage and downgraded.revision != original.revision
    with workspace_database.begin() as connection:
        revoke_membership(connection, *identity)
    assert registry.lookup(*identity) is None
    assert registry.lookup(*other_identity) == other
    with workspace_database.begin() as connection:
        grant_membership(connection, *identity, access.workspace_id, can_manage=False)
    restored = registry.lookup(*identity)
    assert restored.principal_id == original.principal_id and restored.revision != downgraded.revision

    for engine, statement in (
        (membership_reader, "UPDATE workspace_auth.memberships SET active = false"),
        (membership_reader, f"SELECT * FROM {schema}.dataset_versions"),
        (workspace_reader, "SELECT * FROM workspace_auth.memberships"),
    ):
        with engine.connect() as connection, pytest.raises(ProgrammingError) as error:
            connection.exec_driver_sql(statement)
        assert error.value.orig.pgcode == "42501"
    with pytest.raises(PermissionError):
        MembershipRegistry(workspace_database).lookup(*identity)


def test_same_filename_isolated_and_delete_preserves_other_data(make_workspace, workspace_database):
    alice, _, _, alice_schema, alice_reader, alice_writer = make_workspace("alice")
    bob, _, _, bob_schema, _, _ = make_workspace("bob")
    a = alice.upload(pd.DataFrame({'Alice "note"': ["a-first", "a-second"]}, index=[5, 3]), "titanic.csv")
    b = bob.upload(pd.DataFrame({"Bob note": ["b-only"]}), "titanic.csv")
    assert a.dataset_id != b.dataset_id and a.version_id != b.version_id
    assert alice.list_datasets() == (a,)
    assert bob.list_datasets() == (b,)
    assert alice.metadata(a.version_id) == a
    assert bob.metadata(b.version_id) == b
    assert alice.preview(a.version_id, limit=1) == [{'Alice "note"': "a-first"}]
    assert bob.preview(b.version_id) == [{"Bob note": "b-only"}]

    for service, foreign in ((alice, b.version_id), (bob, a.version_id), (alice, uuid4())):
        for operation in (service.metadata, service.preview, service.preview_csv, service.delete):
            with pytest.raises(PermissionError, match="This dataset version is unavailable"):
                operation(foreign)

    a_table = f"{alice_schema}.v_{a.version_id.hex}"
    b_table = f"{bob_schema}.v_{b.version_id.hex}"
    demo_reader = create_engine(workspace_database.url.set(username=DEMO_READER), poolclass=NullPool)
    try:
        for engine, statement in (
            (alice_reader, f"SELECT * FROM {bob_schema}.dataset_versions"),
            (alice_reader, f"SELECT * FROM {b_table}"),
            (alice_reader, f"DELETE FROM {a_table}"),
            (alice_reader, f"DROP TABLE {a_table}"),
            (alice_writer, f"DROP TABLE {b_table}"),
            (alice_writer, "DROP TABLE demo.titanic"),
            (demo_reader, f"SELECT * FROM {a_table}"),
        ):
            with engine.connect() as connection, pytest.raises(ProgrammingError) as error:
                connection.exec_driver_sql(statement)
            assert error.value.orig.pgcode == "42501"

        alice.delete(a.version_id)
        assert alice.list_datasets() == ()
        with pytest.raises(PermissionError):
            alice.metadata(a.version_id)
        with pytest.raises(PermissionError):
            alice.preview_csv(a.version_id)
        assert f"v_{a.version_id.hex}" not in inspect(workspace_database).get_table_names(schema=alice_schema)
        assert bob.metadata(b.version_id) == b
        assert bob.preview(b.version_id) == [{"Bob note": "b-only"}]
        with demo_reader.connect() as connection:
            assert connection.exec_driver_sql("SELECT count(*) FROM demo.titanic").scalar_one() == 891
    finally:
        demo_reader.dispose()


def test_stale_access_and_wrong_database_login_are_denied(make_workspace, workspace_database):
    service, access, state, _, reader, writer = make_workspace("alice")
    frame = pd.DataFrame({"value": [1]})
    dataset = service.upload(frame, "private.csv")

    for stale in (None, replace(access, principal_id="bob"), replace(access, session_id=uuid4()),
                  replace(access, workspace_id=uuid4()), replace(access, can_manage=False)):
        state.current = stale
        for operation in (service.metadata, service.preview, service.delete):
            with pytest.raises(PermissionError):
                operation(dataset.version_id)
        with pytest.raises(PermissionError):
            service.upload(frame, "stale.csv")
        with pytest.raises(PermissionError):
            service.list_datasets()

    view_only = replace(access, can_manage=False)
    state.current = view_only
    viewer = WorkspaceDatasets(view_only, lambda: state.current, reader, writer)
    assert viewer.preview(dataset.version_id) == [{"value": 1}]
    with pytest.raises(PermissionError):
        viewer.delete(dataset.version_id)
    with pytest.raises(PermissionError):
        viewer.upload(frame, "viewer.csv")

    state.current = access
    _, _, _, _, foreign_reader, foreign_writer = make_workspace("bob")
    for wrong_reader, wrong_writer in ((foreign_reader, foreign_writer), (workspace_database, workspace_database)):
        wrong_login = WorkspaceDatasets(access, lambda: state.current, wrong_reader, wrong_writer)
        with pytest.raises(PermissionError):
            wrong_login.preview(dataset.version_id)
        with pytest.raises(PermissionError):
            wrong_login.upload(frame, "wrong.csv")
    assert service.metadata(dataset.version_id) == dataset


def test_data_and_catalog_roll_back_together(make_workspace, workspace_database):
    service, _, _, schema, _, _ = make_workspace("alice")
    frame = pd.DataFrame({"value": [5]})
    original = service.upload(frame, "dataset_versions.csv")
    table = f"v_{original.version_id.hex}"
    with workspace_database.begin() as connection:
        connection.exec_driver_sql(
            f"ALTER TABLE {schema}.dataset_versions ADD CHECK (filename <> 'rejected.csv')"
        )
        connection.exec_driver_sql(f"CREATE VIEW {schema}.dependency AS SELECT * FROM {schema}.{table}")

    with pytest.raises(IntegrityError):
        service.upload(frame, "rejected.csv")
    assert set(inspect(workspace_database).get_table_names(schema=schema)) == {
        "dataset_versions", "upload_receipts", table,
    }

    with pytest.raises(DBAPIError) as error:
        service.delete(original.version_id)
    assert error.value.orig.pgcode == "2BP01"
    assert service.metadata(original.version_id) == original
    assert service.preview(original.version_id) == [{"value": 5}]


def test_twenty_uploads_contend_for_the_last_daily_slot(make_workspace, workspace_database):
    service, _, _, schema, _, _ = make_workspace("alice")
    payload = b"value\n1\n"
    for _ in range(WORKSPACE_UPLOADS_PER_DAY - 1):
        service.upload_csv(payload, "existing.csv", uuid4())
    start = Barrier(20)

    def submit(request_id):
        start.wait(timeout=10)
        try:
            return service.upload_csv(payload, "contender.csv", request_id)
        except UploadLimitReached:
            return None

    ids = [uuid4() for _ in range(20)]
    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(submit, ids))
    assert sum(result is not None for result in results) == 1
    assert len(service.list_datasets()) == WORKSPACE_UPLOADS_PER_DAY
    winner_id, winner = next((key, result) for key, result in zip(ids, results) if result is not None)
    assert service.upload_csv(payload, "contender.csv", winner_id) == winner
    service.delete(winner.version_id)
    with pytest.raises(UploadLimitReached):
        service.upload_csv(payload, "after-delete.csv", uuid4())

    other, *_ = make_workspace("bob")
    assert other.upload_csv(payload, "other.csv", uuid4()).row_count == 1
    with workspace_database.begin() as connection:
        connection.exec_driver_sql(f"UPDATE {schema}.upload_receipts SET admitted_on = admitted_on - 1")
    assert service.upload_csv(payload, "next-day.csv", uuid4()).row_count == 1


def test_concurrent_retries_publish_one_version_and_do_not_recreate_deleted_data(make_workspace):
    service, *_ = make_workspace("alice")
    request_id = uuid4()
    payload = b"value\n1\n"
    start = Barrier(2)

    def submit():
        start.wait(timeout=10)
        return service.upload_csv(payload, "sample.csv", request_id)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = (executor.submit(submit) for _ in range(2))
        dataset = first.result()
        assert second.result() == dataset
    assert service.list_datasets() == (dataset,)
    with pytest.raises(UploadRequestConflict, match="different file"):
        service.upload_csv(b"value\n2\n", "sample.csv", request_id)
    service.delete(dataset.version_id)
    with pytest.raises(UploadRequestConflict, match="deleted"):
        service.upload_csv(payload, "sample.csv", request_id)
    assert service.list_datasets() == ()


def test_failed_receipt_rolls_back_dataset_and_leaves_slot_for_retry(make_workspace, workspace_database):
    service, _, _, schema, _, _ = make_workspace("alice")
    payload = b"value\n1\n"
    for _ in range(WORKSPACE_UPLOADS_PER_DAY - 1):
        service.upload_csv(payload, "existing.csv", uuid4())
    existing = service.list_datasets()
    request_id = uuid4()
    with workspace_database.begin() as connection:
        connection.exec_driver_sql(
            f"ALTER TABLE {schema}.upload_receipts ADD CONSTRAINT fail_receipt CHECK (request_id <> %s::uuid)",
            (str(request_id),),
        )
    tables = set(inspect(workspace_database).get_table_names(schema=schema))
    with pytest.raises(IntegrityError):
        service.upload_csv(payload, "retry.csv", request_id)
    assert service.list_datasets() == existing
    assert set(inspect(workspace_database).get_table_names(schema=schema)) == tables

    with workspace_database.begin() as connection:
        connection.exec_driver_sql(f"ALTER TABLE {schema}.upload_receipts DROP CONSTRAINT fail_receipt")
    assert service.upload_csv(payload, "retry.csv", request_id).row_count == 1
    with pytest.raises(UploadLimitReached):
        service.upload_csv(payload, "over-limit.csv", uuid4())


def test_preview_csv_is_bounded_and_uses_current_access(make_workspace):
    service, access, state, *_ = make_workspace("alice")
    dataset = service.upload(
        pd.DataFrame({"=header": [" 	=SUM(1,2)", "café"] + ["text"] * 50, "number": range(-5, 47)}),
        "preview.csv",
    )
    output = service.preview_csv(dataset.version_id).decode("utf-8-sig")
    rows = list(csv.reader(StringIO(output)))
    assert len(rows) == 51
    assert rows[:3] == [["'=header", "number"], ["' 	=SUM(1,2)", "-5"], ["café", "-4"]]

    state.current = None
    with pytest.raises(PermissionError):
        service.preview_csv(dataset.version_id)
    state.current = replace(access, session_id=uuid4())
    with pytest.raises(PermissionError):
        service.preview_csv(dataset.version_id)


def test_catalog_search_matches_literal_metadata_in_the_current_workspace(make_workspace):
    alice, _, state, *_ = make_workspace("alice")
    bob, *_ = make_workspace("bob")
    local = alice.upload(pd.DataFrame({"Total %_Due": ["foreign-row-token"]}), "Sales.csv")
    alice.upload(pd.DataFrame({"Total xxDue": [1], "Suffix": [2], "Prefix": [3]}), "Other.csv")
    foreign = bob.upload(pd.DataFrame({"Confidential margin": [1]}), "Hidden.csv")

    assert alice.list_datasets(" sALeS ") == (local,)
    assert alice.list_datasets("total %_due") == (local,)
    assert alice.list_datasets("suffix,prefix") == ()
    assert alice.list_datasets("foreign-row-token") == ()
    assert alice.list_datasets("hidden") == ()
    assert alice.list_datasets("confidential margin") == ()
    assert bob.list_datasets("confidential margin") == (foreign,)
    assert bob.list_datasets("total %_due") == ()
    assert alice.list_datasets(" ") == alice.list_datasets()

    alice.delete(local.version_id)
    assert alice.list_datasets("total %_due") == ()
    assert bob.list_datasets("confidential margin") == (foreign,)
    state.current = None
    with pytest.raises(PermissionError):
        alice.list_datasets("other")


def test_preview_treats_concurrent_deletion_as_unavailable(make_workspace):
    service, _, _, schema, reader, _ = make_workspace("alice")
    dataset = service.upload(pd.DataFrame({"value": [5]}), "private.csv")
    relation = f"{schema}.v_{dataset.version_id.hex}"

    @event.listens_for(reader, "before_cursor_execute")
    def delete_before_preview(connection, cursor, statement, parameters, context, executemany):
        if f"FROM {relation}" in statement:
            service.delete(dataset.version_id)

    with pytest.raises(PermissionError, match="This dataset version is unavailable"):
        service.preview(dataset.version_id)
    with pytest.raises(PermissionError):
        service.delete(dataset.version_id)


def test_private_analysis_resolves_types_and_enforces_workspace_and_row_bounds(
    make_workspace, workspace_database,
):
    alice, _, state, schema, *_ = make_workspace("alice")
    bob, *_ = make_workspace("bob")
    local = alice.upload(pd.DataFrame({
        "Amount total": [-2.5, None, 7.5],
        "Name": ["a", None, "b"],
        "Flag": [True, False, True],
        "Exact": [9007199254740993, 1, 1],
    }), "local.csv")
    foreign = bob.upload(pd.DataFrame({"Amount total": [42]}), "foreign.csv")
    empty = alice.upload(pd.DataFrame({"Value": pd.Series(dtype=float)}), "empty.csv")

    count = WorkspaceRowCount(version_id=local.version_id)
    assert alice.summarize(count) == {"row_count": 3}
    assert alice.summarize(WorkspaceColumnSummary(version_id=local.version_id, column="Amount total")) == {
        "row_count": 3, "non_null_count": 2, "null_count": 1,
        "minimum": -2.5, "maximum": 7.5, "total": 5.0, "mean": 2.5,
    }
    for column, non_null in (("Name", 2), ("Flag", 3)):
        assert alice.summarize(WorkspaceColumnSummary(version_id=local.version_id, column=column)) == {
            "row_count": 3, "non_null_count": non_null, "null_count": 3 - non_null,
        }
    exact = alice.summarize(WorkspaceColumnSummary(version_id=local.version_id, column="Exact"))
    assert exact["total"] == 9007199254740995
    assert alice.summarize(WorkspaceColumnSummary(version_id=empty.version_id, column="Value")) == {
        "row_count": 0, "non_null_count": 0, "null_count": 0,
        "minimum": None, "maximum": None, "total": None, "mean": None,
    }
    with pytest.raises(ValidationError):
        WorkspaceRowCount.model_validate({"version_id": local.version_id, "sql": "unsupported"})
    for column in (ROW_KEY, "Not a column"):
        with pytest.raises(PermissionError):
            alice.summarize(WorkspaceColumnSummary(version_id=local.version_id, column=column))
    with pytest.raises(PermissionError):
        alice.summarize(WorkspaceRowCount(version_id=foreign.version_id))
    with pytest.raises(PermissionError):
        bob.summarize(WorkspaceColumnSummary(version_id=local.version_id, column="Amount total"))
    alice.delete(empty.version_id)
    with pytest.raises(PermissionError):
        alice.summarize(WorkspaceRowCount(version_id=empty.version_id))

    with workspace_database.begin() as connection:
        connection.exec_driver_sql(
            f"INSERT INTO {schema}.v_{local.version_id.hex} ({ROW_KEY}) "
            "SELECT value FROM generate_series(3, %s) AS value", (MAX_UPLOAD_ROWS,),
        )
    with pytest.raises(ValueError, match="analysis row limit"):
        alice.summarize(count)
    state.current = None
    with pytest.raises(PermissionError):
        alice.summarize(count)


def _workspace_app(lookup, engines):
    from ui.private_workspace import render_private_workspace

    render_private_workspace(lookup, engines)


def test_pending_delete_cannot_survive_membership_change(
    make_workspace, workspace_database, membership_reader, monkeypatch,
):
    service, access, _, _, reader, writer = make_workspace("alice")
    dataset = service.upload(pd.DataFrame({"value": [1]}), "private.csv")
    duplicate = service.upload(pd.DataFrame({"other": [99]}), "private.csv")
    claims = {"is_logged_in": True, "iss": "https://issuer.example", "sub": "alice", "exp": int(time()) + 60}
    monkeypatch.setattr(st, "user", SimpleNamespace(to_dict=lambda: dict(claims), is_logged_in=True))
    registry = MembershipRegistry(membership_reader)
    with workspace_database.begin() as connection:
        grant_membership(connection, claims["iss"], claims["sub"], access.workspace_id, can_manage=True)

    def engines(workspace_id):
        assert workspace_id == access.workspace_id
        return reader, writer

    app = AppTest.from_function(_workspace_app, args=(registry.lookup, engines)).run()
    duplicate_label = f"{duplicate.filename} · {duplicate.version_id}"
    assert set(app.selectbox[0].options) == {
        f"{dataset.filename} · {dataset.version_id}", duplicate_label,
    }
    app.selectbox[0].set_value(duplicate_label).run()
    assert "<td>99</td>" in app.get("html")[0].proto.body
    app.selectbox[0].set_value(f"{dataset.filename} · {dataset.version_id}").run()
    app.text_input[0].set_value("no match").run()
    assert not app.selectbox and not app.get("html")
    assert app.info[0].value == "No matching datasets."
    app.text_input[0].set_value("VaLuE").run()
    assert app.selectbox[0].value is None
    app.selectbox[0].set_value(f"{dataset.filename} · {dataset.version_id}").run()
    next(button for button in app.button if button.label == "Run analysis").click().run()
    assert [item.value for item in app.text] == ["Rows: 1"]
    app.radio[0].set_value("Summarize column").run()
    next(button for button in app.button if button.label == "Run analysis").click().run()
    assert "Sum: 1" in [item.value for item in app.text]
    next(button for button in app.button if button.label == "Prepare preview CSV").click().run()
    assert len(app.get("iframe")) == 1
    next(button for button in app.button if button.label == "Delete dataset").click().run()
    confirm = next(button for button in app.button if button.label == "Confirm deletion")

    with workspace_database.begin() as connection:
        grant_membership(connection, claims["iss"], claims["sub"], access.workspace_id, can_manage=False)
    confirm.click().run()
    assert not app.exception
    assert not app.get("html") and not app.get("iframe")
    assert not app.get("file_uploader")
    assert all(button.label != "Confirm deletion" for button in app.button)
    assert service.metadata(dataset.version_id) == dataset

    app.selectbox[0].set_value(f"{dataset.filename} · {dataset.version_id}").run()
    assert len(app.get("html")) == 1
    prepare = next(button for button in app.button if button.label == "Prepare preview CSV")
    claims["exp"] = int(time()) - 1
    prepare.click().run()
    assert not app.get("html") and not app.get("iframe")
    assert not app.selectbox
