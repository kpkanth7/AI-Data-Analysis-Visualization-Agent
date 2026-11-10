"""Run the demo readers against a fresh PostgreSQL cluster on a private socket."""

import shutil
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.pool import NullPool
from streamlit.testing.v1 import AppTest

from db.demo_catalog import DEMO_READER, DEMO_WORKSPACE, ROW_KEY, load_demo_versions
from db.demo_queries import DemoAccess, DemoCount, DemoFilter, DemoPreview, DemoQueries, DemoSort
from scripts.stage_demo_catalog import grant_demo_access, revoke_demo_access, stage_demo_catalog


APP_DIR = Path(__file__).resolve().parents[1]
DEMOS = {"titanic": 891, "sp500_financials": 503, "world_co2_emissions": 8617}


@pytest.fixture(scope="module")
def demo_postgres():
    initdb, pg_ctl = (shutil.which(name) for name in ("initdb", "pg_ctl"))
    if initdb is None or pg_ctl is None:
        pytest.skip("Local PostgreSQL initdb and pg_ctl are required")

    with tempfile.TemporaryDirectory(prefix="w1-demo-pg-", dir="/tmp") as socket_dir:
        cluster = Path(socket_dir) / "data"
        initialized = subprocess.run(
            [initdb, "-D", str(cluster), "-A", "trust", "-U", "w1_owner", "--no-locale", "-E", "UTF8"],
            capture_output=True, text=True, timeout=30,
        )
        if initialized.returncode and "Operation not permitted" in initialized.stderr:
            pytest.skip("Local PostgreSQL needs shared memory outside this sandbox")
        assert initialized.returncode == 0, initialized.stderr
        subprocess.run(
            [pg_ctl, "-D", str(cluster), "-l", str(Path(socket_dir) / "postgres.log"),
             "-o", f"-c listen_addresses='' -c unix_socket_directories={socket_dir} -c unix_socket_permissions=0700",
             "-w", "start"],
            check=True, capture_output=True, text=True, timeout=30,
        )
        url = URL.create("postgresql+psycopg2", database="postgres", query={"host": socket_dir})
        owner = create_engine(url.set(username="w1_owner"), poolclass=NullPool)
        reader = create_engine(url.set(username=DEMO_READER), poolclass=NullPool)
        try:
            with owner.begin() as conn:
                conn.exec_driver_sql(f"CREATE ROLE {DEMO_READER} LOGIN NOINHERIT NOSUPERUSER NOBYPASSRLS NOCREATEROLE NOCREATEDB")
                conn.exec_driver_sql("REVOKE TEMPORARY ON DATABASE postgres FROM PUBLIC")
                conn.exec_driver_sql("REVOKE CREATE ON SCHEMA public FROM PUBLIC")
                conn.exec_driver_sql("CREATE SCHEMA private_data")
                conn.exec_driver_sql("""
                    CREATE TABLE public.datasets (
                        name text, slug text, row_count integer, columns_json jsonb,
                        is_demo boolean, owner_only boolean, session_id text, uploaded_at timestamptz
                    )
                """)
                conn.exec_driver_sql("CREATE TABLE public.legacy_sample (id integer PRIMARY KEY)")
                conn.exec_driver_sql("INSERT INTO public.legacy_sample VALUES (42)")
                conn.exec_driver_sql("""
                    INSERT INTO public.datasets VALUES (
                        'Legacy sample', 'legacy_sample', 1, '[{"name":"id","dtype":"int64"}]',
                        true, false, null, now()
                    )
                """)
                conn.exec_driver_sql("CREATE TABLE public.guest_usage (id integer)")
                conn.exec_driver_sql("CREATE TABLE private_data.uploads (id integer)")
                stage_demo_catalog(conn)
                conn.exec_driver_sql("CREATE TABLE demo.unpublished_version (id integer)")
                grant_demo_access(conn)
            with reader.connect() as conn:
                versions = load_demo_versions(conn)
            yield owner, reader, versions
        finally:
            reader.dispose()
            owner.dispose()
            subprocess.run(
                [pg_ctl, "-D", str(cluster), "-m", "fast", "-w", "stop"],
                check=True, capture_output=True, text=True, timeout=30,
            )


@pytest.fixture
def demo_queries(demo_postgres):
    _, reader, versions = demo_postgres
    return DemoQueries(reader, DemoAccess("test_viewer", DEMO_WORKSPACE), versions)


def test_demo_queries_validate_access_and_bind_filter_values(demo_postgres):
    _, reader, versions = demo_postgres
    titanic = versions[0]
    foreign = replace(titanic, version_id=uuid4(), workspace_id="other_workspace")
    queries = DemoQueries(reader, DemoAccess("test_viewer", DEMO_WORKSPACE), (*versions, foreign))

    assert queries.versions == versions
    with pytest.raises(PermissionError):
        queries.execute(DemoCount(version_id=foreign.version_id))
    with pytest.raises(PermissionError):
        queries.execute(DemoPreview(version_id=titanic.version_id, columns=(ROW_KEY,)))
    with pytest.raises(PermissionError):
        queries.execute(DemoCount(version_id=titanic.version_id, filters=(DemoFilter(column="Name) OR TRUE --", value=0),)))
    with pytest.raises(ValidationError):
        DemoPreview(version_id=titanic.version_id, limit=101)
    with pytest.raises(ValidationError):
        DemoCount.model_validate({"version_id": titanic.version_id, "sql": "SELECT 1"})

    malicious_value = DemoFilter(column="Name", value="' OR TRUE --")
    assert queries.execute(DemoCount(version_id=titanic.version_id, filters=(malicious_value,))) == [{"row_count": 0}]
    last_passengers = DemoFilter(column="PassengerId", comparison="ge", value=888)
    assert queries.execute(DemoCount(version_id=titanic.version_id, filters=(last_passengers,))) == [{"row_count": 4}]
    request = DemoPreview(
        version_id=titanic.version_id, columns=("PassengerId",), filters=(last_passengers,),
        sort=(DemoSort(column="PassengerId", descending=True),), limit=2, offset=1,
    )
    assert queries.execute(request) == [{"PassengerId": 890}, {"PassengerId": 889}]


def test_postgres_reader_denies_other_relations_and_writes(demo_postgres):
    _, reader, _ = demo_postgres
    with reader.connect() as conn:
        role = conn.exec_driver_sql("""
            SELECT current_user, rolsuper, rolbypassrls, rolcreaterole, rolcreatedb,
                   pg_has_role(current_user, 'w1_owner', 'MEMBER')
            FROM pg_roles WHERE rolname = current_user
        """).one()
        assert tuple(role) == (DEMO_READER, False, False, False, False, False)
        assert conn.exec_driver_sql("SELECT count(*) FROM demo.titanic").scalar_one() == DEMOS["titanic"]

    for statement in (
        "SELECT * FROM demo.unpublished_version",
        "SELECT * FROM private_data.uploads",
        "SELECT * FROM public.datasets",
        "SELECT * FROM public.guest_usage",
        'UPDATE demo.titanic SET "PassengerId" = 0',
        "TRUNCATE demo.titanic",
        "CREATE TABLE demo.unauthorized (id integer)",
        "CREATE TEMP TABLE unauthorized (id integer)",
        "SET ROLE w1_owner",
    ):
        with reader.connect() as conn, pytest.raises(ProgrammingError) as error:
            conn.exec_driver_sql(statement)
        assert error.value.orig.pgcode == "42501", statement


def _demo_tabs(queries):
    from ui.demo_tabs import render_demo_chat, render_demo_explorer

    render_demo_chat(queries)
    render_demo_explorer(queries)


@pytest.mark.parametrize("demo", DEMOS)
def test_chat_and_explorer_agree_on_demo_results(demo, demo_queries, monkeypatch):
    from ui import chat, explorer

    def no_legacy_path(*args, **kwargs):
        pytest.fail("Demo tabs must use the injected demo query service")

    monkeypatch.setattr(chat, "run_agent", no_legacy_path)
    monkeypatch.setattr(chat, "list_datasets_from_db", no_legacy_path)
    monkeypatch.setattr(explorer, "list_datasets_from_db", no_legacy_path)
    monkeypatch.setattr(explorer, "get_connection", no_legacy_path)

    version = next(item for item in demo_queries.versions if item.label == demo)
    app = AppTest.from_function(_demo_tabs, args=(demo_queries,)).run(timeout=15)
    app.selectbox(key="demo_chat_dataset").set_value(version.version_id)
    app.selectbox(key="demo_explorer_dataset").set_value(version.version_id)
    app.chat_input(key="demo_chat_input").set_value(f"How many rows are in {demo}?").run(timeout=15)
    assert not app.exception and not app.error
    assert f"{demo} has {DEMOS[demo]:,} rows." in [item.value for item in app.markdown]
    assert f"{DEMOS[demo]:,} matching rows" in [item.value for item in app.caption]

    app.chat_input(key="demo_chat_input").set_value("preview").run(timeout=15)
    assert not app.exception and not app.error
    assert len(app.dataframe) == 2
    pd.testing.assert_frame_equal(app.dataframe[0].value, app.dataframe[1].value.head(10).reset_index(drop=True))

    if demo == "titanic":
        app.selectbox(key=f"demo_filter_{version.version_id}").set_value("PassengerId").run(timeout=15)
        value_key = f"demo_filter_value_{version.version_id}_PassengerId"
        app.text_input(key=value_key).set_value("891").run(timeout=15)
        assert not app.exception and not app.error
        assert [item.value for item in app.caption] == ["1 matching row"]
        assert app.dataframe[0].value["PassengerId"].tolist() == [891]


def test_demo_cutover_and_rollback_preserve_legacy_data(demo_postgres, monkeypatch):
    from db import postgres
    from ui import auth

    owner, reader, _ = demo_postgres
    monkeypatch.setattr(postgres, "get_engine", lambda: owner)
    monkeypatch.setattr(auth, "get_browser_fingerprint", lambda: "w1_test_guest")
    monkeypatch.setattr(auth, "get_guest_limits", lambda: {
        "queries_remaining": 10, "uploads_remaining": 5, "can_query": True, "can_upload": True,
    })

    def legacy_preview():
        app = AppTest.from_file(str(APP_DIR / "legacy_app.py")).run(timeout=15)
        assert not app.exception and not app.error
        assert app.selectbox(key="explorer_table_select").value == "legacy_sample"
        return app.dataframe[-1].value.to_dict(orient="records")

    original = legacy_preview()
    assert original == [{"id": 42}]
    demo = AppTest.from_file(str(APP_DIR / "app.py"))
    demo.secrets["demo_reader"] = {"url": reader.url.render_as_string(hide_password=False)}
    demo.run(timeout=15)
    assert not demo.exception and not demo.error
    assert [tab.label for tab in demo.tabs] == ["Explorer", "Chat"]
    assert not demo.get("file_uploader") and not demo.get("download_button")
    assert "891 matching rows" in [item.value for item in demo.caption]
    demo.chat_input(key="demo_chat_input").set_value("How many rows are in titanic?").run(timeout=15)
    assert "titanic has 891 rows." in [item.value for item in demo.markdown]

    try:
        with owner.begin() as conn:
            revoke_demo_access(conn)
        demo.run(timeout=15)
        assert not demo.exception and not demo.tabs and not demo.dataframe
        assert "temporarily unavailable" in demo.error[0].value
        assert legacy_preview() == original
        with owner.connect() as conn:
            assert conn.exec_driver_sql("SELECT count(*) FROM demo.titanic").scalar_one() == 891
    finally:
        with owner.begin() as conn:
            grant_demo_access(conn)
    demo.button(key="retry_demo_startup").click().run(timeout=15)
    assert not demo.exception and not demo.error and len(demo.tabs) == 2


def test_demo_entrypoint_rejects_wrong_missing_and_unavailable_login(demo_postgres, monkeypatch, caplog):
    from db import postgres

    owner, reader, _ = demo_postgres

    def no_legacy_connection(*args, **kwargs):
        pytest.fail("The demo entry point must never fall back to the legacy database")

    monkeypatch.setattr(postgres, "get_db_url", no_legacy_connection)
    monkeypatch.setattr(postgres, "get_engine", no_legacy_connection)
    for candidate in (None, owner.url, reader.url.set(database="private_connection_canary")):
        app = AppTest.from_file(str(APP_DIR / "app.py"))
        app.secrets["DATABASE_URL"] = owner.url.render_as_string(hide_password=False)
        if candidate is not None:
            app.secrets["demo_reader"] = {"url": candidate.render_as_string(hide_password=False)}
        app.run(timeout=15)
        assert not app.exception and not app.tabs and not app.dataframe
        assert len(app.error) == 1 and "temporarily unavailable" in app.error[0].value
        assert "private_connection_canary" not in app.error[0].value
    assert "private_connection_canary" not in caplog.text


def test_demo_staging_refuses_to_replace_existing_tables(demo_postgres):
    owner, _, _ = demo_postgres
    with pytest.raises(ProgrammingError) as error, owner.begin() as conn:
        stage_demo_catalog(conn)
    assert error.value.orig.pgcode == "42P06"
    with owner.connect() as conn:
        assert conn.exec_driver_sql("SELECT count(*) FROM demo.titanic").scalar_one() == 891
        assert conn.exec_driver_sql("SELECT id FROM public.legacy_sample").scalar_one() == 42
