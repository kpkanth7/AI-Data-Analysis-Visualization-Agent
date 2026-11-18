"""Run the demo readers against a fresh PostgreSQL cluster on a private socket."""

import csv
import json
import shutil
import subprocess
import sys
import tempfile
from base64 import b64decode
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from io import StringIO
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace
from uuid import uuid4

import pandas as pd
import pytest
from pydantic import ValidationError
from sqlalchemy import Column, Integer, MetaData, Table, Text, create_engine, event, func
from sqlalchemy.engine import URL
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.pool import NullPool
from streamlit.testing.v1 import AppTest

from db.demo_catalog import DEMO_READER, DEMO_WORKSPACE, ROW_KEY, load_demo_versions
from db.demo_queries import (
    DEMO_PREVIEW_MAX_BYTES, DemoAccess, DemoCount, DemoCountTask, DemoEvidenceFilter, DemoFilter,
    DemoPlan, DemoPreview, DemoPreviewTask, DemoPreviewTooLarge, DemoQueries, DemoSort, DemoVersion,
)
from scripts.stage_demo_catalog import grant_demo_access, revoke_demo_access, stage_demo_catalog


APP_DIR = Path(__file__).resolve().parents[1]
DEMOS = {"titanic": 891, "sp500_financials": 503, "world_co2_emissions": 8617}
DEMO_BRIEFS = {
    "titanic": "One row per passenger, with ticket, travel, and survival details.",
    "sp500_financials": "One row per stock symbol, with company, sector, and financial metrics.",
    "world_co2_emissions": "One row per country or region and year, with emissions and related indicators.",
}


@pytest.fixture(scope="module")
def demo_postgres(synthetic_demo_files):
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


@pytest.fixture(scope="module")
def wide_demo(demo_postgres):
    owner, reader, _ = demo_postgres
    table = Table(
        "wide_preview", MetaData(), Column("id", Integer, primary_key=True),
        Column("note", Text), Column("score", Integer), schema="demo",
    )
    exact = "é" * 32_761 + "a"
    with owner.begin() as connection:
        table.create(connection)
        connection.execute(table.insert(), [
            {"id": 1, "note": exact, "score": 1},
            {"id": 2, "note": exact + '"', "score": 2},
            {"id": 3, "note": "a" * 36_000, "score": 3},
            {"id": 4, "note": "b" * 36_000, "score": 3},
            {"id": 5, "note": "é" * 524_288, "score": 5},
            {"id": 6, "note": None, "score": 6},
        ])
        connection.exec_driver_sql(f"GRANT SELECT ON demo.wide_preview TO {DEMO_READER}")
    version = DemoVersion(uuid4(), DEMO_WORKSPACE, "Wide fixture", table, ("note", "score"), "id")
    return DemoQueries(reader, DemoAccess("wide_viewer", DEMO_WORKSPACE), (version,))


def test_demo_preview_limits_server_payload_bytes_with_original_order_and_types(demo_postgres, wide_demo):
    _, reader, _ = demo_postgres
    version = wide_demo.versions[0].version_id

    def notes(score, **options):
        return DemoPreview(
            version_id=version, filters=(DemoFilter(column="score", value=score),),
            columns=("note",), **options,
        )

    boundary = wide_demo.execute(notes(1))
    assert len(json.dumps(boundary, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) == DEMO_PREVIEW_MAX_BYTES
    with pytest.raises(DemoPreviewTooLarge, match="result size limit"):
        wide_demo.execute(notes(2))
    with pytest.raises(DemoPreviewTooLarge):
        wide_demo.execute(notes(3))
    assert wide_demo.execute(notes(3, limit=1, offset=1))[0]["note"].startswith("b")
    assert wide_demo.execute(notes(6)) == [{"note": None}]
    assert wide_demo.execute(notes(7)) == []
    assert wide_demo.execute(DemoPreview(
        version_id=version, columns=("score",), sort=(DemoSort(column="note", descending=True),), limit=2,
    )) == [{"score": 6}, {"score": 5}]

    with reader.begin() as connection:
        # The cursor receives the size sentinel only, even for a one-row 1 MiB cell.
        row = connection.execute(wide_demo._statement(notes(5))).one()
    assert row[0] is None and row[1] == 1_048_589


def test_demo_over_budget_preview_keeps_independent_count_and_blocks_dependents(wide_demo):
    version = wide_demo.versions[0].version_id
    count = DemoCount(version_id=version)
    preview = DemoPreview(version_id=version, columns=("score", "note"))
    plan = DemoPlan(tasks=(
        DemoPreviewTask(task_id="source", operation="preview", request=preview),
        DemoCountTask(task_id="independent", operation="count", request=count),
        DemoCountTask(
            task_id="dependent", operation="count", request=count, depends_on=("source",),
            evidence_filters=(DemoEvidenceFilter(
                column="score", source_task_id="source", source_column="score",
            ),),
        ),
    ))
    source, independent, dependent = wide_demo.run_plan(plan, max_calls=2)
    assert (source.status, source.reason, source.result) == ("failed", "result_limit", None)
    assert independent.status == "success" and independent.result.rows == wide_demo.execute(count)
    assert dependent.status == "skipped_dependency" and dependent.result is None
    with pytest.raises(DemoPreviewTooLarge):
        wide_demo.execute_plan(plan)


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


@pytest.mark.parametrize(
    ("specifications", "error"),
    [
        ((), "at least 1"),
        (tuple((f"task_{index}", ()) for index in range(5)), "at most 4"),
        ((("count", ()), ("count", ())), "Task IDs must be unique"),
        ((("count", ("missing",)),), "Every dependency must name a task"),
        ((("count", ("count",)),), "dependencies must be acyclic"),
        ((("first", ("second",)), ("second", ("first",))), "dependencies must be acyclic"),
        ((("first", ()), ("second", ("first", "first"))), "Dependencies must be unique"),
    ],
)
def test_demo_plan_rejects_invalid_dependencies(specifications, error):
    count = DemoCount(version_id=uuid4())
    tasks = tuple(
        DemoCountTask(task_id=task_id, operation="count", request=count, depends_on=depends_on)
        for task_id, depends_on in specifications
    )
    with pytest.raises(ValidationError, match=error):
        DemoPlan(tasks=tasks)


def test_demo_plan_executes_typed_requests_in_dependency_order(demo_queries):
    titanic, financials, _ = demo_queries.versions
    count = DemoCountTask(
        task_id="passengers", operation="count", request=DemoCount(version_id=titanic.version_id),
    )
    preview = DemoPreviewTask(
        task_id="passenger_preview", operation="preview", depends_on=("passengers",),
        request=DemoPreview(version_id=titanic.version_id, columns=("PassengerId",), limit=2),
    )
    independent = DemoCountTask(
        task_id="symbols", operation="count", request=DemoCount(version_id=financials.version_id),
    )
    plan = DemoPlan.model_validate_json(DemoPlan(tasks=(preview, independent, count)).model_dump_json())
    results = demo_queries.execute_plan(plan)
    assert [result.task_id for result in results] == ["passengers", "symbols", "passenger_preview"]
    assert [result.request for result in results] == [count.request, independent.request, preview.request]
    assert [result.rows for result in results] == [
        [{"row_count": 891}], [{"row_count": 503}], [{"PassengerId": 1}, {"PassengerId": 2}],
    ]
    with pytest.raises(ValidationError, match="requires a count request"):
        DemoCountTask(task_id="wrong_shape", operation="count", request=preview.request)


def test_demo_plan_checks_all_dataset_and_column_access_before_execution(demo_queries, monkeypatch):
    def unexpected_statement(*args, **kwargs):
        pytest.fail("Plan authorization must finish before executing any statement")

    monkeypatch.setattr(demo_queries, "_execute_statement", unexpected_statement)
    version = demo_queries.versions[0].version_id
    allowed = DemoCountTask(task_id="count", operation="count", request=DemoCount(version_id=version))
    for request, message in (
        (DemoPreview(version_id=uuid4()), "dataset version is unavailable"),
        (DemoPreview(version_id=version, columns=(ROW_KEY,)), "column is unavailable"),
    ):
        denied = DemoPreviewTask(
            task_id="preview", operation="preview", request=request, depends_on=("count",),
        )
        with pytest.raises(PermissionError, match=message):
            demo_queries.execute_plan(DemoPlan(tasks=(allowed, denied)))
        with pytest.raises(PermissionError, match=message):
            demo_queries.run_plan(DemoPlan(tasks=(allowed, denied)))
        with pytest.raises(PermissionError, match=message):
            demo_queries.run_plan(DemoPlan(tasks=(allowed, denied)), max_calls=0, cancelled=lambda: True)


def test_demo_plan_binds_count_and_preview_evidence(demo_queries):
    version = demo_queries.versions[0].version_id
    source = DemoCountTask(
        task_id="count", operation="count", request=DemoCount(
            version_id=version, filters=(DemoFilter(column="PassengerId", comparison="le", value=3),),
        ),
    )
    upper_bound = DemoFilter(column="PassengerId", comparison="le", value=5)
    preview = DemoPreviewTask(
        task_id="preview", operation="preview", depends_on=("count",),
        request=DemoPreview(version_id=version, columns=("PassengerId",), filters=(upper_bound,), limit=2),
        evidence_filters=(DemoEvidenceFilter(
            column="PassengerId", comparison="ge", source_task_id="count", source_column="row_count",
        ),),
    )
    dependent = DemoCountTask(
        task_id="dependent", operation="count", depends_on=("preview",),
        request=DemoCount(version_id=version),
        evidence_filters=(DemoEvidenceFilter(
            column="PassengerId", comparison="le", source_task_id="preview",
            source_column="PassengerId", row_index=1,
        ),),
    )
    plan = DemoPlan.model_validate_json(DemoPlan(tasks=(dependent, preview, source)).model_dump_json())
    count_result, preview_result, dependent_result = demo_queries.execute_plan(plan)
    assert count_result.rows == [{"row_count": 3}]
    assert preview_result.rows == [{"PassengerId": 3}, {"PassengerId": 4}]
    assert preview_result.request.filters == (
        upper_bound, DemoFilter(column="PassengerId", comparison="ge", value=3),
    )
    assert dependent_result.rows == [{"row_count": 4}]
    assert dependent_result.request.filters == (DemoFilter(column="PassengerId", comparison="le", value=4),)


@pytest.mark.parametrize(
    ("operation", "depends_on", "source_column", "row_index", "target_column", "literal_count"),
    [
        ("preview", (), "PassengerId", 0, "PassengerId", 0),
        ("preview", ("source",), "Name", 0, "PassengerId", 0),
        ("preview", ("source",), ROW_KEY, 0, "PassengerId", 0),
        ("preview", ("source",), "PassengerId", 1, "PassengerId", 0),
        ("preview", ("source",), "PassengerId", 0, ROW_KEY, 0),
        ("preview", ("source",), "PassengerId", 0, "PassengerId", 10),
        ("count", ("source",), "PassengerId", 0, "PassengerId", 0),
        ("count", ("source",), "row_count", 1, "PassengerId", 0),
    ],
)
def test_invalid_plan_evidence_fails_before_any_read(
    demo_queries, monkeypatch, operation, depends_on, source_column, row_index, target_column, literal_count,
):
    def unexpected_read(statement):
        pytest.fail("Invalid evidence must be rejected before reading any task")

    monkeypatch.setattr(demo_queries, "_execute_statement", unexpected_read)
    version = demo_queries.versions[0].version_id
    if operation == "preview":
        source = DemoPreviewTask(
            task_id="source", operation="preview",
            request=DemoPreview(version_id=version, columns=("PassengerId",), limit=1),
        )
    else:
        source = DemoCountTask(task_id="source", operation="count", request=DemoCount(version_id=version))
    binding = DemoEvidenceFilter(
        column=target_column, source_task_id="source", source_column=source_column, row_index=row_index,
    )
    request = DemoCount(version_id=version, filters=(DemoFilter(column="PassengerId", value=1),) * literal_count)
    with pytest.raises((ValueError, PermissionError)):
        dependent = DemoCountTask(
            task_id="dependent", operation="count", request=request, depends_on=depends_on,
            evidence_filters=(binding,),
        )
        demo_queries.execute_plan(DemoPlan(tasks=(source, dependent)))


@pytest.mark.parametrize("row_index", [-1, 100, True])
def test_demo_evidence_row_index_is_a_bounded_integer(row_index):
    with pytest.raises(ValidationError):
        DemoEvidenceFilter(column="PassengerId", source_task_id="source", source_column="row_count", row_index=row_index)


@pytest.mark.parametrize(
    "source_rows",
    [[], [{}], [{"PassengerId": None}], [{"PassengerId": {"private": "fixture-marker"}}],
     [{"PassengerId": "fixture-marker"}], [{"PassengerId": True}], [{"PassengerId": float("nan")}]],
)
def test_missing_or_invalid_evidence_stops_dependent_and_ignores_prior_run(demo_queries, monkeypatch, source_rows):
    version = demo_queries.versions[0].version_id
    source = DemoPreviewTask(
        task_id="source", operation="preview",
        request=DemoPreview(version_id=version, columns=("PassengerId",), limit=1),
    )
    dependent = DemoCountTask(
        task_id="dependent", operation="count", request=DemoCount(version_id=version),
        depends_on=("source",), evidence_filters=(
            DemoEvidenceFilter(column="PassengerId", source_task_id="source", source_column="PassengerId"),
        ),
    )
    plan = DemoPlan(tasks=(source, dependent))
    assert demo_queries.execute_plan(plan)[-1].rows == [{"row_count": 1}]
    calls = []

    def missing_evidence(statement):
        calls.append(statement)
        return source_rows

    monkeypatch.setattr(demo_queries, "_execute_statement", missing_evidence)
    with pytest.raises(ValueError) as failure:
        demo_queries.execute_plan(plan)
    assert len(calls) == 1
    assert "fixture-marker" not in str(failure.value)


def _count_plan(version, *specifications):
    return DemoPlan(tasks=tuple(
        DemoCountTask(
            task_id=task_id, operation="count", depends_on=depends_on, request=DemoCount(version_id=version),
        )
        for task_id, depends_on in specifications
    ))


def test_demo_run_overlaps_frozen_reads_before_binding_dependent_evidence(demo_queries, monkeypatch):
    titanic, financials, _ = demo_queries.versions
    source = DemoCountTask(
        task_id="source", operation="count", request=DemoCount(version_id=titanic.version_id),
    )
    independent = DemoCountTask(
        task_id="independent", operation="count", request=DemoCount(version_id=financials.version_id),
    )
    dependent = DemoPreviewTask(
        task_id="dependent", operation="preview", depends_on=("source",),
        request=DemoPreview(version_id=titanic.version_id, columns=("PassengerId",), limit=1),
        evidence_filters=(DemoEvidenceFilter(
            column="PassengerId", source_task_id="source", source_column="row_count",
        ),),
    )
    plan = DemoPlan(tasks=(source, independent, dependent))
    expected = demo_queries.execute_plan(plan)
    roots_started = Barrier(2)
    independent_finished = Event()
    source_finished = Event()
    execute = demo_queries._execute_statement

    def synchronized_read(statement, *, deadline):
        if "row_count" in statement.selected_columns:
            roots_started.wait(timeout=5)
            if statement.get_final_froms() == [titanic.relation]:
                assert independent_finished.wait(timeout=5)
                rows = execute(statement, deadline=deadline)
                source_finished.set()
                return rows
            rows = execute(statement, deadline=deadline)
            independent_finished.set()
            return rows
        assert source_finished.is_set()
        assert 891 in statement.compile().params.values()
        return execute(statement, deadline=deadline)

    monkeypatch.setattr(demo_queries, "_execute_statement", synchronized_read)
    outcomes = demo_queries.run_plan(plan)
    assert [outcome.status for outcome in outcomes] == ["success"] * 3
    assert tuple(outcome.result for outcome in outcomes) == expected
    assert outcomes[-1].result.rows == [{"PassengerId": 891}]


def test_demo_run_caps_workers_and_shares_calls_when_one_read_fails(demo_queries, monkeypatch):
    version = demo_queries.versions[0].version_id
    plan = DemoPlan(tasks=tuple(
        DemoCountTask(
            task_id=f"task_{number}", operation="count",
            request=DemoCount(version_id=version, filters=(DemoFilter(column="PassengerId", value=number),)),
        )
        for number in range(1, 5)
    ))
    initial_reads = Barrier(3)
    release_first = Event()
    fail_second = Event()
    third_started = Event()
    calls = []
    execute = demo_queries._execute_statement

    def synchronized_read(statement, *, deadline):
        (number,) = statement.compile().params.values()
        calls.append(number)
        if number in (1, 2):
            initial_reads.wait(timeout=5)
        if number == 1:
            assert release_first.wait(timeout=5)
        if number == 2:
            assert fail_second.wait(timeout=5)
            raise ProgrammingError(None, None, RuntimeError("fixture-marker"))
        if number == 3:
            third_started.set()
        return execute(statement, deadline=deadline)

    monkeypatch.setattr(demo_queries, "_execute_statement", synchronized_read)
    with ThreadPoolExecutor(max_workers=1) as caller:
        future = caller.submit(demo_queries.run_plan, plan, max_calls=3)
        try:
            initial_reads.wait(timeout=5)
            assert not third_started.wait(timeout=0.1)
            fail_second.set()
            assert third_started.wait(timeout=5)
        finally:
            fail_second.set()
            release_first.set()
        outcomes = future.result(timeout=5)

    assert sorted(calls) == [1, 2, 3]
    assert [(outcome.task_id, outcome.status) for outcome in outcomes] == [
        ("task_1", "success"), ("task_2", "failed"), ("task_3", "success"), ("task_4", "call_limit"),
    ]
    assert outcomes[0].result.rows == outcomes[2].result.rows == [{"row_count": 1}]
    assert outcomes[1].reason == "query_failed"
    assert outcomes[1].result is outcomes[3].result is None
    assert "fixture-marker" not in repr(outcomes)


@pytest.mark.parametrize("call_budget, independent_status", [(1, "call_limit"), (4, "success")])
def test_demo_run_counts_failed_calls_and_skips_their_dependents(demo_queries, monkeypatch, call_budget, independent_status):
    version = demo_queries.versions[0].version_id
    plan = _count_plan(version, ("first", ()), ("independent", ()),
                       ("child", ("first",)), ("grandchild", ("child",)))
    first, *remaining = plan.tasks
    request = DemoCount(version_id=version, filters=(DemoFilter(column="PassengerId", value=1),))
    plan = DemoPlan(tasks=(first.model_copy(update={"request": request}), *remaining))
    calls = []

    def fail_first(statement, **kwargs):
        calls.append(statement)
        if statement.compile().params:
            raise ProgrammingError(None, None, RuntimeError("fixture-marker"))
        return [{"row_count": 7}]

    monkeypatch.setattr(demo_queries, "_execute_statement", fail_first)
    outcomes = demo_queries.run_plan(plan, max_calls=call_budget)
    assert [(item.task_id, item.status) for item in outcomes] == [
        ("first", "failed"), ("independent", independent_status), ("child", "skipped_dependency"),
        ("grandchild", "skipped_dependency"),
    ]
    assert outcomes[0].reason == "query_failed"
    assert "fixture-marker" not in repr(outcomes)
    assert len(calls) == (2 if call_budget == 4 else 1)
    assert all(item.result is None for item in outcomes if item.status != "success")
    if independent_status == "success":
        assert outcomes[1].result.rows == [{"row_count": 7}]

    calls.clear()
    with pytest.raises(ProgrammingError):
        demo_queries.execute_plan(plan)
    assert len(calls) == 1


def test_demo_run_missing_evidence_preserves_independent_results(demo_queries):
    version = demo_queries.versions[0].version_id
    empty = DemoPreviewTask(
        task_id="empty", operation="preview", request=DemoPreview(version_id=version, columns=("PassengerId",),
            filters=(DemoFilter(column="PassengerId", value=0),)),
    )
    dependent = DemoCountTask(
        task_id="dependent", operation="count", request=DemoCount(version_id=version),
        depends_on=("empty",), evidence_filters=(
            DemoEvidenceFilter(column="PassengerId", source_task_id="empty", source_column="PassengerId"),
        ),
    )
    independent = DemoCountTask(task_id="independent", operation="count", request=DemoCount(version_id=version))
    outcomes = demo_queries.run_plan(DemoPlan(tasks=(empty, dependent, independent)))
    assert [item.status for item in outcomes] == ["success", "success", "failed"]
    assert outcomes[0].result.rows == []
    assert outcomes[1].result.rows == [{"row_count": 891}]
    assert outcomes[2].reason == "invalid_evidence" and outcomes[2].result is None


@pytest.mark.parametrize("options", [
    {"max_calls": -1}, {"max_calls": 5}, {"max_calls": True}, {"timeout_seconds": 0},
    {"timeout_seconds": float("nan")}, {"timeout_seconds": 13},
])
def test_demo_run_rejects_invalid_budgets_before_reads(demo_queries, monkeypatch, options):
    plan = _count_plan(demo_queries.versions[0].version_id, ("first", ()))
    monkeypatch.setattr(demo_queries, "_execute_statement", lambda *args, **kwargs: pytest.fail("unexpected read"))
    with pytest.raises(ValueError):
        demo_queries.run_plan(plan, **options)


def test_demo_run_cancels_before_next_read(demo_queries, monkeypatch):
    plan = _count_plan(demo_queries.versions[0].version_id,
                       ("first", ()), ("second", ("first",)), ("third", ("second",)))
    calls = []

    def execute(statement, **kwargs):
        calls.append(statement)
        return [{"row_count": 1}]

    monkeypatch.setattr(demo_queries, "_execute_statement", execute)
    assert [item.status for item in demo_queries.run_plan(plan, cancelled=lambda: True)] == ["cancelled"] * 3
    assert [item.status for item in demo_queries.run_plan(plan, max_calls=0)] == [
        "call_limit", "skipped_dependency", "skipped_dependency",
    ]
    assert calls == []
    outcomes = demo_queries.run_plan(plan, cancelled=lambda: bool(calls))
    assert [item.status for item in outcomes] == ["cancelled"] * 3
    assert len(calls) == 1


def test_demo_run_discards_late_results_and_stops_new_reads(demo_queries, monkeypatch):
    plan = _count_plan(demo_queries.versions[0].version_id,
                       ("first", ()), ("second", ("first",)), ("third", ("second",)))
    clock = 10.0
    deadlines = []

    def execute(statement, *, deadline):
        nonlocal clock
        deadlines.append(deadline)
        clock += 0.75
        return [{"row_count": 1}]

    monkeypatch.setattr("db.demo_queries.time", SimpleNamespace(monotonic=lambda: clock))
    monkeypatch.setattr(demo_queries, "_execute_statement", execute)
    outcomes = demo_queries.run_plan(plan, timeout_seconds=1.0)
    assert [item.status for item in outcomes] == ["success", "deadline", "deadline"]
    assert [item.result for item in outcomes[1:]] == [None, None]
    assert deadlines == [11.0, 11.0]


def test_demo_run_sets_remaining_timeout_on_postgres_transaction(demo_queries, monkeypatch):
    plan = _count_plan(demo_queries.versions[0].version_id, ("first", ()), ("second", ("first",)))
    execute = demo_queries._execute_statement
    clock = 20.0

    def read_timeout(statement, *, deadline):
        nonlocal clock
        clock += 0.25
        statement = statement.add_columns(func.current_setting("statement_timeout").label("timeout"))
        return execute(statement, deadline=deadline)

    monkeypatch.setattr("db.demo_queries.time", SimpleNamespace(monotonic=lambda: clock))
    monkeypatch.setattr(demo_queries, "_execute_statement", read_timeout)
    outcomes = demo_queries.run_plan(plan, timeout_seconds=1.0)
    assert [item.result.rows for item in outcomes] == [
        [{"row_count": 891, "timeout": "750ms"}], [{"row_count": 891, "timeout": "500ms"}],
    ]


def test_demo_run_interrupts_concurrent_postgres_statements_at_shared_deadline(demo_postgres, demo_queries, monkeypatch):
    _, reader, _ = demo_postgres
    plan = _count_plan(demo_queries.versions[0].version_id,
                       ("first", ()), ("second", ()), ("pending", ("first",)))
    both_started = Barrier(2)
    deadlines = []
    sqlstates = []
    execute = demo_queries._execute_statement

    def delayed_read(statement, *, deadline):
        deadlines.append(deadline)
        both_started.wait(timeout=5)
        statement = statement.add_columns(func.pg_sleep(3).label("delay"))
        return execute(statement, deadline=deadline)

    def record_error(context):
        sqlstates.append(context.original_exception.pgcode)

    monkeypatch.setattr(demo_queries, "_execute_statement", delayed_read)
    event.listen(reader, "handle_error", record_error)
    try:
        outcomes = demo_queries.run_plan(plan, timeout_seconds=1)
    finally:
        event.remove(reader, "handle_error", record_error)

    assert len(deadlines) == 2 and deadlines[0] == deadlines[1]
    assert sqlstates == ["57014", "57014"]
    assert [outcome.status for outcome in outcomes] == ["deadline"] * 3
    assert all(outcome.result is None for outcome in outcomes)


def test_demo_run_preserves_completed_result_when_cancelled_with_reads_in_flight(demo_queries, monkeypatch):
    version = demo_queries.versions[0].version_id
    plan = DemoPlan(tasks=tuple(
        DemoCountTask(
            task_id=task_id, operation="count", depends_on=dependencies,
            request=DemoCount(version_id=version, filters=(DemoFilter(column="PassengerId", value=number),)),
        )
        for number, (task_id, dependencies) in enumerate(
            (("completed", ()), ("slow", ()), ("child", ("completed",)), ("pending", ("slow",))), start=1,
        )
    ))
    in_flight = Barrier(2)
    cancellation_requested = Event()
    cancellation_observed = Event()
    calls = []
    execute = demo_queries._execute_statement

    def synchronized_read(statement, *, deadline):
        (number,) = statement.compile().params.values()
        calls.append(number)
        if number in (2, 3):
            in_flight.wait(timeout=5)
            cancellation_requested.set()
            assert cancellation_observed.wait(timeout=5)
        return execute(statement, deadline=deadline)

    def cancelled():
        if cancellation_requested.is_set():
            cancellation_observed.set()
            return True
        return False

    monkeypatch.setattr(demo_queries, "_execute_statement", synchronized_read)
    outcomes = demo_queries.run_plan(plan, cancelled=cancelled)
    assert sorted(calls) == [1, 2, 3]
    assert [(outcome.task_id, outcome.status) for outcome in outcomes] == [
        ("completed", "success"), ("slow", "cancelled"), ("child", "cancelled"), ("pending", "cancelled"),
    ]
    assert outcomes[0].result.rows == [{"row_count": 1}]
    assert all(outcome.result is None for outcome in outcomes[1:])


def test_concurrent_demo_runs_isolate_caller_catalog_and_predecessor_evidence(demo_postgres, monkeypatch):
    _, reader, (titanic, financials, _) = demo_postgres
    all_roots_started = Barrier(4)
    both_sources_finished = Barrier(2)

    def caller(workspace, versions, source):
        access = DemoAccess(f"{workspace}_viewer", workspace)
        catalog = tuple(replace(version, workspace_id=workspace) for version in versions)
        queries = DemoQueries(reader, access, catalog)
        bind_task = queries._bind_task
        execute = queries._execute_statement

        def bind_after_both_sources(prepared, results):
            if prepared.task.task_id == "dependent":
                both_sources_finished.wait(timeout=5)
            return bind_task(prepared, results)

        def read_after_all_roots_start(statement, *, deadline):
            if "row_count" in statement.selected_columns:
                all_roots_started.wait(timeout=5)
            return execute(statement, deadline=deadline)

        monkeypatch.setattr(queries, "_bind_task", bind_after_both_sources)
        monkeypatch.setattr(queries, "_execute_statement", read_after_all_roots_start)
        count = DemoCountTask(task_id="source", operation="count", request=DemoCount(version_id=source.version_id))
        independent = DemoCountTask(task_id="independent", operation="count", request=count.request)
        preview = DemoPreviewTask(
            task_id="dependent", operation="preview", depends_on=("source",),
            request=DemoPreview(version_id=titanic.version_id, columns=("PassengerId",), limit=1),
            evidence_filters=(DemoEvidenceFilter(
                column="PassengerId", source_task_id="source", source_column="row_count",
            ),),
        )
        return queries, DemoPlan(tasks=(count, independent, preview))

    first, first_plan = caller("first", (titanic,), titanic)
    second, second_plan = caller("second", (financials, titanic), financials)
    with pytest.raises(PermissionError):
        first.run_plan(second_plan)
    with ThreadPoolExecutor(max_workers=2) as callers:
        first_future = callers.submit(first.run_plan, first_plan)
        second_future = callers.submit(second.run_plan, second_plan)
        for future, expected in ((first_future, 891), (second_future, 503)):
            source, independent, dependent = future.result(timeout=15)
            assert source.result.rows == [{"row_count": expected}]
            assert independent.result.rows == source.result.rows
            assert dependent.result.request.filters == (DemoFilter(column="PassengerId", value=expected),)
            assert dependent.result.rows == [{"PassengerId": expected}]


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


def _guarded_chat(catalog):
    from demo_app import _render_chat

    _render_chat(catalog)


def test_demo_chat_clarifies_before_read_and_runs_compound_with_fresh_access(demo_queries, monkeypatch):
    import demo_app

    catalog = SimpleNamespace(access=demo_queries.access, versions=demo_queries.versions)
    admissions = []

    def fresh_access():
        admissions.append(True)
        return nullcontext(demo_queries)

    monkeypatch.setattr(demo_app, "_demo_queries", fresh_access)
    app = AppTest.from_function(_guarded_chat, args=(catalog,)).run(timeout=15)
    for question, hints in (
        ("preview sp500_financials", ("Choose sp500_financials in the dataset picker",)),
        ("what is the average?", ("row count", "preview")),
    ):
        app.chat_input(key="demo_chat_input").set_value(question).run(timeout=15)
        assert not app.exception and not app.error and not app.dataframe
        clarification = " ".join(item.value for item in app.info)
        assert all(hint in clarification for hint in hints)
    assert admissions == []
    app.chat_input(key="demo_chat_input").set_value("count and preview").run(timeout=15)
    assert not app.exception and not app.error
    assert admissions == [True]
    assert "titanic has 891 rows." in [item.value for item in app.markdown]
    assert app.dataframe[0].value["PassengerId"].tolist() == list(range(1, 11))
    assert len(_preview_csv(app, "Preview CSV (10 rows)")) == 11
    app.chat_input(key="demo_chat_input").set_value("repeat").run(timeout=15)
    assert admissions == [True, True]
    assert "titanic has 891 rows." in [item.value for item in app.markdown]
    assert len(app.dataframe[0].value) == 10


def test_demo_chat_paraphrases_keep_the_typed_and_scoped_read_paths(demo_queries, monkeypatch):
    import demo_app

    version = demo_queries.versions[0].version_id
    calls = []
    execute = demo_queries.execute
    run_plan = demo_queries.run_plan

    def track_request(request):
        calls.append(request)
        return execute(request)

    def track_plan(plan):
        calls.append(plan)
        return run_plan(plan)

    monkeypatch.setattr(demo_app, "_demo_queries", lambda: nullcontext(demo_queries))
    monkeypatch.setattr(demo_queries, "execute", track_request)
    monkeypatch.setattr(demo_queries, "run_plan", track_plan)
    app = AppTest.from_function(_guarded_chat, args=(demo_queries,)).run(timeout=15)

    def ask(question):
        app.chat_input(key="demo_chat_input").set_value(question).run(timeout=15)
        assert not app.exception and not app.error

    ask("How many records are there?")
    count = DemoCount(version_id=version)
    assert calls == [count]
    ask("Give me a preview of titanic.")
    preview = DemoPreview(version_id=version, limit=10)
    assert calls == [count, preview]

    ask("A preview and the row count.")
    plan = calls[2]
    assert isinstance(plan, DemoPlan)
    assert [(task.operation, task.request) for task in plan.tasks] == [("count", count), ("preview", preview)]
    ask("again")
    assert calls == [count, preview, plan, plan]

    for question, hint in (
        ("How many records are in sp500_financials?", "Choose sp500_financials in the dataset picker"),
        ("Show a preview of world_co2_emissions.", "Choose world_co2_emissions in the dataset picker"),
        ("Count the records in my uploaded CSV.", "would you like a row count"),
        ("Count the rows where Survived = 1.", "would you like a row count"),
        ("Show a preview and delete the dataset.", "would you like a row count"),
        ("A preview of titanic and a count of sp500_financials.", "would you like a row count"),
        ("Show me 5 rows.", "would you like a row count"),
        ("again", "Ask a new question"),
    ):
        ask(question)
        assert not app.dataframe and hint in app.info[0].value
        assert calls == [count, preview, plan, plan]


def test_demo_chat_shows_successful_count_after_preview_failure(demo_queries, monkeypatch):
    import demo_app

    monkeypatch.setattr(demo_app, "_demo_queries", lambda: nullcontext(demo_queries))
    execute = demo_queries._execute_statement
    calls = []
    preview_failed = Event()

    def fail_preview_once(statement, **kwargs):
        operation = "count" if "row_count" in statement.selected_columns else "preview"
        calls.append(operation)
        if operation == "preview" and not preview_failed.is_set():
            preview_failed.set()
            raise ProgrammingError(None, None, RuntimeError("fixture-marker"))
        return execute(statement, **kwargs)

    monkeypatch.setattr(demo_queries, "_execute_statement", fail_preview_once)
    catalog = SimpleNamespace(access=demo_queries.access, versions=demo_queries.versions)
    app = AppTest.from_function(_guarded_chat, args=(catalog,)).run(timeout=15)
    app.chat_input(key="demo_chat_input").set_value("count and preview").run(timeout=15)
    assert not app.exception and not app.dataframe
    assert "titanic has 891 rows." in [item.value for item in app.markdown]
    feedback = [item.value for item in [*app.warning, *app.error, *app.info, *app.markdown]]
    assert any("The preview could not finish." in message for message in feedback)
    assert all("fixture-marker" not in message for message in feedback)
    assert not any(button.key == "demo_chat_retry" for button in app.button)
    assert sorted(calls) == ["count", "preview"]
    app.run(timeout=15)
    assert not app.exception and len(calls) == 2
    app.chat_input(key="demo_chat_input").set_value("again").run(timeout=15)
    assert any("Ask a new question" in item.value for item in app.info)
    assert len(calls) == 2
    app.chat_input(key="demo_chat_input").set_value("count and preview").run(timeout=15)
    assert not app.exception and not app.error
    assert sorted(calls[2:]) == ["count", "preview"]
    assert len(app.dataframe[0].value) == 10


def test_demo_chat_followup_recomputes_and_is_local_to_one_conversation(demo_queries, monkeypatch):
    import demo_app

    admissions = []
    counts = iter((20, 21))

    def fresh_access():
        admissions.append(True)
        return nullcontext(demo_queries)

    monkeypatch.setattr(demo_app, "_demo_queries", fresh_access)
    monkeypatch.setattr(demo_queries, "execute", lambda request: [{"row_count": next(counts)}])
    first = AppTest.from_function(_guarded_chat, args=(demo_queries,)).run(timeout=15)
    second = AppTest.from_function(_guarded_chat, args=(demo_queries,)).run(timeout=15)
    first.chat_input(key="demo_chat_input").set_value("row count").run(timeout=15)
    assert "titanic has 20 rows." in [item.value for item in first.markdown]
    assert first.session_state["demo_chat_last_operation"] == "count"
    assert first.session_state["demo_chat_conversation"] != second.session_state["demo_chat_conversation"]
    second.chat_input(key="demo_chat_input").set_value("again").run(timeout=15)
    assert "Ask a new question" in second.info[0].value and admissions == [True]
    first.chat_input(key="demo_chat_input").set_value("repeat").run(timeout=15)
    assert "titanic has 21 rows." in [item.value for item in first.markdown]
    first.chat_input(key="demo_chat_input").set_value("what is the average?").run(timeout=15)
    first.chat_input(key="demo_chat_input").set_value("again").run(timeout=15)
    assert "Ask a new question" in first.info[0].value and admissions == [True, True]


@pytest.mark.parametrize("change", ["principal", "workspace", "dataset", "columns", "policy", "conversation"])
def test_demo_chat_followup_invalidates_on_scope_change(demo_queries, monkeypatch, change):
    import demo_app
    from ui import demo_tabs

    catalog = SimpleNamespace(access=demo_queries.access, versions=demo_queries.versions)
    admissions = []

    def fresh_access():
        admissions.append(True)
        return nullcontext(demo_queries)

    monkeypatch.setattr(demo_app, "_demo_queries", fresh_access)
    app = AppTest.from_function(_guarded_chat, args=(catalog,)).run(timeout=15)
    app.chat_input(key="demo_chat_input").set_value("preview").run(timeout=15)
    assert len(app.dataframe[0].value) == 10
    if change == "principal":
        catalog.access = replace(catalog.access, principal_id="another_reader")
    elif change == "workspace":
        catalog.access = replace(catalog.access, workspace_id="another_workspace")
    elif change == "dataset":
        app.selectbox(key="demo_chat_dataset").set_value(catalog.versions[1].version_id).run(timeout=15)
        app.selectbox(key="demo_chat_dataset").set_value(catalog.versions[0].version_id).run(timeout=15)
    elif change == "columns":
        first, *others = catalog.versions
        catalog.versions = (replace(first, columns=("PassengerId",)), *others)
    elif change == "policy":
        monkeypatch.setattr(demo_tabs, "_DEMO_QUERY_POLICY_VERSION", 2)
    else:
        app.session_state["demo_chat_conversation"] = uuid4()
    app.chat_input(key="demo_chat_input").set_value("again").run(timeout=15)
    assert not app.exception and not app.dataframe and "Ask a new question" in app.info[0].value
    assert admissions == [True]


@pytest.mark.parametrize("operation", ["preview", "count and preview"])
def test_guarded_followup_detects_fresh_catalog_change_before_reading(demo_queries, monkeypatch, operation):
    import demo_app

    catalog = SimpleNamespace(access=demo_queries.access, versions=demo_queries.versions)
    reader = demo_queries
    monkeypatch.setattr(demo_app, "_demo_queries", lambda: nullcontext(reader))
    app = AppTest.from_function(_guarded_chat, args=(catalog,)).run(timeout=15)
    app.chat_input(key="demo_chat_input").set_value(operation).run(timeout=15)
    assert len(app.dataframe[0].value) == 10
    version, *others = catalog.versions
    reader = SimpleNamespace(access=catalog.access, versions=(replace(version, columns=("PassengerId",)), *others))
    app.chat_input(key="demo_chat_input").set_value("again").run(timeout=15)
    assert not app.exception and not app.dataframe and "access changed" in app.info[0].value.lower()
    reader = demo_queries
    app.chat_input(key="demo_chat_input").set_value("again").run(timeout=15)
    assert "Ask a new question" in app.info[0].value


def _preview_csv(app, label):
    frames = [item.proto.srcdoc for item in app.get("iframe")]
    source, = [frame for frame in frames if f"Save {label}</a>" in frame]
    encoded = source.split("base64,", 1)[1].split('"', 1)[0]
    return list(csv.reader(StringIO(b64decode(encoded).decode("utf-8-sig"))))


def test_demo_chat_and_explorer_explain_size_limit_without_losing_count(wide_demo):
    app = AppTest.from_function(_demo_tabs, args=(wide_demo,)).run(timeout=15)
    assert not app.exception and not app.dataframe and not app.get("iframe")
    assert any(caption.value == "6 matching rows" for caption in app.caption)
    assert any("preview exceeds the size limit" in warning.value for warning in app.warning)

    app.chat_input(key="demo_chat_input").set_value("preview").run(timeout=15)
    assert not app.exception and not app.get("iframe")
    assert "preview exceeds the size limit" in app.chat_message[0].warning[0].value
    assert not any(button.label == "Retry question" for button in app.button)

    app.chat_input(key="demo_chat_input").set_value("count and preview").run(timeout=15)
    assert not app.exception and not app.get("iframe")
    assert any("Wide fixture has 6 rows" in message.value for message in app.chat_message[0].markdown)
    assert "preview exceeds the size limit" in app.chat_message[0].warning[0].value

    version = wide_demo.versions[0].version_id
    app.selectbox(key=f"demo_filter_{version}").set_value("score").run(timeout=15)
    app.text_input(key=f"demo_filter_value_{version}_score").set_value("6").run(timeout=15)
    assert not app.exception and _preview_csv(app, "Preview CSV (1 row)") == [["note", "score"], ["", "6"]]


@pytest.mark.parametrize("demo", DEMOS)
def test_chat_and_explorer_agree_on_demo_results(demo, demo_queries, monkeypatch):
    import ui

    for module in ("chat", "explorer"):
        monkeypatch.delattr(ui, module, raising=False)
        monkeypatch.setitem(sys.modules, f"ui.{module}", None)

    version = next(item for item in demo_queries.versions if item.label == demo)
    app = AppTest.from_function(_demo_tabs, args=(demo_queries,)).run(timeout=15)
    app.selectbox(key="demo_chat_dataset").set_value(version.version_id)
    app.selectbox(key="demo_explorer_dataset").set_value(version.version_id)
    app.chat_input(key="demo_chat_input").set_value(f"How many rows are in {demo}?").run(timeout=15)
    assert not app.exception and not app.error
    assert [item.label for item in app.expander] == [f"Columns ({len(version.columns)})"] * 2
    markdown = [item.value for item in app.markdown]
    assert markdown.count(DEMO_BRIEFS[demo]) == 2
    assert markdown.count(", ".join(version.columns)) == 2
    assert f"{demo} has {DEMOS[demo]:,} rows." in [item.value for item in app.markdown]
    assert any(
        'role="status"' in item.value and f"{demo} has {DEMOS[demo]:,} rows." in item.value
        for item in app.markdown
    )
    assert f"{DEMOS[demo]:,} matching rows" in [item.value for item in app.caption]

    app.chat_input(key="demo_chat_input").set_value("preview").run(timeout=15)
    assert not app.exception and not app.error
    assert len(app.dataframe) == 2
    assert any(
        'role="status"' in item.value and f"Showing 10 rows from {demo}." in item.value
        for item in app.markdown
    )
    pd.testing.assert_frame_equal(app.dataframe[0].value, app.dataframe[1].value.head(10).reset_index(drop=True))
    assert len(_preview_csv(app, "Preview CSV (10 rows)")) == 11
    assert len(_preview_csv(app, "Preview CSV (50 rows)")) == 51

    if demo == "titanic":
        app.selectbox(key=f"demo_filter_{version.version_id}").set_value("PassengerId").run(timeout=15)
        value_key = f"demo_filter_value_{version.version_id}_PassengerId"
        app.text_input(key=value_key).set_value("891").run(timeout=15)
        assert not app.exception and not app.error
        assert [item.value for item in app.caption] == ["1 matching row"]
        assert app.dataframe[0].value["PassengerId"].tolist() == [891]
        header, row = _preview_csv(app, "Preview CSV (1 row)")
        assert header == list(version.columns)
        assert row[header.index("PassengerId")] == "891"

        app.text_input(key=value_key).set_value("0").run(timeout=15)
        assert [item.value for item in app.caption] == ["0 matching rows"]
        assert _preview_csv(app, "Preview CSV (0 rows)") == [header]


def test_demo_chat_retries_only_on_request_and_discards_failure_on_dataset_change(demo_queries, monkeypatch):
    attempts = []
    execute = demo_queries.execute
    fail_next = True

    def execute_with_transient_failure(request):
        nonlocal fail_next
        if isinstance(request, DemoPreview) and request.limit == 10:
            attempts.append(request)
            if fail_next:
                fail_next = False
                raise ProgrammingError(None, None, RuntimeError("synthetic failure"))
        return execute(request)

    monkeypatch.setattr(demo_queries, "execute", execute_with_transient_failure)
    app = AppTest.from_function(_demo_tabs, args=(demo_queries,)).run(timeout=15)
    app.chat_input(key="demo_chat_input").set_value("preview").run()
    assert len(attempts) == 1
    assert app.error[0].value == "This question could not run. Please retry in a moment."
    app.run()
    assert len(attempts) == 1
    app.button(key="demo_chat_retry").click().run()
    assert attempts == [attempts[0], attempts[0]]
    assert not app.exception and not app.error
    assert len(app.dataframe[0].value) == 10

    fail_next = True
    app.chat_input(key="demo_chat_input").set_value("preview").run()
    assert len(attempts) == 3
    for version in (demo_queries.versions[1], demo_queries.versions[0]):
        app.selectbox(key="demo_chat_dataset").set_value(version.version_id).run()
        assert not app.exception and not app.error
        assert not any(button.key == "demo_chat_retry" for button in app.button)
        assert len(attempts) == 3


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
    assert "Curated samples cannot be deleted." in [item.value for item in demo.caption]
    assert len(_preview_csv(demo, "Preview CSV (50 rows)")) == 51
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
    import db

    owner, reader, _ = demo_postgres
    monkeypatch.delattr(db, "postgres", raising=False)
    monkeypatch.setitem(sys.modules, "db.postgres", None)
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
