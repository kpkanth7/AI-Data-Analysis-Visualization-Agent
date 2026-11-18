"""Run with python -m pytest -q -s tests/benchmark_postgres.py."""

import json
import math
import os
import platform
import subprocess
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median
from threading import Barrier, Lock
from time import perf_counter
from uuid import UUID

import pandas as pd
import pytest
from sqlalchemy import Column, Integer, MetaData, Table, Text, create_engine, event, func, select
from sqlalchemy.pool import NullPool

from db.demo_queries import (
    DemoAccess, DemoCount, DemoCountTask, DemoFilter, DemoPlan, DemoPreview, DemoPreviewTask,
    DemoQueries, DemoSort, DemoVersion, demo_read_transaction,
)
from db.workspace_datasets import WorkspaceColumnSummary
from tests.test_workspace_datasets import make_workspace, workspace_database  # noqa: F401


def report(**measurements):
    print(json.dumps(measurements, sort_keys=True))


def latency(samples):
    ordered = sorted(samples)
    return {
        "samples": len(samples),
        "p50_ms": round(median(ordered) * 1000, 2),
        "p95_ms": round(ordered[math.ceil(len(ordered) * 0.95) - 1] * 1000, 2),
    }


def measure(operation, expected, *, case, rows):
    start = perf_counter()
    assert operation() == expected
    first = perf_counter() - start
    samples = []
    for _ in range(30):
        start = perf_counter()
        actual = operation()
        samples.append(perf_counter() - start)
        assert actual == expected

    tracemalloc.start()
    try:
        assert operation() == expected
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    report(
        case=case, rows=rows, first_ms=round(first * 1000, 2), python_peak_bytes=peak,
        compact_response_bytes=len(json.dumps(expected, separators=(",", ":")).encode()),
        **latency(samples),
    )


def explain_preview_cache(owner, reader, statement, rows):
    with owner.connect() as connection:
        directory = Path(connection.exec_driver_sql("SHOW data_directory").scalar_one())
    # Only reset this fixture's PostgreSQL buffers. The OS may still cache the file.
    subprocess.run(
        ["pg_ctl", "-D", str(directory), "-l", str(directory.parent / "postgres.log"), "-m", "fast", "-w", "restart"],
        check=True, capture_output=True, timeout=30,
    )

    def nodes(plan):
        yield plan
        for child in plan.get("Plans", ()):
            yield from nodes(child)

    for cache in ("after_restart", "warm"):
        with demo_read_transaction(reader) as connection:
            compiled = statement.compile(dialect=connection.dialect, compile_kwargs={"literal_binds": True})
            details = connection.exec_driver_sql(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {compiled}").scalar_one()[0]
        root = details["Plan"]
        assert root["Actual Rows"] == 25
        assert root["Shared Read Blocks"] > 0 if cache == "after_restart" else root["Shared Read Blocks"] == 0
        report(case="preview_plan", rows=rows, cache=cache, database_ms=details["Execution Time"],
               returned_rows=root["Actual Rows"], shared_hits=root["Shared Hit Blocks"],
               shared_reads=root["Shared Read Blocks"],
               sort_methods=[node["Sort Method"] for node in nodes(root) if "Sort Method" in node])


@pytest.fixture(scope="module")
def benchmark_database(postgres_owner):
    owner = postgres_owner
    with owner.begin() as connection:
        connection.exec_driver_sql("CREATE ROLE w5_benchmark_reader LOGIN NOINHERIT")
        connection.exec_driver_sql("CREATE SCHEMA demo")
        connection.exec_driver_sql("GRANT USAGE ON SCHEMA demo TO w5_benchmark_reader")
        report(
            case="environment", postgres=connection.exec_driver_sql("SHOW server_version").scalar_one(),
            python=platform.python_version(), machine=platform.machine(), logical_cpus=os.cpu_count(),
            pandas=pd.__version__,
        )
    yield owner


@pytest.mark.parametrize("size", [1_000, 50_000])
def test_typed_reads_and_grouping(benchmark_database, size):
    owner = benchmark_database
    relation = Table(
        f"w5_rows_{size}", MetaData(),
        Column("id", Integer, primary_key=True), Column("bucket", Integer),
        Column("amount", Integer), Column("note", Text), schema="demo",
    )
    name = f"demo.{relation.name}"
    with owner.begin() as connection:
        start = perf_counter()
        relation.create(connection)
        connection.exec_driver_sql(f"""
            INSERT INTO {name}
            SELECT id, mod(id, 10), mod(id, 101), repeat(md5(id::text), 8)
            FROM generate_series(1, %s) AS id
        """, (size,))
        connection.exec_driver_sql(f"GRANT SELECT ON {name} TO w5_benchmark_reader")
        connection.exec_driver_sql(f"ANALYZE {name}")
        stored = connection.exec_driver_sql("SELECT pg_total_relation_size(%s)", (name,)).scalar_one()
    report(case="fixture_ingest", rows=size, elapsed_ms=round((perf_counter() - start) * 1000, 2), storage_bytes=stored)

    reader = create_engine(owner.url.set(username="w5_benchmark_reader"), poolclass=NullPool)
    try:
        version = DemoVersion(UUID(int=size), "synthetic", "Frozen fixture", relation, ("id", "bucket", "amount", "note"), "id")
        service = DemoQueries(reader, DemoAccess("reader", "synthetic"), (version,))
        filters = (DemoFilter(column="bucket", value=3),)
        count = DemoCount(version_id=version.version_id, filters=filters)
        preview = DemoPreview(
            version_id=version.version_id, columns=("id", "amount"), filters=filters,
            sort=(DemoSort(column="amount", descending=True),), limit=25,
        )
        ids = range(3, size + 1, 10)
        expected_count = [{"row_count": len(ids)}]
        expected_preview = [
            {"id": row, "amount": row % 101}
            for row in sorted(ids, key=lambda row: (-(row % 101), row))[:25]
        ]
        expected_group = [
            (bucket, sum(row % 101 for row in range(bucket or 10, size + 1, 10)))
            for bucket in range(10)
        ]

        def pandas_preview():
            with demo_read_transaction(reader) as connection:
                frame = pd.read_sql(select(relation), connection)
            selected = frame.loc[frame.bucket == 3, ["id", "amount"]]
            return selected.sort_values(["amount", "id"], ascending=[False, True]).head(25).to_dict("records")

        group = select(relation.c.bucket, func.sum(relation.c.amount)).group_by(relation.c.bucket).order_by(relation.c.bucket)

        def sql_group():
            with demo_read_transaction(reader) as connection:
                return [tuple(row) for row in connection.execute(group)]

        def pandas_group():
            with demo_read_transaction(reader) as connection:
                frame = pd.read_sql(select(relation), connection)
            return list(frame.groupby("bucket")["amount"].sum().items())

        explain_preview_cache(owner, reader, service._statement(preview), size)
        measure(lambda: service.execute(count), expected_count, case="typed_count", rows=size)
        measure(lambda: service.execute(preview), expected_preview, case="typed_preview", rows=size)
        measure(pandas_preview, expected_preview, case="pandas_preview_reference", rows=size)
        measure(sql_group, expected_group, case="sql_grouping_probe", rows=size)
        measure(pandas_group, expected_group, case="pandas_grouping_reference", rows=size)
        with pytest.raises(PermissionError):
            DemoQueries(reader, DemoAccess("other", "other-workspace"), (version,)).execute(preview)
        if size == 50_000:
            plan = DemoPlan(tasks=(
                DemoCountTask(task_id="count", operation="count", request=count),
                DemoPreviewTask(task_id="preview", operation="preview", request=preview),
            ))
            measure_concurrent(reader, service, plan, [expected_count, expected_preview])
    finally:
        reader.dispose()


def measure_concurrent(reader, service, plan, expected):
    lock = Lock()
    active = peak = 0

    def checkout(*_):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)

    def checkin(*_):
        nonlocal active
        with lock:
            active -= 1

    event.listen(reader, "checkout", checkout)
    event.listen(reader, "checkin", checkin)
    def run_load(callers, duration):
        nonlocal peak
        peak = 0
        started = 0.0

        def start_together():
            nonlocal started
            started = perf_counter()

        barrier = Barrier(callers, action=start_together, timeout=10)

        def requests():
            barrier.wait()
            times = []
            while perf_counter() < started + duration:
                start = perf_counter()
                outcomes = service.run_plan(plan)
                times.append(perf_counter() - start)
                assert all(outcome.status == "success" for outcome in outcomes)
                assert [outcome.result.rows for outcome in outcomes] == expected
            return times

        with ThreadPoolExecutor(max_workers=callers) as executor:
            samples = [sample for times in executor.map(lambda _: requests(), range(callers)) for sample in times]
        elapsed = perf_counter() - started
        assert active == 0 and 1 <= peak <= 2 * callers
        return samples, elapsed

    for callers in (1, 4, 8):
        samples, elapsed = run_load(callers, 5)
        report(case="concurrent_plans", callers=callers, peak_connections=peak,
               elapsed_seconds=round(elapsed, 2),
               successful_plans_per_second=round(len(samples) / elapsed, 2), **latency(samples))

    tracemalloc.start()
    try:
        samples, elapsed = run_load(8, 2)
        _, python_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    report(case="concurrent_python_memory", callers=8, successful_plans=len(samples),
           elapsed_seconds=round(elapsed, 2), python_peak_bytes=python_peak)

    assert [outcome.status for outcome in service.run_plan(plan, max_calls=1)] == ["success", "call_limit"]


def test_workspace_summary(make_workspace):
    workspace, _, _, _, reader, _ = make_workspace("w5-synthetic")
    contents = ("bucket,amount\n" + "".join(f"{i % 10},{i % 101}\n" for i in range(1_000))).encode()
    start = perf_counter()
    version = workspace.upload_csv(contents, "synthetic.csv", UUID(int=1))
    report(case="workspace_ingest", rows=1_000, csv_bytes=len(contents), elapsed_ms=round((perf_counter() - start) * 1000, 2))
    request = WorkspaceColumnSummary(version_id=version.version_id, column="amount")
    expected = {
        "row_count": 1_000, "non_null_count": 1_000, "null_count": 0,
        "minimum": 0, "maximum": 100, "total": 49_545, "mean": "49.5450000000000000",
    }
    statements = []

    def capture(connection, cursor, statement, *args):
        statements.append(statement)

    event.listen(reader, "before_cursor_execute", capture)

    def summarize():
        result = workspace.summarize(request)
        return {**result.values, "mean": str(result.values["mean"])}

    try:
        assert summarize() == expected
        report(case="workspace_summary_statements", per_request=len(statements),
               catalog_inspection=sum("pg_attribute" in statement for statement in statements))
    finally:
        event.remove(reader, "before_cursor_execute", capture)
    measure(summarize, expected, case="workspace_summary", rows=1_000)
