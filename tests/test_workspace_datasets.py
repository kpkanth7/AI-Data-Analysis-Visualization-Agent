"""Exercise private dataset operations with actual PostgreSQL roles."""

import csv
import json
from base64 import b64decode
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
from sqlalchemy.exc import DBAPIError, IntegrityError, ProgrammingError, SQLAlchemyError
from sqlalchemy.pool import NullPool
from streamlit.testing.v1 import AppTest

from db.demo_catalog import DEMO_READER
from db.workspace_datasets import (
    MAX_UPLOAD_ROWS, ROW_KEY, WORKSPACE_UPLOADS_PER_DAY, CleaningPreviewChanged,
    UploadLimitReached, UploadRequestConflict,
    WorkspaceAccess, WorkspaceColumnSummary, WorkspaceDatasets, WorkspaceRowCount, WorkspaceTrimWhitespace,
    create_analysis_results, create_upload_receipts, version_catalog,
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


def test_same_filename_isolated_and_delete_preserves_other_data(
    make_workspace, workspace_database, synthetic_demo_files,
):
    with workspace_database.begin() as connection:
        connection.exec_driver_sql(f"CREATE ROLE {DEMO_READER} LOGIN NOINHERIT")
        stage_demo_catalog(connection)
        grant_demo_access(connection)

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
        pd.DataFrame({"=header": [" 	=SUM(1,2)", "café"] + ["text"] * 49, "number": range(-5, 46)}),
        "preview.csv",
    )
    count = service.summarize(WorkspaceRowCount(version_id=dataset.version_id))
    assert count.request.version_id == dataset.version_id
    assert count.values == {"row_count": 51}
    preview = service.preview_csv(dataset.version_id)
    assert (preview.version_id, preview.row_count, preview.truncated) == (dataset.version_id, 50, True)
    rows = list(csv.reader(StringIO(preview.data.decode("utf-8-sig"))))
    assert rows[:3] == [["'=header", "number"], ["' 	=SUM(1,2)", "-5"], ["café", "-4"]]
    assert [int(row[1]) for row in rows[1:]] == list(range(-5, 45))

    full = service.full_csv(dataset.version_id)
    assert (full.version_id, full.row_count, full.truncated) == (dataset.version_id, 51, False)
    assert list(csv.reader(StringIO(full.data.decode("utf-8-sig")))) == [*rows, ["text", "45"]]

    state.current = None
    with pytest.raises(PermissionError):
        service.preview_csv(dataset.version_id)
    with pytest.raises(PermissionError):
        service.full_csv(dataset.version_id)
    state.current = replace(access, session_id=uuid4())
    with pytest.raises(PermissionError):
        service.preview_csv(dataset.version_id)


def test_csv_structure_is_validated_before_publication(make_workspace):
    service, *_ = make_workspace("alice")
    request_id = uuid4()
    invalid_files = (
        b"",
        b"\n",
        b"value,value\n1,2\n",
        b"value,\n1,2\n",
        b"value\n1,2\n",
        b"value,other\n1\n",
        b'value\n"unclosed\n',
        b"value\n\x00\n",
        b"value\n\xff\n",
        b"value\n" + b"1\n" * (MAX_UPLOAD_ROWS + 1),
        ",".join(f"col_{i}" for i in range(65)).encode(),
    )
    for payload in invalid_files:
        with pytest.raises(ValueError):
            service.upload_csv(payload, "sample.csv", request_id)
    assert not service.list_datasets()

    payload = '\ufeff"label, name",value\r\n"café\nline",2\r\n\r\n"comma, here",3\r\n'.encode()
    dataset = service.upload_csv(payload, "sample.csv", request_id)
    assert service.preview(dataset.version_id) == [
        {"label, name": "café\nline", "value": 2},
        {"label, name": "comma, here", "value": 3},
    ]
    assert service.upload_csv(payload, "sample.csv", request_id) == dataset
    for _ in range(WORKSPACE_UPLOADS_PER_DAY - 1):
        service.upload_csv(b"value\n", "header-only.csv", uuid4())
    assert len(service.list_datasets()) == WORKSPACE_UPLOADS_PER_DAY


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
    assert alice.summarize(count).values == {"row_count": 3}
    request = WorkspaceColumnSummary(version_id=local.version_id, column="Amount total")
    summary = alice.summarize(request)
    assert summary.request == request
    assert summary.values == {
        "row_count": 3, "non_null_count": 2, "null_count": 1,
        "minimum": -2.5, "maximum": 7.5, "total": 5.0, "mean": 2.5,
    }
    for column, non_null in (("Name", 2), ("Flag", 3)):
        assert alice.summarize(WorkspaceColumnSummary(version_id=local.version_id, column=column)).values == {
            "row_count": 3, "non_null_count": non_null, "null_count": 3 - non_null,
        }
    exact = alice.summarize(WorkspaceColumnSummary(version_id=local.version_id, column="Exact"))
    assert exact.values["total"] == 9007199254740995
    assert alice.summarize(WorkspaceColumnSummary(version_id=empty.version_id, column="Value")).values == {
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
        alice.full_csv(foreign.version_id)
    with pytest.raises(PermissionError):
        bob.summarize(WorkspaceColumnSummary(version_id=local.version_id, column="Amount total"))
    alice.delete(empty.version_id)
    with pytest.raises(PermissionError):
        alice.summarize(WorkspaceRowCount(version_id=empty.version_id))
    with pytest.raises(PermissionError):
        alice.full_csv(empty.version_id)

    with workspace_database.begin() as connection:
        connection.exec_driver_sql(
            f"INSERT INTO {schema}.v_{local.version_id.hex} ({ROW_KEY}) "
            "SELECT value FROM generate_series(3, %s) AS value", (MAX_UPLOAD_ROWS - 1,),
        )
    assert alice.full_csv(local.version_id).row_count == MAX_UPLOAD_ROWS
    with workspace_database.begin() as connection:
        connection.exec_driver_sql(
            f"INSERT INTO {schema}.v_{local.version_id.hex} ({ROW_KEY}) VALUES (%s)", (MAX_UPLOAD_ROWS,),
        )
    with pytest.raises(ValueError, match="analysis row limit"):
        alice.summarize(count)
    with pytest.raises(ValueError, match="full CSV row limit"):
        alice.full_csv(local.version_id)
    state.current = None
    with pytest.raises(PermissionError):
        alice.summarize(count)


def _workspace_app(lookup, engines):
    from ui.private_workspace import render_private_workspace

    render_private_workspace(lookup, engines)


def test_analysis_csv_preserves_executed_values_and_rechecks_access(make_workspace):
    service, access, state, _, reader, writer = make_workspace("alice")
    dataset = service.upload(pd.DataFrame({"=集計": [9007199254740993, -2]}), "summary.csv")
    request = WorkspaceColumnSummary(version_id=dataset.version_id, column="=集計")
    result = service.summarize(request)
    rows = list(csv.DictReader(StringIO(service.analysis_csv(result).decode("utf-8-sig"))))
    assert {row["result_id"] for row in rows} == {str(result.result_id)}
    assert {row["version_id"] for row in rows} == {str(dataset.version_id)}
    assert {row["operation"] for row in rows} == {"summarize_column"}
    assert {row["column"] for row in rows} == {"'=集計"}
    assert {row["metric"]: row["value"] for row in rows} == {
        key: str(value) for key, value in result.values.items()
    }
    assert next(row["value"] for row in rows if row["metric"] == "maximum") == "9007199254740993"

    new_access = replace(access, session_id=uuid4())
    new_session = WorkspaceDatasets(new_access, lambda: new_access, reader, writer)
    with pytest.raises(PermissionError):
        new_session.analysis_csv(result)
    state.current = None
    with pytest.raises(PermissionError):
        service.analysis_csv(result)
    state.current = access
    service.delete(dataset.version_id)
    with pytest.raises(PermissionError):
        service.analysis_csv(result)


def test_saved_analysis_is_exact_owner_scoped_and_removed_with_version(make_workspace, workspace_database):
    service, access, state, schema, reader, writer = make_workspace("alice")
    dataset = service.upload(pd.DataFrame({"value": [9007199254740993, -2]}), "summary.csv")
    with workspace_database.begin() as connection:
        create_analysis_results(connection, access.workspace_id)
    request = WorkspaceColumnSummary(version_id=dataset.version_id, column="value")
    result = service.summarize(request)
    service.save_analysis(result)
    service.save_analysis(result)

    new_access = replace(access, session_id=uuid4())
    new_session = WorkspaceDatasets(new_access, lambda: new_access, reader, writer)
    restored = new_session.restore_analysis(dataset.version_id, result.result_id)
    assert restored.access == new_access and restored.request == request
    assert restored.result_id == result.result_id
    assert restored.values == result.values
    assert new_session.analysis_csv(restored) == service.analysis_csv(result)
    with pytest.raises(PermissionError):
        new_session.save_analysis(result)

    other_access = replace(new_access, principal_id="bob")
    other = WorkspaceDatasets(other_access, lambda: other_access, reader, writer)
    with pytest.raises(PermissionError):
        other.restore_analysis(dataset.version_id, result.result_id)
    viewer_access = replace(new_access, can_manage=False)
    viewer = WorkspaceDatasets(viewer_access, lambda: viewer_access, reader, writer)
    assert viewer.restore_analysis(dataset.version_id, result.result_id).values == result.values
    with pytest.raises(PermissionError):
        viewer.save_analysis(viewer.summarize(request))

    wrong_version = service.upload(pd.DataFrame({"value": [0]}), "other.csv")
    with pytest.raises(PermissionError):
        new_session.restore_analysis(wrong_version.version_id, result.result_id)
    state.current = None
    with pytest.raises(PermissionError):
        service.restore_analysis(dataset.version_id, result.result_id)
    state.current = access
    with workspace_database.begin() as connection:
        create_analysis_results(connection, access.workspace_id)
        assert connection.exec_driver_sql(f"SELECT count(*) FROM {schema}.analysis_results").scalar_one() == 1
    service.delete(dataset.version_id)
    with workspace_database.connect() as connection:
        assert connection.exec_driver_sql(f"SELECT count(*) FROM {schema}.analysis_results").scalar_one() == 0
    with pytest.raises(PermissionError):
        new_session.restore_analysis(dataset.version_id, result.result_id)


def test_cleaning_preview_is_bounded_and_preserves_source(make_workspace, workspace_database):
    service, access, state, schema, reader, writer = make_workspace("alice")
    other, *_ = make_workspace("bob")
    values = ["  café\t", None, "   ", "001", *(["same"] * 46), " last "]
    dataset = service.upload(pd.DataFrame({"label": values, "number": range(51)}), "source.csv")
    original_csv = service.full_csv(dataset.version_id)
    plan = WorkspaceTrimWhitespace(version_id=dataset.version_id, column="label")

    result = service.preview_cleaning(plan)
    assert result.plan == plan
    assert (result.row_count, result.changed_count, len(result.rows)) == (51, 3, 50)
    assert result.rows[:4] == (("  café\t", "café"), (None, None), ("   ", ""), ("001", "001"))
    assert service.preview_cleaning(plan) == result
    assert service.replay_cleaning(result) == result
    numeric = service.preview_cleaning(WorkspaceTrimWhitespace(version_id=dataset.version_id, column="number"))
    assert numeric.changed_count == 0 and numeric.rows[0] == (0, 0)
    assert service.full_csv(dataset.version_id) == original_csv
    assert service.list_datasets() == (dataset,)

    new_access = replace(access, session_id=uuid4())
    new_session = WorkspaceDatasets(new_access, lambda: new_access, reader, writer)
    with pytest.raises(PermissionError):
        new_session.replay_cleaning(result)
    viewer_access = replace(access, can_manage=False)
    viewer = WorkspaceDatasets(viewer_access, lambda: viewer_access, reader, writer)
    with pytest.raises(PermissionError):
        viewer.replay_cleaning(viewer.preview_cleaning(plan))

    with workspace_database.begin() as connection:
        connection.exec_driver_sql(
            f"UPDATE {schema}.v_{dataset.version_id.hex} SET label = %s WHERE {ROW_KEY} = %s",
            (" updated ", 50),
        )
    changed = service.preview_cleaning(plan)
    assert changed.rows == result.rows and changed.changed_count == result.changed_count
    with pytest.raises(CleaningPreviewChanged):
        service.replay_cleaning(result)
    with pytest.raises(CleaningPreviewChanged):
        service.publish_cleaning(result, uuid4())

    with pytest.raises(PermissionError):
        other.preview_cleaning(plan)
    with pytest.raises(PermissionError):
        service.preview_cleaning(WorkspaceTrimWhitespace(version_id=dataset.version_id, column=ROW_KEY))
    state.current = None
    with pytest.raises(PermissionError):
        service.preview_cleaning(plan)
    with pytest.raises(PermissionError):
        service.replay_cleaning(changed)


def test_publish_cleaned_version_is_atomic_and_retry_safe(make_workspace, workspace_database):
    service, access, state, schema, reader, writer = make_workspace("alice")
    source = service.upload_csv(
        b'"label, name",number\n" first ",9007199254740993\n,7\n"  ",-2\n',
        "sample.csv", uuid4(),
    )
    original = service.full_csv(source.version_id)
    plan = WorkspaceTrimWhitespace(version_id=source.version_id, column="label, name")
    reviewed = service.replay_cleaning(service.preview_cleaning(plan))
    request_id = uuid4()
    viewer_access = replace(access, can_manage=False)
    viewer = WorkspaceDatasets(viewer_access, lambda: viewer_access, reader, writer)
    with pytest.raises(PermissionError):
        viewer.publish_cleaning(viewer.preview_cleaning(plan), request_id)
    tables = set(inspect(workspace_database).get_table_names(schema=schema))
    with workspace_database.begin() as connection:
        connection.exec_driver_sql(
            f"ALTER TABLE {schema}.upload_receipts ADD CONSTRAINT fail_cleaning CHECK (request_id <> %s::uuid)",
            (str(request_id),),
        )
    with pytest.raises(IntegrityError):
        service.publish_cleaning(reviewed, request_id)
    assert service.list_datasets() == (source,)
    assert set(inspect(workspace_database).get_table_names(schema=schema)) == tables
    with workspace_database.begin() as connection:
        connection.exec_driver_sql(f"ALTER TABLE {schema}.upload_receipts DROP CONSTRAINT fail_cleaning")

    start = Barrier(2)

    def retry():
        start.wait(timeout=10)
        return service.publish_cleaning(reviewed, request_id)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = (executor.submit(retry) for _ in range(2))
        cleaned = first.result()
        assert second.result() == cleaned
    assert cleaned.dataset_id == source.dataset_id and cleaned.version_id != source.version_id
    assert set(service.list_datasets()) == {source, cleaned}
    assert service.full_csv(source.version_id) == original
    assert service.preview(cleaned.version_id) == [
        {"label, name": "first", "number": 9007199254740993},
        {"label, name": None, "number": 7},
        {"label, name": "", "number": -2},
    ]
    request = WorkspaceColumnSummary(version_id=cleaned.version_id, column="number")
    assert service.summarize(request).values["total"] == 9007199254740998
    assert service.full_csv(cleaned.version_id).row_count == 3
    state.current = None
    with pytest.raises(PermissionError):
        service.publish_cleaning(reviewed, request_id)
    state.current = access
    service.delete(cleaned.version_id)
    with pytest.raises(UploadRequestConflict, match="deleted"):
        service.publish_cleaning(reviewed, request_id)
    assert service.full_csv(source.version_id) == original


def test_pilot_evidence_follows_the_authorized_version(
    make_workspace, workspace_database, membership_reader, monkeypatch,
):
    service, access, _, _, reader, writer = make_workspace("w2-alice")
    large = service.upload(pd.DataFrame({"value": range(1, 52)}), "same.csv")
    small = service.upload(pd.DataFrame({"value": [101, 102]}), "same.csv")
    issuer, subject = "https://issuer.example", "w2-alice"
    claims = {"is_logged_in": True, "iss": issuer, "sub": subject, "exp": int(time()) + 60}
    monkeypatch.setattr(st, "user", SimpleNamespace(to_dict=lambda: dict(claims), is_logged_in=True))
    registry = MembershipRegistry(membership_reader)
    with workspace_database.begin() as connection:
        grant_membership(connection, issuer, subject, access.workspace_id, can_manage=True)
    app = AppTest.from_function(_workspace_app, args=(registry.lookup, lambda _: (reader, writer))).run()
    assert len(set(app.selectbox[0].options)) == 2

    def click(label):
        next(button for button in app.button if button.label == label).click().run()
        assert not app.exception

    def assert_csv(version, expected, status, *, full=False):
        label = "Full dataset CSV" if full else "Preview CSV"
        click("Prepare full dataset CSV" if full else "Prepare preview CSV")
        assert f"{label}: {len(expected)} data rows ({status}). Version: {version}" in [
            caption.value for caption in app.caption
        ]
        frame = app.get("iframe")[0].proto
        if hasattr(st, "iframe"):
            assert frame.alt == f"{label} download"
        source = frame.srcdoc
        assert f"Save {label}" in source
        encoded = source.split("base64,", 1)[1].split('"', 1)[0]
        rows = list(csv.reader(StringIO(b64decode(encoded).decode("utf-8-sig"))))
        assert rows == [["value"], *[[str(value)] for value in expected]]

    app.selectbox[0].set_value(large.version_id).run()
    click("Run analysis")
    assert [item.value for item in app.text] == ["Rows: 51"]
    assert f"Analysis scope: Entire dataset. Version: {large.version_id}" in [
        caption.value for caption in app.caption
    ]
    assert_csv(large.version_id, range(1, 51), "truncated")
    assert_csv(large.version_id, range(1, 52), "complete", full=True)
    click("Preview whitespace trim")
    assert f"Whitespace trim preview for version {large.version_id}. 0 of 51 rows would change." in [
        caption.value for caption in app.caption
    ]
    assert len(app.get("html")) == 2
    click("Approve and verify replay")
    assert "Replay matched the reviewed preview." in app.success[0].value

    app.selectbox[0].set_value(small.version_id).run()
    assert not app.text and not app.get("iframe")
    assert len(app.get("html")) == 1
    assert all(button.label != "Approve and verify replay" for button in app.button)
    assert all(str(large.version_id) not in caption.value for caption in app.caption)
    app.radio[0].set_value("Summarize column").run()
    click("Run analysis")
    assert {"Column: value", "Sum: 203"} <= {item.value for item in app.text}
    assert f"Analysis scope: Entire dataset. Version: {small.version_id}" in [
        caption.value for caption in app.caption
    ]
    assert_csv(small.version_id, [101, 102], "not truncated")
    assert_csv(small.version_id, [101, 102], "complete", full=True)

    service.delete(small.version_id)
    app.run()
    assert not app.exception and app.selectbox[0].value is None
    assert not app.text and not app.get("iframe")
    assert all(not button.label.startswith("Prepare ") for button in app.button)

    app.selectbox[0].set_value(large.version_id).run()
    assert_csv(large.version_id, range(1, 52), "complete", full=True)
    with workspace_database.begin() as connection:
        revoke_membership(connection, issuer, subject)
    app.run()
    assert not app.exception and not app.selectbox
    assert not app.text and not app.get("iframe") and not app.caption
    assert all(not button.label.startswith("Prepare ") for button in app.button)


def test_pilot_publishes_reviewed_cleaning_and_reuses_request_on_retry(
    make_workspace, workspace_database, membership_reader, monkeypatch,
):
    service, access, _, schema, reader, writer = make_workspace("w2-manager")
    source = service.upload_csv(b'label,value\n"  first  ",7\n" second ",9\n', "source.csv", uuid4())
    original = service.full_csv(source.version_id)
    issuer, subject = "https://issuer.example", "w2-manager"
    claims = {"is_logged_in": True, "iss": issuer, "sub": subject, "exp": int(time()) + 60}
    monkeypatch.setattr(st, "user", SimpleNamespace(to_dict=lambda: dict(claims), is_logged_in=True))
    registry = MembershipRegistry(membership_reader)
    with workspace_database.begin() as connection:
        grant_membership(connection, issuer, subject, access.workspace_id, can_manage=True)
    app = AppTest.from_function(_workspace_app, args=(registry.lookup, lambda _: (reader, writer))).run()

    def click(label):
        next(button for button in app.button if button.label == label).click().run()
        assert not app.exception

    app.selectbox[0].set_value(source.version_id).run()
    click("Preview whitespace trim")
    assert all(button.label != "Publish cleaned version" for button in app.button)
    click("Approve and verify replay")

    attempts = []

    @event.listens_for(writer, "before_cursor_execute")
    def fail_first_receipt(connection, cursor, statement, parameters, context, executemany):
        if statement.startswith(f"INSERT INTO {schema}.upload_receipts"):
            attempts.append(parameters["request_id"])
            if len(attempts) == 1:
                raise SQLAlchemyError("synthetic receipt failure")

    click("Publish cleaned version")
    assert "again to retry" in app.warning[0].value
    assert service.list_datasets() == (source,)

    click("Publish cleaned version")
    assert len(attempts) == 2 and attempts[0] == attempts[1]
    cleaned, = (dataset for dataset in service.list_datasets() if dataset != source)
    assert app.selectbox[0].value == cleaned.version_id
    assert f"Published cleaned version {cleaned.version_id}." == app.success[0].value
    assert all(button.label != "Publish cleaned version" for button in app.button)
    assert "<td>first</td>" in app.get("html")[0].proto.body
    click("Run analysis")
    assert [item.value for item in app.text] == ["Rows: 2"]
    click("Prepare full dataset CSV")
    encoded = app.get("iframe")[0].proto.srcdoc.split("base64,", 1)[1].split('"', 1)[0]
    assert list(csv.reader(StringIO(b64decode(encoded).decode("utf-8-sig")))) == [
        ["label", "value"], ["first", "7"], ["second", "9"],
    ]
    click("Delete dataset")
    click("Confirm deletion")
    assert service.list_datasets() == (source,)
    assert service.full_csv(source.version_id) == original

    app = AppTest.from_function(_workspace_app, args=(registry.lookup, lambda _: (reader, writer))).run()
    app.selectbox[0].set_value(source.version_id).run()
    click("Preview whitespace trim")
    click("Approve and verify replay")
    with workspace_database.begin() as connection:
        revoke_membership(connection, issuer, subject)
    click("Publish cleaned version")
    assert not app.selectbox and not app.get("html")
    assert all(button.label != "Publish cleaned version" for button in app.button)
    assert service.list_datasets() == (source,)


def test_pilot_summary_chart_uses_authorized_aggregate_values(
    make_workspace, workspace_database, membership_reader, monkeypatch,
):
    service, access, _, _, reader, writer = make_workspace("chart-viewer")
    dataset = service.upload(pd.DataFrame({
        "mixed": [-0.000002, 0.000004],
        "negative": [-7, -1],
        "exact": [9007199254740992, 9007199254740993],
        "text": ["a", "b"],
    }), "summary.csv")
    issuer, subject = "https://issuer.example", "chart-viewer"
    claims = {"is_logged_in": True, "iss": issuer, "sub": subject, "exp": int(time()) + 60}
    monkeypatch.setattr(st, "user", SimpleNamespace(to_dict=lambda: dict(claims), is_logged_in=True))
    registry = MembershipRegistry(membership_reader)
    with workspace_database.begin() as connection:
        grant_membership(connection, issuer, subject, access.workspace_id, can_manage=False)
    app = AppTest.from_function(_workspace_app, args=(registry.lookup, lambda _: (reader, writer))).run()
    app.selectbox[0].set_value(dataset.version_id).run()

    def analyze():
        next(button for button in app.button if button.label == "Run analysis").click().run()
        assert not app.exception

    analyze()
    assert not app.get("plotly_chart")
    source = app.get("iframe")[0].proto.srcdoc
    assert "Save Analysis CSV" in source
    encoded = source.split("base64,", 1)[1].split('"', 1)[0]
    count, = csv.DictReader(StringIO(b64decode(encoded).decode("utf-8-sig")))
    assert (count["operation"], count["column"], count["metric"], count["value"]) == (
        "count_rows", "", "row_count", "2",
    )
    app.radio[0].set_value("Summarize column").run()

    def summarize(column):
        next(box for box in app.selectbox if box.label == "Column").set_value(column).run()
        assert not app.get("plotly_chart") and not app.get("iframe")
        analyze()

    summarize("mixed")
    mixed = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert mixed["data"][0]["x"] == ["Minimum", "Mean", "Maximum"]
    assert mixed["data"][0]["y"] == [-0.000002, 0.000001, 0.000004]
    assert mixed["data"][0]["customdata"] == ["-2e-06", "1e-06", "4e-06"]
    assert mixed["layout"]["yaxis"] == {"rangemode": "tozero"}
    assert mixed["layout"]["meta"]["version_id"] == str(dataset.version_id)
    encoded = app.get("iframe")[0].proto.srcdoc.split("base64,", 1)[1].split('"', 1)[0]
    exported = list(csv.DictReader(StringIO(b64decode(encoded).decode("utf-8-sig"))))
    assert {row["result_id"] for row in exported} == {mixed["layout"]["meta"]["result_id"]}
    assert {row["version_id"] for row in exported} == {str(dataset.version_id)}
    metrics = {row["metric"]: row["value"] for row in exported}
    assert [metrics[key] for key in ("minimum", "mean", "maximum")] == mixed["data"][0]["customdata"]
    summarize("negative")
    negative = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert negative["data"][0]["y"] == [-7, -4, -1]
    assert negative["layout"]["meta"]["result_id"] != mixed["layout"]["meta"]["result_id"]
    summarize("exact")
    assert not app.get("plotly_chart")
    assert "Maximum: 9007199254740993" in [item.value for item in app.text]
    summarize("text")
    assert not app.get("plotly_chart")
    summarize("mixed")
    with workspace_database.begin() as connection:
        revoke_membership(connection, issuer, subject)
    app.run()
    assert not app.exception and not app.get("plotly_chart") and not app.text and not app.get("iframe")


def test_pilot_restores_saved_analysis_and_keeps_csv_on_save_failure(
    make_workspace, workspace_database, membership_reader, monkeypatch,
):
    service, access, _, schema, reader, writer = make_workspace("save-manager")
    dataset = service.upload(pd.DataFrame({"value": [-3, 5]}), "summary.csv")
    issuer, subject = "https://issuer.example", "save-manager"
    claims = {"is_logged_in": True, "iss": issuer, "sub": subject, "exp": int(time()) + 60}
    monkeypatch.setattr(st, "user", SimpleNamespace(to_dict=lambda: dict(claims), is_logged_in=True))
    registry = MembershipRegistry(membership_reader)
    with workspace_database.begin() as connection:
        create_analysis_results(connection, access.workspace_id)
        grant_membership(connection, issuer, subject, access.workspace_id, can_manage=True)
    app = AppTest.from_function(_workspace_app, args=(registry.lookup, lambda _: (reader, writer))).run()
    app.selectbox[0].set_value(dataset.version_id).run()
    app.radio[0].set_value("Summarize column").run()

    def click(label):
        next(button for button in app.button if button.label == label).click().run()
        assert not app.exception

    click("Run analysis")
    original_csv = app.get("iframe")[0].proto.srcdoc
    result_id = json.loads(app.get("plotly_chart")[0].proto.spec)["layout"]["meta"]["result_id"]
    attempts = []

    @event.listens_for(writer, "before_cursor_execute")
    def fail_first_save(connection, cursor, statement, parameters, context, executemany):
        if statement.startswith(f"INSERT INTO {schema}.analysis_results"):
            attempts.append(parameters["result_id"])
            if len(attempts) == 1:
                raise SQLAlchemyError("synthetic result storage failure")

    click("Save result")
    assert "could not be saved" in app.warning[0].value
    assert app.get("iframe")[0].proto.srcdoc == original_csv
    click("Save result")
    assert str(attempts[0]) == str(attempts[1]) == result_id
    assert app.success[0].value == f"Saved analysis result {result_id}."

    app = AppTest.from_function(_workspace_app, args=(registry.lookup, lambda _: (reader, writer))).run()
    app.selectbox[0].set_value(dataset.version_id).run()
    assert not app.text and not app.get("iframe")
    next(widget for widget in app.text_input if widget.label == "Saved result ID").set_value(result_id)
    click("Restore result")
    assert app.get("iframe")[0].proto.srcdoc == original_csv
    restored = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert restored["layout"]["meta"]["result_id"] == result_id
    assert "Sum: 2" in [item.value for item in app.text]
    with workspace_database.begin() as connection:
        revoke_membership(connection, issuer, subject)
    app.run()
    assert not app.exception and not app.get("plotly_chart") and not app.get("iframe")


def test_pending_delete_cannot_survive_membership_change(
    make_workspace, workspace_database, membership_reader, monkeypatch,
):
    service, access, _, _, reader, writer = make_workspace("alice")
    dataset = service.upload(pd.DataFrame({"value": [1], "label": [" sample "]}), "private.csv")
    claims = {"is_logged_in": True, "iss": "https://issuer.example", "sub": "alice", "exp": int(time()) + 60}
    monkeypatch.setattr(st, "user", SimpleNamespace(to_dict=lambda: dict(claims), is_logged_in=True))
    registry = MembershipRegistry(membership_reader)
    with workspace_database.begin() as connection:
        grant_membership(connection, claims["iss"], claims["sub"], access.workspace_id, can_manage=True)

    def engines(workspace_id):
        assert workspace_id == access.workspace_id
        return reader, writer

    app = AppTest.from_function(_workspace_app, args=(registry.lookup, engines)).run()
    app.selectbox[0].set_value(dataset.version_id).run()
    app.text_input[0].set_value("no match").run()
    assert not app.selectbox and not app.get("html")
    assert app.info[0].value == "No matching datasets."
    app.text_input[0].set_value("VaLuE").run()
    assert app.selectbox[0].value is None
    app.selectbox[0].set_value(dataset.version_id).run()
    next(button for button in app.button if button.label == "Preview whitespace trim").click().run()
    assert any(button.label == "Approve and verify replay" for button in app.button)
    next(box for box in app.selectbox if box.label == "Column to trim").set_value("label").run()
    assert all(button.label != "Approve and verify replay" for button in app.button)
    next(box for box in app.selectbox if box.label == "Column to trim").set_value("value").run()
    assert all(button.label != "Approve and verify replay" for button in app.button)
    next(button for button in app.button if button.label == "Run analysis").click().run()
    assert [item.value for item in app.text] == ["Rows: 1"]
    app.radio[0].set_value("Summarize column").run()
    next(button for button in app.button if button.label == "Run analysis").click().run()
    assert "Sum: 1" in [item.value for item in app.text]
    next(button for button in app.button if button.label == "Prepare preview CSV").click().run()
    assert len(app.get("iframe")) == 1
    next(button for button in app.button if button.label == "Delete dataset").click().run()
    assert "Delete this dataset version? This permanently removes its rows and saved analysis results." in [
        warning.value for warning in app.warning
    ]
    confirm = next(button for button in app.button if button.label == "Confirm deletion")

    with workspace_database.begin() as connection:
        grant_membership(connection, claims["iss"], claims["sub"], access.workspace_id, can_manage=False)
    confirm.click().run()
    assert not app.exception
    assert not app.get("html") and not app.get("iframe")
    assert not app.get("file_uploader")
    assert all(button.label != "Confirm deletion" for button in app.button)
    assert service.metadata(dataset.version_id) == dataset

    app.selectbox[0].set_value(dataset.version_id).run()
    assert len(app.get("html")) == 1
    prepare = next(button for button in app.button if button.label == "Prepare preview CSV")
    claims["exp"] = int(time()) - 1
    prepare.click().run()
    assert not app.get("html") and not app.get("iframe")
    assert not app.selectbox
