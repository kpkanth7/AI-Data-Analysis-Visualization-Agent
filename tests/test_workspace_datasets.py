"""Exercise private dataset operations with actual PostgreSQL roles."""

import csv
import json
import shutil
import subprocess
import sys
from base64 import b64decode
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier, Event
from time import time
from types import SimpleNamespace
from uuid import uuid4

import pandas as pd
import pytest
import streamlit as st
from pydantic import ValidationError
from sqlalchemy import create_engine, event, inspect
from sqlalchemy.exc import DBAPIError, IntegrityError, OperationalError, ProgrammingError, SQLAlchemyError
from sqlalchemy.pool import NullPool
from streamlit.testing.v1 import AppTest

from db.demo_catalog import DEMO_READER
from db.workspace_datasets import (
    ANALYSIS_RUN_CALL_LIMIT, MAX_UPLOAD_ROWS, ROW_KEY, WORKSPACE_UPLOADS_PER_DAY, CleaningPreviewChanged,
    UploadLimitReached, UploadRequestConflict,
    WorkspaceAccess, WorkspaceColumnSummary, WorkspaceDatasets, WorkspaceRowCount, WorkspaceTrimWhitespace,
    create_analysis_results, create_analysis_run_purge, create_analysis_runs, create_upload_receipts,
    purge_expired_analysis_runs, version_catalog,
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


def test_raw_and_cleaned_versions_share_admission_through_accounting_outage(
    make_workspace, workspace_database,
):
    service, access, _, schema, reader, writer = make_workspace("upload-manager")
    other_access = replace(access, principal_id="cleaning-manager", session_id=uuid4())
    cleaner = WorkspaceDatasets(other_access, lambda: other_access, reader, writer)
    payload = b'label\n"  example  "\n'
    existing = [
        service.upload_csv(payload, "source.csv", uuid4())
        for _ in range(WORKSPACE_UPLOADS_PER_DAY - 1)
    ]
    source = existing[0]
    approved = cleaner.preview_cleaning(WorkspaceTrimWhitespace(version_id=source.version_id, column="label"))
    upload_id, cleaning_id = uuid4(), uuid4()

    def upload():
        return service.upload_csv(payload, "new.csv", upload_id)

    def publish():
        return cleaner.publish_cleaning(approved, cleaning_id)

    def receipt_count():
        with workspace_database.connect() as connection:
            return connection.exec_driver_sql(f"SELECT count(*) FROM {schema}.upload_receipts").scalar_one()

    tables = set(inspect(workspace_database).get_table_names(schema=schema))
    with workspace_database.begin() as connection:
        connection.exec_driver_sql(f"REVOKE SELECT ON {schema}.upload_receipts FROM {schema}_writer")
    try:
        for submit in (upload, publish):
            with pytest.raises(ProgrammingError) as denied:
                submit()
            assert denied.value.orig.pgcode == "42501"
            assert f"{schema}.upload_receipts" in denied.value.statement
        assert set(service.list_datasets()) == set(existing)
        assert set(inspect(workspace_database).get_table_names(schema=schema)) == tables
        assert receipt_count() == WORKSPACE_UPLOADS_PER_DAY - 1
        assert service.preview(source.version_id) == [{"label": "  example  "}]
        other_workspace, *_ = make_workspace("independent-manager")
        assert other_workspace.upload_csv(payload, "other.csv", uuid4()).row_count == 1
    finally:
        with workspace_database.begin() as connection:
            connection.exec_driver_sql(f"GRANT SELECT ON {schema}.upload_receipts TO {schema}_writer")

    ready = Barrier(2)

    def attempt(submit):
        ready.wait(timeout=10)
        try:
            return submit()
        except UploadLimitReached:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(attempt, (upload, publish)))
    winning_version, = (version for version in outcomes if version is not None)
    winning_index = outcomes.index(winning_version)
    retry = (upload, publish)[winning_index]
    denied_request = (publish, upload)[winning_index]
    assert retry() == winning_version
    assert receipt_count() == WORKSPACE_UPLOADS_PER_DAY
    assert set(service.list_datasets()) == {*existing, winning_version}
    with pytest.raises(UploadLimitReached):
        denied_request()
    assert len(set(inspect(workspace_database).get_table_names(schema=schema)) - tables) == 1


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


def test_preview_access_checks_all_projected_columns_without_returning_rows(make_workspace, workspace_database):
    service, _, _, schema, reader, _ = make_workspace("preview-check")
    version = service.upload(pd.DataFrame({"label": ["private"], "value": [12]}), "preview.csv")
    relation = f"{schema}.v_{version.version_id.hex}"
    row_counts = []

    @event.listens_for(reader, "after_cursor_execute")
    def record_result(connection, cursor, statement, parameters, context, executemany):
        if f"FROM {relation}" in statement:
            row_counts.append(cursor.rowcount)

    service.check_preview_access(version.version_id)
    assert row_counts == [0]
    with workspace_database.begin() as connection:
        connection.exec_driver_sql(f"REVOKE SELECT ON {relation} FROM {schema}_reader")
        connection.exec_driver_sql(f"GRANT SELECT (label, {ROW_KEY}) ON {relation} TO {schema}_reader")

    assert service.metadata(version.version_id) == version
    with pytest.raises(ProgrammingError) as denied:
        service.check_preview_access(version.version_id)
    assert denied.value.orig.pgcode == "42501"


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


def test_summary_handles_existing_numeric_types_and_missing_columns(make_workspace, workspace_database):
    workspace, _, _, schema, *_ = make_workspace("alice")
    version = workspace.upload(pd.DataFrame({
        "Exact amount": ["9007199254740993.01", "0.02", None],
        "Removed": [1, 2, 3],
    }), "typed.csv")
    with workspace_database.begin() as connection:
        connection.exec_driver_sql(
            f'ALTER TABLE {schema}.v_{version.version_id.hex} '
            'ALTER COLUMN "Exact amount" TYPE numeric USING "Exact amount"::numeric, DROP COLUMN "Removed"'
        )

    request = WorkspaceColumnSummary(version_id=version.version_id, column="Exact amount")
    summary = workspace.summarize(request)
    assert summary.values["non_null_count"] == 2
    assert summary.values["total"] == Decimal("9007199254740993.03")
    with pytest.raises(PermissionError, match="column is unavailable"):
        workspace.summarize(WorkspaceColumnSummary(version_id=version.version_id, column="Removed"))


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


@pytest.fixture
def analysis_run(make_workspace, workspace_database):
    _, access, state, schema, reader, writer = make_workspace("alice")
    access = replace(access, membership_revision=uuid4())
    state.current = access
    service = WorkspaceDatasets(access, lambda: state.current, reader, writer)
    dataset = service.upload(pd.DataFrame({"value": [9007199254740993, -2]}), "analysis.csv")
    request = WorkspaceColumnSummary(version_id=dataset.version_id, column="value")
    scope = {"version_id": dataset.version_id, "conversation_id": uuid4()}
    with workspace_database.begin() as connection:
        create_analysis_runs(connection, access.workspace_id)
    started = service.start_analysis_run(request, conversation_id=scope["conversation_id"])
    return SimpleNamespace(
        service=service, access=access, state=state, schema=schema, reader=reader, writer=writer,
        request=request, scope=scope, started=started, run_id=started.run_id,
    )


def test_analysis_run_restores_exact_completed_work_in_a_fresh_session(analysis_run, monkeypatch):
    run = analysis_run
    first_attempt = uuid4()
    first = run.service.advance_analysis_run(run.run_id, attempt_id=first_attempt, **run.scope)
    assert first.calls_used == 1 and not first.complete
    assert first.results[0].values == {"row_count": 2}

    access = replace(run.access, session_id=uuid4())
    run.state.current = access
    fresh = WorkspaceDatasets(access, lambda: run.state.current, run.reader, run.writer)
    calls = []
    summarize = fresh.summarize

    def track(request):
        calls.append(request)
        return summarize(request)

    monkeypatch.setattr(fresh, "summarize", track)
    restored = fresh.resume_analysis_run(run.run_id, **run.scope)
    assert restored.results == (replace(first.results[0], access=access),)
    assert restored.expires_at == run.started.expires_at
    assert fresh.advance_analysis_run(run.run_id, attempt_id=first_attempt, **run.scope) == restored
    assert calls == []

    finished = fresh.advance_analysis_run(run.run_id, attempt_id=uuid4(), **run.scope)
    assert finished.complete and finished.calls_used == 2
    assert calls == [run.request]
    assert finished.results[0] == restored.results[0]
    assert finished.results[1].values["maximum"] == 9007199254740993
    assert finished.results[1].values == summarize(run.request).values
    assert fresh.advance_analysis_run(run.run_id, attempt_id=uuid4(), **run.scope) == finished
    assert calls == [run.request]


def test_analysis_run_survives_a_fresh_python_process(analysis_run):
    run = analysis_run
    first = run.service.advance_analysis_run(run.run_id, attempt_id=uuid4(), **run.scope)
    configuration = {
        "socket": run.reader.url.query["host"],
        "schema": run.schema,
        "principal": run.access.principal_id,
        "workspace": str(run.access.workspace_id),
        "revision": str(run.access.membership_revision),
        "run": str(run.run_id),
        "version": str(run.scope["version_id"]),
        "conversation": str(run.scope["conversation_id"]),
    }
    process = subprocess.run(
        [sys.executable, "-c", """
import json
import sys
from uuid import UUID, uuid4

from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.pool import NullPool

from db.workspace_datasets import WorkspaceAccess, WorkspaceDatasets

config = json.load(sys.stdin)
url = URL.create("postgresql+psycopg2", database="postgres", query={"host": config["socket"]})
reader, writer = (
    create_engine(url.set(username=f"{config['schema']}_{role}"), poolclass=NullPool)
    for role in ("reader", "writer")
)
access = WorkspaceAccess(config["principal"], uuid4(), UUID(config["workspace"]), True, UUID(config["revision"]))
service = WorkspaceDatasets(access, lambda: access, reader, writer)
scope = {"version_id": UUID(config["version"]), "conversation_id": UUID(config["conversation"])}
finished = service.advance_analysis_run(UUID(config["run"]), attempt_id=uuid4(), **scope)
print(json.dumps({
    "results": [str(result.result_id) for result in finished.results],
    "calls_used": finished.calls_used,
    "complete": finished.complete,
}))
"""],
        input=json.dumps(configuration), capture_output=True, text=True, check=True, timeout=15,
        cwd=Path(__file__).resolve().parents[1],
    )
    finished = run.service.resume_analysis_run(run.run_id, **run.scope)
    assert json.loads(process.stdout) == {
        "results": [str(result.result_id) for result in finished.results],
        "calls_used": 2,
        "complete": True,
    }
    assert finished.complete and finished.results[0] == first.results[0]
    assert finished.results[1].request == run.request


def test_analysis_run_reserves_once_and_rejects_late_completion(analysis_run, workspace_database, monkeypatch):
    run = analysis_run
    first_attempt = uuid4()
    started, release = Event(), Event()
    calls = []
    summarize = run.service.summarize

    def hold_first(request):
        calls.append(request)
        result = summarize(request)
        if len(calls) == 1:
            started.set()
            assert release.wait(timeout=10)
            return replace(result, result_id=late_result_id)
        return result

    late_result_id = uuid4()
    monkeypatch.setattr(run.service, "summarize", hold_first)
    with ThreadPoolExecutor(max_workers=1) as workers:
        pending = workers.submit(run.service.advance_analysis_run, run.run_id, attempt_id=first_attempt, **run.scope)
        try:
            assert started.wait(timeout=10)
            duplicate = run.service.advance_analysis_run(run.run_id, attempt_id=first_attempt, **run.scope)
            assert duplicate.in_progress and duplicate.calls_used == 1
            assert run.service.advance_analysis_run(run.run_id, attempt_id=uuid4(), **run.scope) == duplicate
            assert len(calls) == 1
            with workspace_database.begin() as connection:
                connection.exec_driver_sql(
                    f"UPDATE {run.schema}.analysis_runs SET lease_until = clock_timestamp() - interval '1 second' "
                    "WHERE run_id = %s", (str(run.run_id),),
                )
            interrupted = run.service.resume_analysis_run(run.run_id, **run.scope)
            assert not interrupted.in_progress and interrupted.last_attempt_failed and interrupted.calls_used == 1
            assert run.service.advance_analysis_run(run.run_id, attempt_id=first_attempt, **run.scope) == interrupted
            recovered = run.service.advance_analysis_run(run.run_id, attempt_id=uuid4(), **run.scope)
            assert recovered.calls_used == 2 and len(recovered.results) == 1
        finally:
            release.set()
        assert pending.result(timeout=10) == recovered
    assert len(calls) == 2 and recovered.results[0].result_id != late_result_id


def test_analysis_run_failed_attempts_share_the_original_call_budget(analysis_run, monkeypatch):
    run = analysis_run
    first = run.service.advance_analysis_run(run.run_id, attempt_id=uuid4(), **run.scope)
    calls = []

    def fail(request):
        calls.append(request)
        raise ProgrammingError(None, None, RuntimeError("synthetic failure"))

    monkeypatch.setattr(run.service, "summarize", fail)
    for expected in range(2, ANALYSIS_RUN_CALL_LIMIT + 1):
        attempt = uuid4()
        status = run.service.advance_analysis_run(run.run_id, attempt_id=attempt, **run.scope)
        assert status.last_attempt_failed and not status.in_progress
        assert status.calls_used == expected and status.results == first.results
        assert run.service.advance_analysis_run(run.run_id, attempt_id=attempt, **run.scope) == status
    assert run.service.advance_analysis_run(run.run_id, attempt_id=uuid4(), **run.scope) == status
    assert status.calls_remaining == 0
    assert calls == [run.request] * (ANALYSIS_RUN_CALL_LIMIT - 1)


def test_analysis_run_rechecks_ownership_and_revocation_during_a_read(
    analysis_run, make_workspace, workspace_database, monkeypatch,
):
    run = analysis_run
    for access in (
        replace(run.access, principal_id="bob"),
        replace(run.access, can_manage=False),
        replace(run.access, membership_revision=None),
        replace(run.access, membership_revision=uuid4()),
    ):
        actor = WorkspaceDatasets(access, lambda: access, run.reader, run.writer)
        with pytest.raises(PermissionError):
            actor.resume_analysis_run(run.run_id, **run.scope)

    for field in ("conversation_id", "version_id"):
        with pytest.raises(PermissionError):
            run.service.resume_analysis_run(run.run_id, **{**run.scope, field: uuid4()})
    _, foreign, _, _, reader, writer = make_workspace(run.access.principal_id)
    foreign = replace(foreign, membership_revision=run.access.membership_revision)
    with workspace_database.begin() as connection:
        create_analysis_runs(connection, foreign.workspace_id)
    other_workspace = WorkspaceDatasets(foreign, lambda: foreign, reader, writer)
    with pytest.raises(PermissionError):
        other_workspace.resume_analysis_run(run.run_id, **run.scope)
    with monkeypatch.context() as policy:
        policy.setattr("db.workspace_datasets._ANALYSIS_RUN_POLICY", 2)
        with pytest.raises(PermissionError):
            run.service.resume_analysis_run(run.run_id, **run.scope)
    bad_request = run.request.model_copy(update={"column": "hidden"})
    with pytest.raises(PermissionError):
        run.service.start_analysis_run(bad_request, conversation_id=uuid4())

    summarize = run.service.summarize

    def revoke(request):
        result = summarize(request)
        run.state.current = None
        return result

    monkeypatch.setattr(run.service, "summarize", revoke)
    with pytest.raises(PermissionError):
        run.service.advance_analysis_run(run.run_id, attempt_id=uuid4(), **run.scope)
    with pytest.raises(PermissionError):
        run.service.resume_analysis_run(run.run_id, **run.scope)
    run.state.current = run.access
    recovered = run.service.resume_analysis_run(run.run_id, **run.scope)
    assert recovered.calls_used == 1 and recovered.in_progress
    assert recovered.results == ()


def test_analysis_run_expiry_purge_and_version_deletion(analysis_run, workspace_database):
    run = analysis_run
    with workspace_database.begin() as connection:
        remaining = connection.exec_driver_sql(
            f"SELECT expires_at - clock_timestamp() FROM {run.schema}.analysis_runs",
        ).scalar_one()
        assert timedelta(hours=23, minutes=59) < remaining <= timedelta(hours=24)
        assert purge_expired_analysis_runs(connection, run.access.workspace_id) == 0
        connection.exec_driver_sql(f"UPDATE {run.schema}.analysis_runs SET expires_at = clock_timestamp()")
    with pytest.raises(PermissionError):
        run.service.resume_analysis_run(run.run_id, **run.scope)
    with pytest.raises(PermissionError):
        run.service.advance_analysis_run(run.run_id, attempt_id=uuid4(), **run.scope)
    with workspace_database.begin() as connection:
        assert purge_expired_analysis_runs(connection, run.access.workspace_id) == 1

    fresh = run.service.start_analysis_run(run.request, conversation_id=run.scope["conversation_id"])
    with run.reader.connect() as connection, pytest.raises(ProgrammingError) as denied:
        connection.exec_driver_sql(f"SELECT run_id FROM {run.schema}.analysis_runs")
    assert denied.value.orig.pgcode == "42501"
    with run.writer.connect() as connection, pytest.raises(ProgrammingError) as denied:
        connection.exec_driver_sql(f"UPDATE {run.schema}.analysis_runs SET expires_at = clock_timestamp()")
    assert denied.value.orig.pgcode == "42501"
    run.service.delete(run.scope["version_id"])
    with pytest.raises(PermissionError):
        run.service.resume_analysis_run(fresh.run_id, **run.scope)
    with workspace_database.connect() as connection:
        assert connection.exec_driver_sql(f"SELECT count(*) FROM {run.schema}.analysis_runs").scalar_one() == 0


def test_workspace_backup_is_quarantined_until_deletion_and_revocation_are_replayed(
    make_workspace, workspace_database, membership_reader, tmp_path,
):
    dump_tool, restore_tool = (shutil.which(name) for name in ("pg_dump", "pg_restore"))
    if not dump_tool or not restore_tool:
        pytest.skip("Local PostgreSQL pg_dump and pg_restore are required")

    owner = workspace_database
    _, access, _, schema, reader, writer = make_workspace("backup-drill")
    manager = ("https://issuer.example", "backup-manager")
    revoked = ("https://issuer.example", "backup-former-member")

    def manager_service(registry_engine, read_engine, write_engine):
        registry = MembershipRegistry(registry_engine)
        session_id = uuid4()

        def resolve():
            membership = registry.lookup(*manager)
            return WorkspaceAccess(
                membership.principal_id, session_id, membership.workspace_id,
                membership.can_manage, membership.revision,
            ) if membership else None

        binding = resolve()
        assert binding is not None
        return WorkspaceDatasets(binding, resolve, read_engine, write_engine)

    with owner.begin() as connection:
        grant_membership(connection, *manager, access.workspace_id, can_manage=True)
        grant_membership(connection, *revoked, access.workspace_id, can_manage=False)
        create_analysis_results(connection, access.workspace_id)
        create_analysis_runs(connection, access.workspace_id)

    service = manager_service(membership_reader, reader, writer)
    deleted_upload_id = uuid4()
    removed_bytes, removed_name = b"value\n91\n", "remove.csv"
    removed = service.upload_csv(removed_bytes, removed_name, deleted_upload_id)
    retained = service.upload_csv(b"value\n12\n30\n", "retain.csv", uuid4())
    expected_csv = service.full_csv(retained.version_id).data
    retained_summary = service.summarize(WorkspaceColumnSummary(version_id=retained.version_id, column="value"))
    service.save_analysis(retained_summary)
    removed_summary = service.summarize(WorkspaceColumnSummary(version_id=removed.version_id, column="value"))
    service.save_analysis(removed_summary)

    def start_run(version):
        scope = {"version_id": version.version_id, "conversation_id": uuid4()}
        run = service.start_analysis_run(
            WorkspaceColumnSummary(version_id=version.version_id, column="value"),
            conversation_id=scope["conversation_id"],
        )
        return scope, service.advance_analysis_run(run.run_id, attempt_id=uuid4(), **scope)

    removed_scope, removed_run = start_run(removed)
    retained_scope, retained_run = start_run(retained)
    _, expired_run = start_run(retained)
    with owner.begin() as connection:
        connection.exec_driver_sql(
            f"UPDATE {schema}.analysis_runs SET expires_at=clock_timestamp() WHERE run_id=%s",
            (expired_run.run_id,),
        )

    target = f"restore_{access.workspace_id.hex}"
    cli = ["--host", owner.url.query["host"], "--username", owner.url.username, "--no-password"]
    with TemporaryDirectory(dir=tmp_path) as backup_folder:
        archive = Path(backup_folder) / "workspace.dump"
        archive.touch(mode=0o600)
        subprocess.run(
            [dump_tool, *cli, "--dbname", owner.url.database, "--format=custom",
             "--schema", schema, "--schema", "workspace_auth", "--file", str(archive)],
            check=True, capture_output=True, timeout=30,
        )

        service.delete(removed.version_id)
        with owner.begin() as connection:
            revoke_membership(connection, *revoked)
            assert purge_expired_analysis_runs(connection, access.workspace_id) == 1
        assert {version.version_id for version in service.list_datasets()} == {retained.version_id}
        assert MembershipRegistry(membership_reader).lookup(*revoked) is None
        for read_deleted in (
            lambda: service.full_csv(removed.version_id),
            lambda: service.restore_analysis(removed.version_id, removed_summary.result_id),
            lambda: service.resume_analysis_run(removed_run.run_id, **removed_scope),
        ):
            with pytest.raises(PermissionError):
                read_deleted()

        with owner.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
            connection.exec_driver_sql(f"CREATE DATABASE {target} ALLOW_CONNECTIONS false")
        with owner.begin() as connection:
            connection.exec_driver_sql(f"REVOKE CONNECT ON DATABASE {target} FROM PUBLIC")
            connection.exec_driver_sql(f"ALTER DATABASE {target} ALLOW_CONNECTIONS true")
        subprocess.run(
            [restore_tool, *cli, "--dbname", target, "--exit-on-error", "--single-transaction", str(archive)],
            check=True, capture_output=True, timeout=30,
        )

    engines = [create_engine(engine.url.set(database=target), poolclass=NullPool, hide_parameters=True)
               for engine in (owner, membership_reader, reader, writer)]
    recovered_owner, recovered_registry, recovered_reader, recovered_writer = engines
    try:
        for engine in engines[1:]:
            with pytest.raises(OperationalError, match="permission denied for database"):
                engine.connect()

        with recovered_owner.begin() as connection:
            removed_table = f"{schema}.v_{removed.version_id.hex}"
            assert connection.exec_driver_sql(f"SELECT value FROM {removed_table}").scalar_one() == 91
            assert connection.exec_driver_sql(
                "SELECT active FROM workspace_auth.memberships WHERE issuer=%s AND subject=%s", revoked,
            ).scalar_one() is True
            assert connection.exec_driver_sql(f"SELECT count(*) FROM {schema}.analysis_results").scalar_one() == 2
            assert connection.exec_driver_sql(f"SELECT count(*) FROM {schema}.analysis_runs").scalar_one() == 3

            # Apply the known post-backup deletion and revocation before opening the restored database.
            revoke_membership(connection, *revoked)
            catalog = version_catalog(schema)
            removed_entry = connection.execute(
                catalog.delete().where(catalog.c.version_id == removed.version_id)
            )
            assert removed_entry.rowcount == 1
            connection.exec_driver_sql(f"DROP TABLE {removed_table}")
            assert purge_expired_analysis_runs(connection, access.workspace_id) == 1
            assert connection.exec_driver_sql("SELECT to_regclass(%s)", (removed_table,)).scalar_one() is None
            assert connection.exec_driver_sql(f"SELECT count(*) FROM {schema}.analysis_results").scalar_one() == 1
            assert connection.exec_driver_sql(f"SELECT count(*) FROM {schema}.analysis_runs").scalar_one() == 1
            assert connection.exec_driver_sql(f"SELECT count(*) FROM {schema}.upload_receipts").scalar_one() == 2
            roles = (MEMBERSHIP_READER, f"{schema}_reader", f"{schema}_writer")
            connection.exec_driver_sql(f"GRANT CONNECT ON DATABASE {target} TO {', '.join(roles)}")

        assert MembershipRegistry(recovered_registry).lookup(*revoked) is None
        recovered = manager_service(recovered_registry, recovered_reader, recovered_writer)
        assert recovered.list_datasets() == (retained,)
        assert recovered.full_csv(retained.version_id).data == expected_csv
        saved = recovered.restore_analysis(retained.version_id, retained_summary.result_id)
        assert saved.values == retained_summary.values and saved.result_id == retained_summary.result_id
        resumed = recovered.resume_analysis_run(retained_run.run_id, **retained_scope)
        assert resumed.calls_remaining == retained_run.calls_remaining == 3
        assert resumed.results[0].values == {"row_count": 2}
        with pytest.raises(UploadRequestConflict, match="upload was deleted"):
            recovered.upload_csv(removed_bytes, removed_name, deleted_upload_id)
    finally:
        for engine in engines:
            engine.dispose()


def test_analysis_run_scheduled_purge_is_expiry_only_and_workspace_scoped(
    analysis_run, make_workspace, workspace_database,
):
    run = analysis_run
    live_run = run.service.start_analysis_run(run.request, conversation_id=run.scope["conversation_id"])
    other, other_access, other_state, other_schema, other_reader, other_writer = make_workspace("other-manager")
    other_dataset = other.upload(pd.DataFrame({"value": [0]}), "another.csv")
    other_access = replace(other_access, membership_revision=uuid4())
    other_state.current = other_access
    other = WorkspaceDatasets(other_access, lambda: other_state.current, other_reader, other_writer)
    with workspace_database.begin() as connection:
        create_analysis_runs(connection, other_access.workspace_id)
        for schema, access in ((run.schema, run.access), (other_schema, other_access)):
            connection.exec_driver_sql(f"CREATE ROLE {schema}_purger LOGIN NOINHERIT")
            create_analysis_run_purge(connection, access.workspace_id)
    other.start_analysis_run(
        WorkspaceColumnSummary(version_id=other_dataset.version_id, column="value"), conversation_id=uuid4(),
    )
    with workspace_database.begin() as connection:
        connection.exec_driver_sql(
            f"UPDATE {run.schema}.analysis_runs SET expires_at = clock_timestamp() WHERE run_id = %s",
            (str(run.run_id),),
        )
        connection.exec_driver_sql(f"UPDATE {other_schema}.analysis_runs SET expires_at = clock_timestamp()")

    job = f"SELECT {run.schema}.purge_expired_analysis_runs()"
    scheduler = create_engine(workspace_database.url.set(username=f"{run.schema}_purger"), poolclass=NullPool)
    try:
        for expected in (1, 0):
            with scheduler.begin() as connection:
                assert connection.exec_driver_sql(job).scalar_one() == expected
        for statement in (
            f"SELECT * FROM {run.schema}.analysis_runs",
            f"DELETE FROM {run.schema}.analysis_runs",
            f"TRUNCATE {run.schema}.analysis_runs",
            f"SELECT * FROM {run.schema}.dataset_versions",
            f"CREATE TABLE {run.schema}.unexpected (id integer)",
            f"SELECT {other_schema}.purge_expired_analysis_runs()",
        ):
            with scheduler.begin() as connection, pytest.raises(ProgrammingError) as denied:
                connection.exec_driver_sql(statement)
            assert denied.value.orig.pgcode == "42501"
    finally:
        scheduler.dispose()

    for engine in (run.reader, run.writer):
        with engine.begin() as connection, pytest.raises(ProgrammingError) as denied:
            connection.exec_driver_sql(job)
        assert denied.value.orig.pgcode == "42501"
    assert run.service.resume_analysis_run(live_run.run_id, **run.scope) == live_run
    with workspace_database.connect() as connection:
        assert connection.exec_driver_sql(f"SELECT count(*) FROM {other_schema}.analysis_runs").scalar_one() == 1


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


@pytest.fixture
def manager_run_ui(make_workspace, workspace_database, membership_reader, monkeypatch):
    service, access, _, schema, reader, writer = make_workspace("checkpoint-manager")
    data = pd.DataFrame({"value": [-3, 5], "other": [40, 60]})
    dataset = service.upload(data, "analysis.csv")
    second = service.upload(data, "another.csv")
    issuer, subject = "https://issuer.example", "checkpoint-manager"
    claims = {"is_logged_in": True, "iss": issuer, "sub": subject, "exp": int(time()) + 120}
    monkeypatch.setattr(st, "user", SimpleNamespace(to_dict=lambda: dict(claims), is_logged_in=True))
    registry = MembershipRegistry(membership_reader)
    with workspace_database.begin() as connection:
        grant_membership(connection, issuer, subject, access.workspace_id, can_manage=True)

    def open_workspace(version_id):
        app = AppTest.from_function(_workspace_app, args=(registry.lookup, lambda _: (reader, writer))).run()
        app.selectbox[0].set_value(version_id).run()
        assert not app.exception
        return app

    return SimpleNamespace(
        open=open_workspace, dataset=dataset, second=second, workspace_id=access.workspace_id,
        schema=schema, issuer=issuer, subject=subject, claims=claims,
    )


def test_pilot_manager_resumes_and_retries_without_replaying_completed_work(
    manager_run_ui, workspace_database, monkeypatch,
):
    pilot = manager_run_ui
    option = "Count and summarize column"
    app = pilot.open(pilot.dataset.version_id)
    assert option not in app.radio[0].options
    assert any(button.label == "Run analysis" for button in app.button)
    with workspace_database.begin() as connection:
        create_analysis_runs(connection, pilot.workspace_id)
    app.run()
    app.radio[0].set_value(option).run()

    calls = []
    summarize = WorkspaceDatasets.summarize

    def fail_first_summary(service, request):
        calls.append(request)
        if len(calls) == 2:
            raise ProgrammingError(None, None, RuntimeError("synthetic query failure"))
        return summarize(service, request)

    monkeypatch.setattr(WorkspaceDatasets, "summarize", fail_first_summary)

    def click(app, label):
        next(button for button in app.button if button.label == label).click().run()
        assert not app.exception

    next(box for box in app.selectbox if box.label == "Column for a new run").set_value("value").run()
    click(app, "Start analysis run")
    code = app.get("code")[0].proto.code_text
    count_csv = app.get("iframe")[0].proto.srcdoc
    assert "3 calls remaining." in [item.value for item in app.caption]
    assert "Rows: 2" in [item.value for item in app.text]
    assert calls == [WorkspaceRowCount(version_id=pilot.dataset.version_id)]

    app = pilot.open(pilot.dataset.version_id)
    app.radio[0].set_value(option).run()
    next(box for box in app.selectbox if box.label == "Column for a new run").set_value("other").run()
    next(widget for widget in app.text_input if widget.label == "Resume code").set_value(code)
    click(app, "Resume run")
    assert app.get("iframe")[0].proto.srcdoc == count_csv
    assert "Run column: value" in [item.value for item in app.text]
    assert len(calls) == 1

    click(app, "Continue analysis")
    assert any(item.value == "The last step did not finish." for item in app.warning)
    assert "2 calls remaining." in [item.value for item in app.caption]
    assert app.get("iframe")[0].proto.srcdoc == count_csv
    app.run()
    assert len(calls) == 2
    click(app, "Retry step")
    assert any(item.value == "Analysis run complete." for item in app.success)
    assert "1 call remaining." in [item.value for item in app.caption]
    assert "Sum: 2" in [item.value for item in app.text]
    assert calls[1:] == [WorkspaceColumnSummary(version_id=pilot.dataset.version_id, column="value")] * 2
    assert not any(button.label in {"Continue analysis", "Retry step"} for button in app.button)

    with workspace_database.begin() as connection:
        grant_membership(connection, pilot.issuer, pilot.subject, pilot.workspace_id, can_manage=False)
    app.run()
    assert not app.exception and not app.radio
    assert not app.get("code") and not app.get("iframe")
    assert not any(button.label == "Resume run" for button in app.button)
    app.selectbox[0].set_value(pilot.dataset.version_id).run()
    assert not app.exception and option not in app.radio[0].options
    assert any(button.label == "Run analysis" for button in app.button)


def test_pilot_resume_rejects_wrong_scope_expiry_and_revocation(manager_run_ui, workspace_database):
    pilot = manager_run_ui
    option = "Count and summarize column"
    with workspace_database.begin() as connection:
        create_analysis_runs(connection, pilot.workspace_id)

    def open_run_tools(version_id):
        app = pilot.open(version_id)
        app.radio[0].set_value(option).run()
        return app

    app = open_run_tools(pilot.dataset.version_id)
    next(button for button in app.button if button.label == "Start analysis run").click().run()
    resume_code = app.get("code")[0].proto.code_text
    assert app.get("iframe")

    def assert_denied(app, code, message):
        next(widget for widget in app.text_input if widget.label == "Resume code").set_value(code)
        next(button for button in app.button if button.label == "Resume run").click().run()
        assert not app.exception and not app.get("code") and not app.get("iframe")
        assert any(item.value == message for item in app.warning)

    denied = "This analysis run is unavailable for the selected dataset version."
    wrong_dataset = open_run_tools(pilot.second.version_id)
    assert_denied(wrong_dataset, "invalid", "Enter a valid resume code.")
    assert_denied(wrong_dataset, resume_code, denied)

    with workspace_database.begin() as connection:
        grant_membership(connection, pilot.issuer, "other-manager", pilot.workspace_id, can_manage=True)
    pilot.claims["sub"] = "other-manager"
    assert_denied(open_run_tools(pilot.dataset.version_id), resume_code, denied)
    pilot.claims["sub"] = pilot.subject
    with workspace_database.begin() as connection:
        connection.exec_driver_sql(f"UPDATE {pilot.schema}.analysis_runs SET expires_at = clock_timestamp()")
    app.run()
    assert not app.exception and not app.get("code") and not app.get("iframe")
    assert any(item.value == denied for item in app.warning)
    assert_denied(open_run_tools(pilot.dataset.version_id), resume_code, denied)
    with workspace_database.begin() as connection:
        revoke_membership(connection, pilot.issuer, pilot.subject)
    app.run()
    assert not app.exception and not app.radio and not app.get("iframe")
    assert not any(button.label == "Resume run" for button in app.button)


def test_pilot_delete_discloses_and_removes_only_the_selected_versions_runs(manager_run_ui, workspace_database):
    pilot = manager_run_ui
    with workspace_database.begin() as connection:
        create_analysis_runs(connection, pilot.workspace_id)

    def click(app, label):
        next(button for button in app.button if button.label == label).click().run()
        assert not app.exception

    def start(version_id):
        app = pilot.open(version_id)
        app.radio[0].set_value("Count and summarize column").run()
        click(app, "Start analysis run")
        return app, app.get("code")[0].proto.code_text

    _, other_code = start(pilot.second.version_id)
    app, code = start(pilot.dataset.version_id)
    click(app, "Delete dataset")
    assert any("any resumable analysis runs" in warning.value for warning in app.warning)
    click(app, "Cancel deletion")
    assert app.get("code")[0].proto.code_text == code
    assert app.get("iframe")

    click(app, "Delete dataset")
    click(app, "Confirm deletion")
    assert not app.get("code") and not app.get("iframe")
    assert app.selectbox[0].options == [f"{pilot.second.filename} ({pilot.second.version_id})"]
    with workspace_database.connect() as connection:
        assert str(connection.exec_driver_sql(
            f"SELECT run_id FROM {pilot.schema}.analysis_runs"
        ).scalar_one()) == other_code.split(".")[0]


def test_loaded_pilot_clears_private_state_when_registry_is_unavailable(
    manager_run_ui, workspace_database, caplog,
):
    pilot = manager_run_ui
    with workspace_database.begin() as connection:
        create_analysis_runs(connection, pilot.workspace_id)
    app = pilot.open(pilot.dataset.version_id)
    app.radio[0].set_value("Count and summarize column").run()
    next(button for button in app.button if button.label == "Start analysis run").click().run()
    next(button for button in app.button if button.label == "Delete dataset").click().run()
    assert not app.exception
    assert app.get("html") and app.get("code") and app.get("iframe")
    assert any(button.label == "Confirm deletion" for button in app.button)

    with workspace_database.begin() as connection:
        connection.exec_driver_sql(f"REVOKE SELECT ON workspace_auth.memberships FROM {MEMBERSHIP_READER}")
    try:
        app.run()
        assert not app.exception and not app.selectbox and not app.radio
        assert not app.get("html") and not app.get("iframe") and not app.get("code")
        assert not app.get("file_uploader")
        assert {button.label for button in app.button} == {"Sign out", "Retry connection"}
        assert [error.value for error in app.error] == [
            "Datasets are temporarily unavailable. Please retry in a moment.",
        ]
        service_messages = [record.getMessage() for record in caplog.records if record.name == "ui.service_status"]
        assert service_messages and set(service_messages) == {"Dataset service unavailable: ProgrammingError"}
    finally:
        with workspace_database.begin() as connection:
            connection.exec_driver_sql(f"GRANT SELECT ON workspace_auth.memberships TO {MEMBERSHIP_READER}")

    app.button(key="retry_private_membership").click().run()
    assert not app.exception and not app.error
    assert app.selectbox[0].value is None
    assert not app.get("html") and not app.get("iframe") and not app.get("code")
    assert all(button.label != "Confirm deletion" for button in app.button)
    with workspace_database.connect() as connection:
        assert connection.exec_driver_sql(
            f"SELECT version_id, cardinality(attempt_ids) FROM {pilot.schema}.analysis_runs"
        ).one() == (pilot.dataset.version_id, 1)


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
    assert any("any resumable analysis runs" in warning.value for warning in app.warning)
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
