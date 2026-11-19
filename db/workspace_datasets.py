"""Private dataset lifecycle for a server-authorized workspace."""

import csv
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from io import BytesIO, StringIO
from typing import Annotated
from uuid import UUID, uuid4

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import (
    Boolean, CheckConstraint, Column, Date, DateTime, ForeignKey, Integer,
    LargeBinary, MetaData, Table, Text, delete, func, insert, inspect, literal, or_, select, text, update,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, insert as pg_insert
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import NoSuchTableError, ProgrammingError, SQLAlchemyError
from sqlalchemy.types import Uuid

from core.csv_export import csv_bytes


ROW_KEY = "_workspace_row_id"
WORKSPACE_UPLOADS_PER_DAY = 5
MAX_CSV_BYTES = 1024 * 1024
MAX_UPLOAD_ROWS = 1_000
MAX_UPLOAD_COLUMNS = 64
PREVIEW_ROWS = 50
CATALOG_SEARCH_MAX_CHARS = 128
CATALOG_SEARCH_LIMIT = 50
ANALYSIS_RUN_CALL_LIMIT = 4
_ANALYSIS_RUN_POLICY = 1
_ANALYSIS_RUN_RETENTION = timedelta(hours=24)
_ANALYSIS_ATTEMPT_LEASE = timedelta(seconds=30)


class UploadLimitReached(ValueError):
    pass


class UploadRequestConflict(ValueError):
    pass


class CleaningPreviewChanged(ValueError):
    pass


def _load_csv(contents: bytes) -> pd.DataFrame:
    decoded = contents.decode("utf-8-sig")
    if "\x00" in decoded:
        raise ValueError("CSV contains a NUL character.")
    try:
        records = csv.reader(StringIO(decoded, newline=""), strict=True)
        header = next((row for row in records if row), [])
        if (
            not 1 <= len(header) <= MAX_UPLOAD_COLUMNS
            or len(set(header)) != len(header)
            or any(not name.strip() for name in header)
        ):
            raise ValueError("CSV requires distinct, nonempty column names within the column limit.")
        row_count = 0
        for row in records:
            if not row:
                continue
            row_count += 1
            if len(row) != len(header) or row_count > MAX_UPLOAD_ROWS:
                raise ValueError("CSV rows must match the header and stay within the row limit.")
    except csv.Error as exc:
        raise ValueError("CSV contains an invalid or oversized field.") from exc

    frame = pd.read_csv(BytesIO(contents), nrows=MAX_UPLOAD_ROWS + 1, compression=None)
    if list(frame.columns) != header or len(frame) != row_count:
        raise ValueError("CSV structure could not be read consistently.")
    return frame


@dataclass(frozen=True)
class WorkspaceAccess:
    """Verified by the server. Rotate session_id when authorization changes."""

    principal_id: str
    session_id: UUID
    workspace_id: UUID
    can_manage: bool
    membership_revision: UUID | None = None


@dataclass(frozen=True)
class WorkspaceDataset:
    dataset_id: UUID
    version_id: UUID
    filename: str
    row_count: int
    columns: tuple[str, ...]


class WorkspaceRowCount(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version_id: UUID


class WorkspaceColumnSummary(WorkspaceRowCount):
    column: Annotated[str, Field(min_length=1, max_length=63)]


class WorkspaceTrimWhitespace(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version_id: UUID
    column: Annotated[str, Field(min_length=1, max_length=63)]


@dataclass(frozen=True)
class WorkspaceCleaningPreview:
    plan: WorkspaceTrimWhitespace
    access: WorkspaceAccess
    fingerprint: str
    row_count: int
    changed_count: int
    rows: tuple[tuple[object, object], ...]


@dataclass(frozen=True)
class WorkspaceAnalysis:
    result_id: UUID
    access: WorkspaceAccess
    request: WorkspaceRowCount
    values: dict[str, int | float | Decimal | None]


@dataclass(frozen=True)
class WorkspaceAnalysisRun:
    run_id: UUID
    request: WorkspaceColumnSummary
    expires_at: datetime
    results: tuple[WorkspaceAnalysis, ...]
    calls_used: int
    in_progress: bool
    last_attempt_failed: bool

    @property
    def complete(self) -> bool:
        return len(self.results) == 2

    @property
    def calls_remaining(self) -> int:
        return ANALYSIS_RUN_CALL_LIMIT - self.calls_used


def _encode_metrics(result: WorkspaceAnalysis) -> list:
    return [
        [key, type(value).__name__, str(value)] if value is not None else [key, None, None]
        for key, value in result.values.items()
    ]


def _decode_metrics(metrics: list) -> dict[str, int | float | Decimal | None]:
    types = {"int": int, "float": float, "Decimal": Decimal}
    return {key: types[kind](value) if kind is not None else None for key, kind, value in metrics}


@dataclass(frozen=True)
class WorkspaceCSV:
    version_id: UUID
    data: bytes
    row_count: int
    truncated: bool


def version_catalog(schema: str) -> Table:
    return Table(
        "dataset_versions", MetaData(),
        Column("version_id", Uuid, primary_key=True),
        Column("dataset_id", Uuid, nullable=False),
        Column("filename", Text, nullable=False),
        Column("row_count", Integer, nullable=False),
        Column("columns", ARRAY(Text), nullable=False),
        schema=schema,
    )


def _upload_receipts(schema: str) -> Table:
    return Table(
        "upload_receipts", MetaData(),
        Column("principal_id", Text, primary_key=True),
        Column("request_id", Uuid, primary_key=True),
        Column("fingerprint", LargeBinary),
        Column("version_id", Uuid, nullable=False),
        Column("admitted_on", Date, nullable=False, index=True),
        schema=schema,
    )


def create_upload_receipts(connection: Connection, workspace_id: UUID) -> None:
    """Add upload accounting to a provisioned workspace using its operator connection."""
    schema = f"w_{workspace_id.hex}"
    _upload_receipts(schema).create(connection)
    connection.exec_driver_sql(f"GRANT SELECT, INSERT ON {schema}.upload_receipts TO {schema}_writer")


def _analysis_results(schema: str) -> Table:
    return Table(
        "analysis_results", MetaData(),
        Column("result_id", Uuid, primary_key=True),
        Column("version_id", Uuid, ForeignKey(version_catalog(schema).c.version_id, ondelete="CASCADE"),
               nullable=False, index=True),
        Column("principal_id", Text, nullable=False),
        Column("column_name", Text),
        Column("metrics", JSONB, nullable=False),
        schema=schema,
    )


def create_analysis_results(connection: Connection, workspace_id: UUID) -> None:
    """Provision saved results for an existing workspace using its operator connection."""
    schema = f"w_{workspace_id.hex}"
    _analysis_results(schema).create(connection, checkfirst=True)
    connection.exec_driver_sql(f"GRANT SELECT ON {schema}.analysis_results TO {schema}_reader")
    connection.exec_driver_sql(f"GRANT SELECT, INSERT ON {schema}.analysis_results TO {schema}_writer")


def _analysis_runs(schema: str) -> Table:
    return Table(
        "analysis_runs", MetaData(),
        Column("run_id", Uuid, primary_key=True),
        Column("version_id", Uuid, ForeignKey(version_catalog(schema).c.version_id, ondelete="CASCADE"),
               nullable=False, index=True),
        Column("principal_id", Text, nullable=False),
        Column("membership_revision", Uuid, nullable=False),
        Column("conversation_id", Uuid, nullable=False),
        Column("policy_version", Integer, nullable=False),
        Column("column_name", Text, nullable=False),
        Column("expires_at", DateTime(timezone=True), nullable=False, index=True),
        Column("attempt_ids", ARRAY(Uuid), nullable=False),
        Column("lease_until", DateTime(timezone=True)),
        Column("results", JSONB, nullable=False),
        Column("last_attempt_failed", Boolean, nullable=False),
        CheckConstraint(f"cardinality(attempt_ids) <= {ANALYSIS_RUN_CALL_LIMIT}"),
        CheckConstraint("jsonb_array_length(results) <= 2"),
        schema=schema,
    )


def create_analysis_runs(connection: Connection, workspace_id: UUID) -> None:
    """Provision the opt-in checkpoint table with the workspace operator connection."""
    schema = f"w_{workspace_id.hex}"
    _analysis_runs(schema).create(connection, checkfirst=True)
    connection.exec_driver_sql(f"GRANT SELECT, INSERT ON {schema}.analysis_runs TO {schema}_writer")
    connection.exec_driver_sql(
        f"GRANT UPDATE (attempt_ids, lease_until, results, last_attempt_failed) "
        f"ON {schema}.analysis_runs TO {schema}_writer"
    )


def purge_expired_analysis_runs(connection: Connection, workspace_id: UUID) -> int:
    """Remove expired checkpoints using the workspace operator connection."""
    runs = _analysis_runs(f"w_{workspace_id.hex}")
    return connection.execute(delete(runs).where(runs.c.expires_at <= func.statement_timestamp())).rowcount


def create_analysis_run_purge(connection: Connection, workspace_id: UUID) -> None:
    """Grant a provisioned workspace purge role access to the scheduled operation."""
    schema = f"w_{workspace_id.hex}"
    operation = f"{schema}.purge_expired_analysis_runs()"
    connection.exec_driver_sql(f"""
        CREATE OR REPLACE FUNCTION {operation}
        RETURNS bigint
        LANGUAGE SQL
        SECURITY DEFINER
        SET search_path = pg_catalog, pg_temp
        BEGIN ATOMIC
            WITH expired AS (
                DELETE FROM {schema}.analysis_runs
                WHERE expires_at <= pg_catalog.statement_timestamp()
                RETURNING 1
            )
            SELECT pg_catalog.count(*) FROM expired;
        END
    """)
    connection.exec_driver_sql(f"REVOKE ALL ON FUNCTION {operation} FROM PUBLIC, {schema}_reader, {schema}_writer")
    connection.exec_driver_sql(f"GRANT USAGE ON SCHEMA {schema} TO {schema}_purger")
    connection.exec_driver_sql(f"GRANT EXECUTE ON FUNCTION {operation} TO {schema}_purger")


class WorkspaceDatasets:
    def __init__(
        self,
        access: WorkspaceAccess,
        resolve_access: Callable[[], WorkspaceAccess | None],
        reader: Engine,
        writer: Engine,
    ) -> None:
        self._access = access
        self._resolve_access = resolve_access
        self._reader = reader
        self._writer = writer
        self._schema = f"w_{access.workspace_id.hex}"
        self._catalog = version_catalog(self._schema)
        self._receipts = _upload_receipts(self._schema)
        self._results = _analysis_results(self._schema)
        self._runs = _analysis_runs(self._schema)

    def _check_access(self, *, write: bool) -> None:
        if self._resolve_access() != self._access or (write and not self._access.can_manage):
            raise PermissionError("This workspace operation is unavailable.")

    @contextmanager
    def _transaction(self, *, write: bool = False) -> Iterator[Connection]:
        self._check_access(write=write)
        engine = self._writer if write else self._reader
        role = f"{self._schema}_{'writer' if write else 'reader'}"
        with engine.begin() as connection:
            if write:
                connection.exec_driver_sql("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
            else:
                connection.exec_driver_sql("SET TRANSACTION READ ONLY")
            connection.exec_driver_sql("SET LOCAL statement_timeout = '3s'")
            connection.exec_driver_sql("SET LOCAL search_path = pg_catalog")
            restricted = connection.execute(text("""
                SELECT session_user = :role AND current_user = :role AND rolcanlogin
                       AND NOT (rolsuper OR rolbypassrls OR rolcreaterole OR rolcreatedb OR rolreplication)
                       AND NOT EXISTS (SELECT 1 FROM pg_catalog.pg_auth_members WHERE member = r.oid)
                FROM pg_catalog.pg_roles AS r WHERE rolname = current_user
            """), {"role": role}).scalar_one()
            if not restricted:
                raise PermissionError("This workspace operation is unavailable.")
            yield connection

    def upload(self, frame: pd.DataFrame, filename: str) -> WorkspaceDataset:
        return self._publish_version(lambda connection: self._store_upload(connection, frame, filename), uuid4(), None)

    def upload_csv(self, contents: bytes, filename: str, request_id: UUID) -> WorkspaceDataset:
        if not contents or len(contents) > MAX_CSV_BYTES or "\x00" in filename:
            raise ValueError("Choose a valid CSV file up to 1 MB.")
        digest = sha256(filename.encode("utf-8") + b"\x00" + contents).digest()
        return self._publish_version(
            lambda connection: self._store_upload(connection, _load_csv(contents), filename),
            request_id, digest,
        )

    def _publish_version(
        self, publish: Callable[[Connection], WorkspaceDataset],
        request_id: UUID, fingerprint: bytes | None,
    ) -> WorkspaceDataset:
        receipts = self._receipts.c
        with self._transaction(write=True) as connection:
            connection.execute(text(
                "SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"
            ), {"key": f"workspace_upload:{self._schema}"})
            self._check_access(write=True)
            prior = connection.execute(
                select(receipts.fingerprint, receipts.version_id).where(
                    receipts.principal_id == self._access.principal_id, receipts.request_id == request_id,
                )
            ).one_or_none()
            if prior is not None:
                if prior.fingerprint != fingerprint:
                    raise UploadRequestConflict("This upload request was used for a different file.")
                try:
                    return self._metadata(connection, prior.version_id)
                except PermissionError:
                    raise UploadRequestConflict("The dataset from this upload was deleted.") from None

            day = connection.scalar(select(func.timezone("UTC", func.statement_timestamp()).cast(Date)))
            used = connection.scalar(
                select(func.count()).select_from(self._receipts).where(receipts.admitted_on == day)
            )
            if used >= WORKSPACE_UPLOADS_PER_DAY:
                raise UploadLimitReached("This workspace has reached its daily upload limit.")

            dataset = publish(connection)
            connection.execute(insert(self._receipts).values(
                principal_id=self._access.principal_id, request_id=request_id, fingerprint=fingerprint,
                version_id=dataset.version_id, admitted_on=day,
            ))
        return dataset

    def _store_upload(self, connection: Connection, frame: pd.DataFrame, filename: str) -> WorkspaceDataset:
        columns = tuple(frame.columns)
        if len(frame) > MAX_UPLOAD_ROWS or len(columns) > MAX_UPLOAD_COLUMNS:
            raise ValueError("CSV exceeds the workspace row or column limit.")
        if not columns or any(
            not isinstance(name, str) or not name.strip() or "\x00" in name
            or len(name.encode("utf-8")) > 63 or name == ROW_KEY
            for name in columns
        ) or len(set(columns)) != len(columns):
            raise ValueError("Upload columns must have distinct, valid names.")
        dataset = WorkspaceDataset(uuid4(), uuid4(), filename, len(frame), columns)
        table_name = f"v_{dataset.version_id.hex}"
        frame.reset_index(drop=True).to_sql(
            table_name, connection, schema=self._schema, if_exists="fail", index_label=ROW_KEY,
        )
        self._register_version(connection, dataset)
        return dataset

    def _register_version(self, connection: Connection, dataset: WorkspaceDataset) -> None:
        table_name = f"v_{dataset.version_id.hex}"
        connection.exec_driver_sql(f"GRANT SELECT ON {self._schema}.{table_name} TO {self._schema}_reader")
        connection.execute(insert(self._catalog).values(
            version_id=dataset.version_id, dataset_id=dataset.dataset_id,
            filename=dataset.filename, row_count=dataset.row_count, columns=list(dataset.columns),
        ))

    def _metadata(self, connection: Connection, version_id: UUID) -> WorkspaceDataset:
        row = connection.execute(
            select(self._catalog).where(self._catalog.c.version_id == version_id)
        ).one_or_none()
        if row is None:
            raise PermissionError("This dataset version is unavailable.")
        return self._dataset(row)

    @staticmethod
    def _dataset(row) -> WorkspaceDataset:
        return WorkspaceDataset(row.dataset_id, row.version_id, row.filename, row.row_count, tuple(row.columns))

    def list_datasets(self, query: str = "") -> tuple[WorkspaceDataset, ...]:
        """Search filenames and column names in the current workspace's live catalog."""
        if len(query) > CATALOG_SEARCH_MAX_CHARS or "\x00" in query:
            raise ValueError("Invalid catalog search.")
        query = query.strip()
        catalog = self._catalog
        statement = select(catalog).order_by(catalog.c.filename, catalog.c.version_id)
        if query:
            column_name = func.unnest(catalog.c.columns, type_=Text()).column_valued("column_name")
            matches_column = select(1).where(column_name.icontains(query, autoescape=True)).exists()
            statement = statement.where(or_(
                catalog.c.filename.icontains(query, autoescape=True), matches_column,
            )).limit(CATALOG_SEARCH_LIMIT)
        with self._transaction() as connection:
            return tuple(self._dataset(row) for row in connection.execute(statement))

    def metadata(self, version_id: UUID) -> WorkspaceDataset:
        with self._transaction() as connection:
            return self._metadata(connection, version_id)

    def summarize(self, request: WorkspaceRowCount) -> WorkspaceAnalysis:
        """Compute one aggregate result over a bounded, authorized version."""
        with self._read_version(request.version_id) as (connection, dataset, relation):
            aggregates = [func.count().label("row_count")]
            if isinstance(request, WorkspaceColumnSummary):
                if request.column not in dataset.columns:
                    raise PermissionError("This dataset column is unavailable.")
                numeric = connection.execute(text("""
                    SELECT atttypid IN (
                        'int2'::regtype, 'int4'::regtype, 'int8'::regtype,
                        'float4'::regtype, 'float8'::regtype, 'numeric'::regtype
                    )
                    FROM pg_catalog.pg_attribute
                    WHERE attrelid = pg_catalog.to_regclass(:relation)
                      AND attname = :column AND attnum > 0 AND NOT attisdropped
                """), {"relation": f"{relation.schema}.{relation.name}", "column": request.column}).scalar_one_or_none()
                if numeric is None:
                    raise PermissionError("This dataset column is unavailable.")
                source = select(relation.c[request.column].label("value")).limit(MAX_UPLOAD_ROWS + 1).subquery()
                value = source.c.value
                aggregates.extend([
                    func.count(value).label("non_null_count"),
                    (func.count() - func.count(value)).label("null_count"),
                ])
                if numeric:
                    aggregates.extend([
                        func.min(value).label("minimum"), func.max(value).label("maximum"),
                        func.sum(value).label("total"), func.avg(value).label("mean"),
                    ])
            else:
                source = select(literal(1)).select_from(relation).limit(MAX_UPLOAD_ROWS + 1).subquery()
            result = connection.execute(select(*aggregates).select_from(source)).mappings().one()
            if result["row_count"] > MAX_UPLOAD_ROWS:
                raise ValueError("This dataset exceeds the analysis row limit.")
            return WorkspaceAnalysis(result_id=uuid4(), access=self._access, request=request, values=dict(result))

    def analysis_csv(self, result: WorkspaceAnalysis) -> bytes:
        """Reauthorize an executed session result and serialize its original values."""
        if result.access != self._access:
            raise PermissionError("This analysis result is unavailable.")
        dataset = self.metadata(result.request.version_id)
        column = None
        operation = "count_rows"
        if isinstance(result.request, WorkspaceColumnSummary):
            column = result.request.column
            operation = "summarize_column"
        return csv_bytes(
            ("result_id", "version_id", "operation", "column", "metric", "value"),
            (
                (result.result_id, dataset.version_id, operation, column, metric, value)
                for metric, value in result.values.items()
            ),
        )

    def save_analysis(self, result: WorkspaceAnalysis) -> None:
        """Save the exact session result once, under its originating principal."""
        if result.access != self._access:
            raise PermissionError("This analysis result is unavailable.")
        column = result.request.column if isinstance(result.request, WorkspaceColumnSummary) else None
        metrics = _encode_metrics(result)
        version_id = result.request.version_id
        with self._transaction(write=True) as connection:
            self._metadata(connection, version_id)
            saved = connection.execute(
                pg_insert(self._results).values(
                    result_id=result.result_id, version_id=version_id, principal_id=self._access.principal_id,
                    column_name=column, metrics=metrics,
                ).on_conflict_do_nothing(index_elements=[self._results.c.result_id])
            )
            if saved.rowcount == 0:
                prior = self._saved_analysis(connection, version_id, result.result_id)
                if prior.column_name != column or prior.metrics != metrics:
                    raise PermissionError("This analysis result is unavailable.")

    def _saved_analysis(self, connection: Connection, version_id: UUID, result_id: UUID):
        results = self._results.c
        row = connection.execute(select(self._results).where(
            results.result_id == result_id,
            results.version_id == version_id,
            results.principal_id == self._access.principal_id,
        )).one_or_none()
        if row is None:
            raise PermissionError("This saved analysis result is unavailable.")
        return row

    def restore_analysis(self, version_id: UUID, result_id: UUID) -> WorkspaceAnalysis:
        """Restore the principal's saved result into their current authorized session."""
        with self._transaction() as connection:
            row = self._saved_analysis(connection, version_id, result_id)
        request = WorkspaceRowCount(version_id=version_id)
        if row.column_name is not None:
            request = WorkspaceColumnSummary(version_id=version_id, column=row.column_name)
        return WorkspaceAnalysis(result_id, self._access, request, _decode_metrics(row.metrics))

    def analysis_runs_available(self) -> bool:
        """Expose the opt-in controls only in provisioned manager workspaces."""
        if not self._access.can_manage or self._access.membership_revision is None:
            return False
        with self._transaction() as connection:
            return inspect(connection).has_table(self._runs.name, schema=self._schema)

    def start_analysis_run(self, request: WorkspaceColumnSummary, *, conversation_id: UUID) -> WorkspaceAnalysisRun:
        """Start a fresh count and column summary for one authorized dataset version."""
        with self._transaction(write=True) as connection:
            if self._access.membership_revision is None:
                raise PermissionError("This analysis run is unavailable.")
            dataset = self._metadata(connection, request.version_id)
            if request.column not in dataset.columns:
                raise PermissionError("This dataset column is unavailable.")
            now = connection.scalar(select(func.clock_timestamp()))
            row = connection.execute(insert(self._runs).values(
                run_id=uuid4(), version_id=request.version_id, column_name=request.column,
                principal_id=self._access.principal_id, membership_revision=self._access.membership_revision,
                conversation_id=conversation_id, policy_version=_ANALYSIS_RUN_POLICY,
                expires_at=now + _ANALYSIS_RUN_RETENTION, attempt_ids=[], results=[], last_attempt_failed=False,
            ).returning(self._runs)).one()
            return self._analysis_run_snapshot(row, now)

    def resume_analysis_run(
        self, run_id: UUID, *, version_id: UUID, conversation_id: UUID,
    ) -> WorkspaceAnalysisRun:
        """Restore collected results into the currently authorized manager session."""
        with self._transaction(write=True) as connection:
            row, now = self._load_analysis_run(connection, run_id, version_id, conversation_id)
            return self._analysis_run_snapshot(row, now)

    def advance_analysis_run(
        self, run_id: UUID, *, version_id: UUID, conversation_id: UUID, attempt_id: UUID,
    ) -> WorkspaceAnalysisRun:
        """Reserve and execute up to one task, charging a repeated attempt ID once."""
        with self._transaction(write=True) as connection:
            row, now = self._load_analysis_run(connection, run_id, version_id, conversation_id, lock=True)
            current = self._analysis_run_snapshot(row, now)
            if attempt_id in row.attempt_ids or current.complete or current.in_progress or not current.calls_remaining:
                return current
            connection.execute(update(self._runs).where(self._runs.c.run_id == run_id).values(
                attempt_ids=[*row.attempt_ids, attempt_id],
                lease_until=min(now + _ANALYSIS_ATTEMPT_LEASE, row.expires_at),
                last_attempt_failed=False,
            ))
            request = self._analysis_run_requests(row)[len(row.results)]

        try:
            result = self.summarize(request)
        except (SQLAlchemyError, ValueError):
            result = None

        with self._transaction(write=True) as connection:
            row, now = self._load_analysis_run(connection, run_id, version_id, conversation_id, lock=True)
            if row.attempt_ids[-1] != attempt_id or row.lease_until is None or row.lease_until <= now:
                return self._analysis_run_snapshot(row, now)
            results = row.results
            if result is not None:
                results = [*results, [str(result.result_id), _encode_metrics(result)]]
            updated = connection.execute(update(self._runs).where(self._runs.c.run_id == run_id).values(
                lease_until=None, results=results, last_attempt_failed=result is None,
            ).returning(self._runs)).one()
            return self._analysis_run_snapshot(updated, now)

    def _load_analysis_run(
        self, connection: Connection, run_id: UUID, version_id: UUID, conversation_id: UUID, *, lock: bool = False,
    ):
        if self._access.membership_revision is None:
            raise PermissionError("This analysis run is unavailable.")
        runs = self._runs.c
        statement = select(self._runs).where(
            runs.run_id == run_id, runs.version_id == version_id, runs.conversation_id == conversation_id,
            runs.principal_id == self._access.principal_id, runs.membership_revision == self._access.membership_revision,
            runs.policy_version == _ANALYSIS_RUN_POLICY,
        )
        if lock:
            statement = statement.with_for_update()
        row = connection.execute(statement).one_or_none()
        self._check_access(write=True)
        now = connection.scalar(select(func.clock_timestamp()))
        if row is None or row.expires_at <= now:
            raise PermissionError("This analysis run is unavailable.")
        return row, now

    @staticmethod
    def _analysis_run_requests(row) -> tuple[WorkspaceRowCount, WorkspaceColumnSummary]:
        return (
            WorkspaceRowCount(version_id=row.version_id),
            WorkspaceColumnSummary(version_id=row.version_id, column=row.column_name),
        )

    def _analysis_run_snapshot(self, row, now: datetime) -> WorkspaceAnalysisRun:
        requests = self._analysis_run_requests(row)
        results = tuple(
            WorkspaceAnalysis(UUID(result_id), self._access, request, _decode_metrics(metrics))
            for (result_id, metrics), request in zip(row.results, requests)
        )
        active = row.lease_until is not None and row.lease_until > now
        interrupted = row.lease_until is not None and not active
        return WorkspaceAnalysisRun(
            row.run_id, requests[1], row.expires_at, results, len(row.attempt_ids), active,
            row.last_attempt_failed or interrupted,
        )

    def preview_cleaning(self, plan: WorkspaceTrimWhitespace) -> WorkspaceCleaningPreview:
        """Evaluate one typed cleaning operation against the authorized source."""
        with self._read_version(plan.version_id) as (connection, dataset, relation):
            if plan.column not in dataset.columns:
                raise PermissionError("This dataset column is unavailable.")
            statement = select(relation.c[plan.column]).order_by(relation.c[ROW_KEY]).limit(MAX_UPLOAD_ROWS + 1)
            values = connection.scalars(statement).all()
        preview, _ = self._evaluate_trim(plan, values)
        return preview

    def _evaluate_trim(
        self, plan: WorkspaceTrimWhitespace, values: list,
    ) -> tuple[WorkspaceCleaningPreview, list]:
        if len(values) > MAX_UPLOAD_ROWS:
            raise ValueError("This dataset exceeds the cleaning preview row limit.")

        changed_count = 0
        preview = []
        trimmed = []
        fingerprint = sha256(plan.model_dump_json().encode("utf-8"))
        for index, before in enumerate(values):
            after = before.strip() if isinstance(before, str) else before
            trimmed.append(after)
            changed_count += isinstance(before, str) and before != after
            fingerprint.update(json.dumps((type(before).__name__, repr(before), repr(after))).encode("ascii"))
            if index < PREVIEW_ROWS:
                preview.append((before, after))
        result = WorkspaceCleaningPreview(
            plan, self._access, fingerprint.hexdigest(), len(values), changed_count, tuple(preview),
        )
        return result, trimmed

    def replay_cleaning(self, approved: WorkspaceCleaningPreview) -> WorkspaceCleaningPreview:
        """Verify the manager-reviewed operation in the same authorized session."""
        self._check_cleaning_access(approved)
        replayed = self.preview_cleaning(approved.plan)
        self._check_cleaning_fingerprint(approved, replayed)
        return replayed

    def _check_cleaning_access(self, approved: WorkspaceCleaningPreview) -> None:
        self._check_access(write=True)
        if approved.access != self._access:
            raise PermissionError("This cleaning preview is unavailable.")

    @staticmethod
    def _check_cleaning_fingerprint(approved: WorkspaceCleaningPreview, current: WorkspaceCleaningPreview) -> None:
        if current.fingerprint != approved.fingerprint:
            raise CleaningPreviewChanged("The column values changed. Generate a new cleaning preview.")

    def publish_cleaning(self, approved: WorkspaceCleaningPreview, request_id: UUID) -> WorkspaceDataset:
        """Publish the reviewed trim as a sibling version using the existing upload receipt."""
        self._check_cleaning_access(approved)
        fingerprint = sha256(f"cleaning:{approved.fingerprint}".encode("ascii")).digest()
        return self._publish_version(
            lambda connection: self._store_cleaned_version(connection, approved), request_id, fingerprint,
        )

    def _store_cleaned_version(self, connection: Connection, approved: WorkspaceCleaningPreview) -> WorkspaceDataset:
        plan = approved.plan
        with self._version_relation(connection, plan.version_id) as (source, relation):
            if plan.column not in source.columns:
                raise PermissionError("This dataset column is unavailable.")
            statement = select(relation).order_by(relation.c[ROW_KEY]).limit(MAX_UPLOAD_ROWS + 1)
            rows = [dict(row) for row in connection.execute(statement).mappings()]
            replayed, trimmed = self._evaluate_trim(plan, [row[plan.column] for row in rows])
            self._check_cleaning_fingerprint(approved, replayed)
            for row, value in zip(rows, trimmed, strict=True):
                row[plan.column] = value

            self._check_access(write=True)
            dataset = WorkspaceDataset(source.dataset_id, uuid4(), source.filename, len(rows), source.columns)
            table_name = f"v_{dataset.version_id.hex}"
            connection.exec_driver_sql(
                f"CREATE TABLE {self._schema}.{table_name} (LIKE {self._schema}.{relation.name} INCLUDING INDEXES)"
            )
            if rows:
                target = Table(table_name, MetaData(), *(Column(name) for name in relation.c.keys()), schema=self._schema)
                connection.execute(insert(target), rows)
            self._register_version(connection, dataset)
            return dataset

    def preview(self, version_id: UUID, limit: int = PREVIEW_ROWS) -> list[dict]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Preview limit must be between 1 and 100.")
        _, rows = self._read_rows(version_id, limit)
        return rows

    def check_preview_access(self, version_id: UUID) -> None:
        """Check the preview projection's permissions without fetching any rows."""
        self._read_rows(version_id, limit=0)

    def preview_csv(self, version_id: UUID) -> WorkspaceCSV:
        """Resolve a fresh authorized preview and serialize it without a server file."""
        dataset, rows = self._read_rows(version_id, PREVIEW_ROWS + 1)
        return self._serialize_csv(dataset, rows[:PREVIEW_ROWS], truncated=len(rows) > PREVIEW_ROWS)

    def full_csv(self, version_id: UUID) -> WorkspaceCSV:
        """Export every row of a fresh authorized version within the pilot row limit."""
        dataset, rows = self._read_rows(version_id, MAX_UPLOAD_ROWS + 1)
        if len(rows) > MAX_UPLOAD_ROWS:
            raise ValueError("This dataset exceeds the full CSV row limit.")
        return self._serialize_csv(dataset, rows, truncated=False)

    @staticmethod
    def _serialize_csv(dataset: WorkspaceDataset, rows: list[dict], *, truncated: bool) -> WorkspaceCSV:
        return WorkspaceCSV(
            version_id=dataset.version_id,
            data=csv_bytes(dataset.columns, ([row[name] for name in dataset.columns] for row in rows)),
            row_count=len(rows),
            truncated=truncated,
        )

    def _read_rows(self, version_id: UUID, limit: int) -> tuple[WorkspaceDataset, list[dict]]:
        with self._read_version(version_id) as (connection, dataset, relation):
            statement = select(*(relation.c[name] for name in dataset.columns)).order_by(relation.c[ROW_KEY]).limit(limit)
            return dataset, [dict(row) for row in connection.execute(statement).mappings()]

    @contextmanager
    def _read_version(self, version_id: UUID) -> Iterator[tuple[Connection, WorkspaceDataset, Table]]:
        with self._transaction() as connection:
            with self._version_relation(connection, version_id) as (dataset, relation):
                yield connection, dataset, relation

    @contextmanager
    def _version_relation(self, connection: Connection, version_id: UUID) -> Iterator[tuple[WorkspaceDataset, Table]]:
        dataset = self._metadata(connection, version_id)
        relation = Table(
            f"v_{dataset.version_id.hex}", MetaData(),
            *(Column(name) for name in (*dataset.columns, ROW_KEY)), schema=self._schema,
        )
        try:
            yield dataset, relation
        except NoSuchTableError:
            raise PermissionError("This dataset version is unavailable.") from None
        except ProgrammingError as exc:
            if getattr(exc.orig, "pgcode", None) != "42P01":
                raise
            # Deletion may commit after the catalog lookup and before the read.
            raise PermissionError("This dataset version is unavailable.") from None

    def delete(self, version_id: UUID) -> None:
        with self._transaction(write=True) as connection:
            deleted = connection.execute(
                delete(self._catalog).where(self._catalog.c.version_id == version_id)
                .returning(self._catalog.c.version_id)
            ).scalar_one_or_none()
            if deleted is None:
                raise PermissionError("This dataset version is unavailable.")
            Table(f"v_{deleted.hex}", MetaData(), schema=self._schema).drop(connection)
