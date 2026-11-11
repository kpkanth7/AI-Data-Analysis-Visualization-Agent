"""Private dataset lifecycle for a server-authorized workspace."""

import csv
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from io import BytesIO, StringIO
from typing import Annotated
from uuid import UUID, uuid4

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import (
    Column, Date, Float, Integer, LargeBinary, MetaData, Numeric, Table, Text,
    delete, func, insert, inspect, literal, or_, select, text,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import NoSuchTableError, ProgrammingError
from sqlalchemy.types import Uuid


ROW_KEY = "_workspace_row_id"
WORKSPACE_UPLOADS_PER_DAY = 5
MAX_CSV_BYTES = 1024 * 1024
MAX_UPLOAD_ROWS = 1_000
MAX_UPLOAD_COLUMNS = 64
PREVIEW_ROWS = 50
CATALOG_SEARCH_MAX_CHARS = 128
CATALOG_SEARCH_LIMIT = 50


class UploadLimitReached(ValueError):
    pass


class UploadRequestConflict(ValueError):
    pass


def _csv_cell(value):
    if isinstance(value, str) and (
        value.startswith(("\t", "\r", "\n")) or value.lstrip().startswith(("=", "+", "-", "@"))
    ):
        return "'" + value
    return value


@dataclass(frozen=True)
class WorkspaceAccess:
    """Verified by the server. Rotate session_id when authorization changes."""

    principal_id: str
    session_id: UUID
    workspace_id: UUID
    can_manage: bool


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
        return self._publish_upload(lambda: frame, filename, uuid4(), None)

    def upload_csv(self, contents: bytes, filename: str, request_id: UUID) -> WorkspaceDataset:
        if not contents or len(contents) > MAX_CSV_BYTES or "\x00" in filename:
            raise ValueError("Choose a valid CSV file up to 1 MB.")
        digest = sha256(filename.encode("utf-8") + b"\x00" + contents).digest()
        return self._publish_upload(
            lambda: pd.read_csv(BytesIO(contents), nrows=MAX_UPLOAD_ROWS + 1, compression=None),
            filename, request_id, digest,
        )

    def _publish_upload(
        self, load_frame: Callable[[], pd.DataFrame], filename: str,
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

            dataset = self._store_upload(connection, load_frame(), filename)
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
        connection.exec_driver_sql(f"GRANT SELECT ON {self._schema}.{table_name} TO {self._schema}_reader")
        connection.execute(insert(self._catalog).values(
            version_id=dataset.version_id, dataset_id=dataset.dataset_id,
            filename=dataset.filename, row_count=dataset.row_count, columns=list(columns),
        ))
        return dataset

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

    def summarize(self, request: WorkspaceRowCount) -> dict[str, int | float | Decimal | None]:
        """Compute one aggregate result over a bounded, authorized version."""
        with self._read_version(request.version_id) as (connection, dataset, relation):
            aggregates = [func.count().label("row_count")]
            if isinstance(request, WorkspaceColumnSummary):
                if request.column not in dataset.columns:
                    raise PermissionError("This dataset column is unavailable.")
                columns = inspect(connection).get_columns(relation.name, schema=relation.schema)
                column_type = next((column["type"] for column in columns if column["name"] == request.column), None)
                if column_type is None:
                    raise PermissionError("This dataset column is unavailable.")
                source = select(relation.c[request.column].label("value")).limit(MAX_UPLOAD_ROWS + 1).subquery()
                value = source.c.value
                aggregates.extend([
                    func.count(value).label("non_null_count"),
                    (func.count() - func.count(value)).label("null_count"),
                ])
                if isinstance(column_type, (Float, Integer, Numeric)):
                    aggregates.extend([
                        func.min(value).label("minimum"), func.max(value).label("maximum"),
                        func.sum(value).label("total"), func.avg(value).label("mean"),
                    ])
            else:
                source = select(literal(1)).select_from(relation).limit(MAX_UPLOAD_ROWS + 1).subquery()
            result = connection.execute(select(*aggregates).select_from(source)).mappings().one()
            if result["row_count"] > MAX_UPLOAD_ROWS:
                raise ValueError("This dataset exceeds the analysis row limit.")
            return dict(result)

    def preview(self, version_id: UUID, limit: int = PREVIEW_ROWS) -> list[dict]:
        _, rows = self._read_preview(version_id, limit)
        return rows

    def preview_csv(self, version_id: UUID) -> bytes:
        """Resolve a fresh authorized preview and serialize it without a server file."""
        dataset, rows = self._read_preview(version_id, PREVIEW_ROWS)
        output = StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow([_csv_cell(name) for name in dataset.columns])
        writer.writerows([_csv_cell(row[name]) for name in dataset.columns] for row in rows)
        return output.getvalue().encode("utf-8-sig")

    def _read_preview(self, version_id: UUID, limit: int) -> tuple[WorkspaceDataset, list[dict]]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Preview limit must be between 1 and 100.")
        with self._read_version(version_id) as (connection, dataset, relation):
            statement = select(*(relation.c[name] for name in dataset.columns)).order_by(relation.c[ROW_KEY]).limit(limit)
            return dataset, [dict(row) for row in connection.execute(statement).mappings()]

    @contextmanager
    def _read_version(self, version_id: UUID) -> Iterator[tuple[Connection, WorkspaceDataset, Table]]:
        with self._transaction() as connection:
            dataset = self._metadata(connection, version_id)
            relation = Table(
                f"v_{dataset.version_id.hex}", MetaData(),
                *(Column(name) for name in (*dataset.columns, ROW_KEY)), schema=self._schema,
            )
            try:
                yield connection, dataset, relation
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
