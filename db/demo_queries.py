"""Typed reads of approved demo versions through a dedicated PostgreSQL login."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from operator import eq, ge, gt, le, lt, ne
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat
from sqlalchemy import Table, func, select
from sqlalchemy.engine import Connection, Engine


_COMPARISONS = {"eq": eq, "ne": ne, "lt": lt, "le": le, "gt": gt, "ge": ge}


@contextmanager
def demo_read_transaction(engine: Engine) -> Iterator[Connection]:
    with engine.connect() as connection:
        connection.exec_driver_sql("SET TRANSACTION READ ONLY")
        connection.exec_driver_sql("SET LOCAL statement_timeout = '3s'")
        connection.exec_driver_sql("SET LOCAL search_path = pg_catalog")
        yield connection


@dataclass(frozen=True)
class DemoAccess:
    principal_id: str
    workspace_id: str

    def __post_init__(self) -> None:
        if not self.principal_id.strip() or not self.workspace_id.strip():
            raise ValueError("Demo reads require a principal and workspace.")


@dataclass(frozen=True)
class DemoVersion:
    version_id: UUID
    workspace_id: str
    label: str
    relation: Table
    columns: tuple[str, ...]
    row_key: str

    def __post_init__(self) -> None:
        if self.relation.schema != "demo" or not self.columns:
            raise ValueError("A demo version needs a schema-qualified demo relation and approved columns.")
        if tuple(self.relation.primary_key.columns.keys()) != (self.row_key,):
            raise ValueError("A demo version needs a primary key for stable previews.")
        if any(name not in self.relation.c for name in self.columns):
            raise ValueError("Demo version columns must belong to its relation.")


class _RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class DemoFilter(_RequestModel):
    column: str
    comparison: Literal["eq", "ne", "lt", "le", "gt", "ge"] = "eq"
    value: Annotated[str, Field(max_length=2048)] | int | FiniteFloat | bool


class DemoSort(_RequestModel):
    column: str
    descending: bool = False


class DemoCount(_RequestModel):
    version_id: UUID
    filters: Annotated[tuple[DemoFilter, ...], Field(max_length=10)] = ()


class DemoPreview(DemoCount):
    columns: Annotated[tuple[str, ...], Field(max_length=64)] = ()
    sort: Annotated[tuple[DemoSort, ...], Field(max_length=3)] = ()
    limit: Annotated[int, Field(ge=1, le=100)] = 50
    offset: Annotated[int, Field(ge=0, le=100_000)] = 0


class DemoQueries:
    def __init__(self, engine: Engine, access: DemoAccess, versions: tuple[DemoVersion, ...]):
        self._engine = engine
        self._versions = {
            version.version_id: version for version in versions
            if version.workspace_id == access.workspace_id
        }

    @property
    def versions(self) -> tuple[DemoVersion, ...]:
        return tuple(self._versions.values())

    def execute(self, request: DemoCount) -> list[dict]:
        version = self._versions.get(request.version_id)
        if version is None:
            raise PermissionError("This dataset version is unavailable in the demo workspace.")

        relation = version.relation
        approved = {name: relation.c[name] for name in version.columns}

        def column(name: str):
            if name not in approved:
                raise PermissionError("The requested column is unavailable in this dataset version.")
            return approved[name]

        if isinstance(request, DemoPreview):
            order = [
                column(item.column).desc() if item.descending else column(item.column).asc()
                for item in request.sort
            ]
            statement = (
                select(*(column(name) for name in dict.fromkeys(request.columns or version.columns)))
                .order_by(*order, relation.c[version.row_key].asc())
                .limit(request.limit)
                .offset(request.offset)
            )
        else:
            statement = select(func.count().label("row_count")).select_from(relation)

        predicates = [
            _COMPARISONS[item.comparison](column(item.column), item.value)
            for item in request.filters
        ]
        with demo_read_transaction(self._engine) as connection:
            result = connection.execute(statement.where(*predicates))
            return [dict(row) for row in result.mappings()]
