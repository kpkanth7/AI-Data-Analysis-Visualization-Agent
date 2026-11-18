"""Typed reads of approved demo versions through a dedicated PostgreSQL login."""

import time
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from operator import eq, ge, gt, le, lt, ne
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator, model_validator
from sqlalchemy import ColumnElement, Select, Table, Text, cast, func, literal_column, select
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import DBAPIError, SQLAlchemyError


_COMPARISONS = {"eq": eq, "ne": ne, "lt": lt, "le": le, "gt": gt, "ge": ge}
_Comparison = Literal["eq", "ne", "lt", "le", "gt", "ge"]
_PLAN_WORKERS = 2
DEMO_PREVIEW_MAX_BYTES = 64 * 1024


class DemoPreviewTooLarge(ValueError):
    """The requested preview exceeds its UTF-8 JSON payload budget."""


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
    comparison: _Comparison = "eq"
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


_TaskId = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")]


class DemoEvidenceFilter(_RequestModel):
    column: str
    comparison: _Comparison = "eq"
    source_task_id: _TaskId
    source_column: str
    row_index: Annotated[int, Field(ge=0, le=99)] = 0


class _DemoTask(_RequestModel):
    task_id: _TaskId
    depends_on: Annotated[tuple[_TaskId, ...], Field(max_length=3)] = ()
    evidence_filters: Annotated[tuple[DemoEvidenceFilter, ...], Field(max_length=10)] = ()


class DemoCountTask(_DemoTask):
    operation: Literal["count"]
    request: DemoCount

    @field_validator("request")
    @classmethod
    def require_count(cls, request: DemoCount) -> DemoCount:
        if isinstance(request, DemoPreview):
            raise ValueError("A count task requires a count request.")
        return request


class DemoPreviewTask(_DemoTask):
    operation: Literal["preview"]
    request: DemoPreview


DemoTask = Annotated[DemoCountTask | DemoPreviewTask, Field(discriminator="operation")]


class DemoPlan(_RequestModel):
    tasks: Annotated[tuple[DemoTask, ...], Field(min_length=1, max_length=4)]

    @model_validator(mode="after")
    def validate_dependencies(self) -> "DemoPlan":
        self.ordered_tasks()
        return self

    def ordered_tasks(self) -> tuple[DemoTask, ...]:
        tasks = {task.task_id: task for task in self.tasks}
        if len(tasks) != len(self.tasks):
            raise ValueError("Task IDs must be unique.")
        for task in self.tasks:
            dependencies = set(task.depends_on)
            if len(dependencies) != len(task.depends_on):
                raise ValueError("Dependencies must be unique within each task.")
            if dependencies - tasks.keys():
                raise ValueError("Every dependency must name a task in the plan.")
            if any(ref.source_task_id not in dependencies for ref in task.evidence_filters):
                raise ValueError("Evidence must come from a declared direct dependency.")
            if len(task.request.filters) + len(task.evidence_filters) > 10:
                raise ValueError("A task can have at most 10 filters in total.")
        graph = {task_id: task.depends_on for task_id, task in tasks.items()}
        try:
            order = tuple(TopologicalSorter(graph).static_order())
        except CycleError:
            raise ValueError("Task dependencies must be acyclic.") from None
        return tuple(tasks[task_id] for task_id in order)


@dataclass(frozen=True)
class DemoTaskResult:
    task_id: str
    request: DemoCount
    rows: list[dict]


_StopStatus = Literal["deadline", "cancelled"]


@dataclass(frozen=True)
class DemoTaskOutcome:
    task_id: str
    status: Literal["success", "failed", "skipped_dependency", "call_limit", "deadline", "cancelled"]
    result: DemoTaskResult | None = None
    reason: Literal["invalid_evidence", "query_failed", "result_limit"] | None = None


@dataclass(frozen=True)
class _PreparedTask:
    task: DemoTask
    statement: Select
    evidence_types: tuple[type, ...]


class _PlanDeadline(Exception):
    pass


def _evidence_type(column: ColumnElement) -> type:
    try:
        value_type = column.type.python_type
    except NotImplementedError:
        raise ValueError("Unsupported evidence column type.") from None
    if value_type not in (str, int, float, bool):
        raise ValueError("Unsupported evidence column type.")
    return value_type


def _resolve_evidence(ref: DemoEvidenceFilter, rows: list[dict], expected: type) -> DemoFilter:
    try:
        value = rows[ref.row_index][ref.source_column]
    except (IndexError, KeyError, TypeError):
        raise ValueError("The requested predecessor evidence is missing.") from None
    if type(value) is not expected:
        raise ValueError("The predecessor evidence has an invalid value.")
    try:
        resolved = DemoFilter(column=ref.column, comparison=ref.comparison, value=value)
        if isinstance(value, str):
            if "\x00" in value:
                raise ValueError
            value.encode("utf-8")
        return resolved
    except ValueError:
        raise ValueError("The predecessor evidence has an invalid value.") from None


class DemoQueries:
    def __init__(self, engine: Engine, access: DemoAccess, versions: tuple[DemoVersion, ...]):
        self._engine = engine
        self._access = access
        self._versions = {
            version.version_id: version for version in versions
            if version.workspace_id == access.workspace_id
        }

    @property
    def access(self) -> DemoAccess:
        return self._access

    @property
    def versions(self) -> tuple[DemoVersion, ...]:
        return tuple(self._versions.values())

    def execute(self, request: DemoCount) -> list[dict]:
        return self._execute_statement(self._statement(request))

    def execute_plan(self, plan: DemoPlan) -> tuple[DemoTaskResult, ...]:
        """Authorize every task, then execute the plan in dependency order."""
        results: dict[str, DemoTaskResult] = {}
        for prepared in self._prepare_plan(plan):
            request, statement = self._bind_task(prepared, results)
            task_id = prepared.task.task_id
            results[task_id] = DemoTaskResult(task_id, request, self._execute_statement(statement))
        return tuple(results.values())

    def run_plan(
        self, plan: DemoPlan, *, max_calls: int = 4, timeout_seconds: float = 12.0,
        cancelled: Callable[[], bool] | None = None,
    ) -> tuple[DemoTaskOutcome, ...]:
        """Run ready reads with two workers and a shared call budget and deadline.

        Cancellation keeps collected outcomes and drains admitted reads under their
        statement timeouts. The cancellation callback runs on the calling thread.
        """
        if type(max_calls) is not int or not 0 <= max_calls <= 4:
            raise ValueError("The call budget must be an integer from zero to four.")
        if type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= 12:
            raise ValueError("The time budget must be positive and at most twelve seconds.")
        deadline = time.monotonic() + timeout_seconds

        def stop_status() -> _StopStatus | None:
            if cancelled is not None and cancelled():
                return "cancelled"
            if time.monotonic() >= deadline:
                return "deadline"
            return None

        prepared_tasks = {item.task.task_id: item for item in self._prepare_plan(plan)}
        outcomes: dict[str, DemoTaskOutcome] = {}
        results: dict[str, DemoTaskResult] = {}
        running: dict[str, Future[DemoTaskOutcome]] = {}
        stop = None
        with ThreadPoolExecutor(max_workers=_PLAN_WORKERS) as workers:
            while True:
                stop = stop or stop_status()
                if stop:
                    break
                for task_id, prepared in prepared_tasks.items():
                    if len(running) == _PLAN_WORKERS:
                        break
                    if task_id in outcomes or task_id in running:
                        continue
                    dependencies = prepared.task.depends_on
                    if any(dependency not in outcomes for dependency in dependencies):
                        continue
                    if any(dependency not in results for dependency in dependencies):
                        outcomes[task_id] = DemoTaskOutcome(task_id, "skipped_dependency")
                        continue
                    if max_calls == 0:
                        outcomes[task_id] = DemoTaskOutcome(task_id, "call_limit")
                        continue
                    try:
                        request, statement = self._bind_task(prepared, results)
                    except ValueError:
                        outcomes[task_id] = DemoTaskOutcome(task_id, "failed", reason="invalid_evidence")
                        continue
                    stop = stop_status()
                    if stop:
                        break
                    max_calls -= 1
                    running[task_id] = workers.submit(self._run_task, task_id, request, statement, deadline)

                if stop or not running:
                    break
                remaining = max(0, deadline - time.monotonic())
                wait_time = min(remaining, 0.05) if cancelled is not None else remaining
                done, _ = wait(running.values(), timeout=wait_time, return_when=FIRST_COMPLETED)
                if cancelled is not None and cancelled():
                    stop = "cancelled"
                    break
                for task_id, future in tuple(running.items()):
                    if future not in done:
                        continue
                    outcome = running.pop(task_id).result()
                    outcomes[task_id] = outcome
                    if outcome.result is not None:
                        results[task_id] = outcome.result
                    if outcome.status == "deadline":
                        stop = "deadline"

        if stop == "deadline":
            # Workers check the deadline at completion, including results that
            # arrive just before it and have yet to be collected.
            outcomes.update((task_id, future.result()) for task_id, future in running.items())
        if stop:
            for task_id in prepared_tasks.keys() - outcomes.keys():
                outcomes[task_id] = DemoTaskOutcome(task_id, stop)
        return tuple(outcomes[task_id] for task_id in prepared_tasks)

    def _run_task(
        self, task_id: str, request: DemoCount, statement: Select, deadline: float,
    ) -> DemoTaskOutcome:
        if time.monotonic() >= deadline:
            return DemoTaskOutcome(task_id, "deadline")
        try:
            rows = self._execute_statement(statement, deadline=deadline)
        except _PlanDeadline:
            return DemoTaskOutcome(task_id, "deadline")
        except DemoPreviewTooLarge:
            outcome = DemoTaskOutcome(task_id, "failed", reason="result_limit")
        except SQLAlchemyError:
            outcome = DemoTaskOutcome(task_id, "failed", reason="query_failed")
        else:
            result = DemoTaskResult(task_id, request, rows)
            outcome = DemoTaskOutcome(task_id, "success", result=result)
        if time.monotonic() >= deadline:
            return DemoTaskOutcome(task_id, "deadline")
        return outcome

    def _prepare_plan(self, plan: DemoPlan) -> tuple[_PreparedTask, ...]:
        tasks = {task.task_id: task for task in plan.ordered_tasks()}
        statements = {task_id: self._statement(task.request) for task_id, task in tasks.items()}
        prepared = []
        for task in tasks.values():
            version = self._versions[task.request.version_id]
            evidence_types = []
            for ref in task.evidence_filters:
                if ref.column not in version.columns:
                    raise PermissionError("The requested column is unavailable in this dataset version.")
                source = tasks[ref.source_task_id].request
                output = statements[ref.source_task_id].selected_columns
                if isinstance(source, DemoPreview):
                    allowed = source.columns or self._versions[source.version_id].columns
                    has_evidence = ref.source_column in allowed and ref.row_index < source.limit
                else:
                    has_evidence = ref.source_column in output and ref.row_index == 0
                if not has_evidence:
                    raise ValueError("The evidence reference is outside the predecessor output.")
                source_type = _evidence_type(output[ref.source_column])
                target_type = _evidence_type(version.relation.c[ref.column])
                if source_type is not target_type and (source_type, target_type) != (int, float):
                    raise ValueError("The predecessor evidence type is incompatible with the filter column.")
                if target_type is bool and ref.comparison not in ("eq", "ne"):
                    raise ValueError("Boolean evidence filters require an equality comparison.")
                evidence_types.append(source_type)
            prepared.append(_PreparedTask(task, statements[task.task_id], tuple(evidence_types)))
        return tuple(prepared)

    def _bind_task(
        self, prepared: _PreparedTask, results: dict[str, DemoTaskResult],
    ) -> tuple[DemoCount, Select]:
        task = prepared.task
        if not task.evidence_filters:
            return task.request, prepared.statement
        filters = tuple(
            _resolve_evidence(ref, results[ref.source_task_id].rows, expected)
            for ref, expected in zip(task.evidence_filters, prepared.evidence_types, strict=True)
        )
        request = task.request.model_copy(update={"filters": task.request.filters + filters})
        return request, self._statement(request)

    def _statement(self, request: DemoCount) -> Select:
        version = self._versions.get(request.version_id)
        if version is None:
            raise PermissionError("This dataset version is unavailable in the demo workspace.")

        relation = version.relation
        approved = {name: relation.c[name] for name in version.columns}

        def column(name: str):
            if name not in approved:
                raise PermissionError("The requested column is unavailable in this dataset version.")
            return approved[name]

        predicates = [
            _COMPARISONS[item.comparison](column(item.column), item.value)
            for item in request.filters
        ]
        if not isinstance(request, DemoPreview):
            return select(func.count().label("row_count")).select_from(relation).where(*predicates)

        projected = tuple(column(name) for name in dict.fromkeys(request.columns or version.columns))
        sort_keys = [(column(item.column), item.descending) for item in request.sort]
        sort_keys.append((relation.c[version.row_key], False))
        bounded = (
            select(*dict.fromkeys((*projected, *(key for key, _ in sort_keys))))
            .where(*predicates)
            .order_by(*(key.desc() if descending else key.asc() for key, descending in sort_keys))
            .limit(request.limit).offset(request.offset).cte()
        )
        visible_columns = tuple(bounded.c[item.key] for item in projected)
        payload = select(*visible_columns).subquery("preview_payload")
        row_json = cast(func.row_to_json(literal_column("preview_payload.*")), Text)
        row_bytes = func.octet_length(func.convert_to(row_json, "UTF8"))
        # Include the array brackets and separators. SUM is NULL for an empty preview.
        size = select((func.sum(row_bytes + 1) + 1).label("byte_count")).select_from(payload).subquery()
        return (
            select(*visible_columns, size.c.byte_count.label(None))
            .select_from(size.outerjoin(bounded, size.c.byte_count <= DEMO_PREVIEW_MAX_BYTES))
            .order_by(*(
                bounded.c[key.key].desc() if descending else bounded.c[key.key].asc()
                for key, descending in sort_keys
            ))
            .execution_options(demo_preview=True)
        )

    def _execute_statement(self, statement: Select, *, deadline: float | None = None) -> list[dict]:
        with demo_read_transaction(self._engine) as connection:
            bounded_by_plan = False
            if deadline is not None:
                remaining_ms = int((deadline - time.monotonic()) * 1000)
                if remaining_ms < 1:
                    raise _PlanDeadline
                if remaining_ms <= 3000:
                    bounded_by_plan = True
                    connection.execute(select(func.set_config("statement_timeout", str(remaining_ms), True)))
            try:
                result = connection.execute(statement)
                if not statement.get_execution_options().get("demo_preview"):
                    return [dict(row) for row in result.mappings()]
                # The size row remains visible when the join suppresses the entire preview.
                rows = result.fetchall()
                byte_count = rows[0][-1]
                if byte_count is None:
                    return []
                if byte_count > DEMO_PREVIEW_MAX_BYTES:
                    raise DemoPreviewTooLarge("This preview exceeds the result size limit.")
                keys = tuple(result.keys())[:-1]
                return [dict(zip(keys, row[:-1], strict=True)) for row in rows]
            except DBAPIError as exc:
                if bounded_by_plan and getattr(exc.orig, "pgcode", None) == "57014":
                    raise _PlanDeadline from None
                raise
