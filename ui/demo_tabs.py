"""Streamlit views for the bounded demo reader."""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from html import escape
from uuid import UUID, uuid4

import streamlit as st
from pydantic import TypeAdapter
from sqlalchemy.exc import SQLAlchemyError

from core.csv_export import csv_bytes
from db.demo_queries import (
    DemoAccess, DemoCount, DemoCountTask, DemoFilter, DemoPlan, DemoPreview, DemoPreviewTask,
    DemoPreviewTooLarge, DemoQueries, DemoSort, DemoTaskOutcome, DemoTaskResult, DemoVersion,
)
from ui.csv_download import render_csv_link
from ui.service_status import render_service_unavailable


logger = logging.getLogger(__name__)

_DEMO_QUERY_POLICY_VERSION = 1
_PREVIEW_LIMIT_MESSAGE = (
    "This preview exceeds the size limit. Try selecting fewer rows or narrowing the filter in Explorer."
)
_COUNT_QUESTION = re.compile(
    r"count(?: (?:the )?(?:rows|records))?"
    r"|how many (?:rows|records)(?: are there)?"
    r"|(?:(?:give me|show(?: me)?|tell me|what(?: is|['’]s)) )?"
    r"(?:(?:the |a )?total |the )?(?:row count|number of (?:rows|records))"
)
_PREVIEW_QUESTION = re.compile(
    r"preview|show (?:me )?(?:the )?rows"
    r"|(?:(?:show(?: me)?|give me|can i see) )?"
    r"a (?:preview|sample)(?: of (?:the )?(?:rows|records))?"
)


@dataclass(frozen=True)
class DemoQueryScope:
    access: DemoAccess
    version_id: UUID
    relation: str
    row_key: str
    columns: tuple[str, ...]
    policy_version: int


def demo_query_scope(access: DemoAccess, version: DemoVersion) -> DemoQueryScope:
    return DemoQueryScope(
        access, version.version_id, version.relation.fullname, version.row_key,
        version.columns, _DEMO_QUERY_POLICY_VERSION,
    )


class DemoScopeChanged(PermissionError):
    """The selected catalog or reader changed after rendering the dataset picker."""


def _clear_chat_checkpoint() -> None:
    for key in ("demo_chat_scope", "demo_chat_last_operation", "demo_chat_failed_request", "demo_chat_retry_requested"):
        st.session_state.pop(key, None)


def _show_changed_access() -> None:
    _clear_chat_checkpoint()
    st.info("Dataset access changed. Reload the page and ask a new question.")

_DEMO_ROW_DESCRIPTIONS = {
    "titanic": "One row per passenger, with ticket, travel, and survival details.",
    "sp500_financials": "One row per stock symbol, with company, sector, and financial metrics.",
    "world_co2_emissions": "One row per country or region and year, with emissions and related indicators.",
}


def _render_dataset_brief(version: DemoVersion) -> None:
    description = _DEMO_ROW_DESCRIPTIONS.get(version.label)
    if description:
        st.write(description)
    with st.expander(f"Columns ({len(version.columns)})"):
        st.write(", ".join(version.columns))


def _render_preview(version: DemoVersion, rows: list[dict]) -> None:
    columns = version.columns
    data = csv_bytes(columns, ([row[column] for column in columns] for row in rows))
    unit = "row" if len(rows) == 1 else "rows"
    render_csv_link(data, f"Preview CSV ({len(rows):,} {unit})", f"{version.label}-preview.csv")
    st.dataframe(rows, hide_index=True)


def _show_demo_question_retry() -> None:
    st.error("This question could not run. Please retry in a moment.")
    st.button("Retry question", key="demo_chat_retry", on_click=_request_demo_chat_retry)


def _request_demo_chat_retry() -> None:
    st.session_state["demo_chat_retry_requested"] = True


def render_demo_chat(
    queries: DemoQueries, *, execute: Callable[[DemoCount], list[dict]] | None = None,
    run_plan: Callable[[DemoPlan], tuple[DemoTaskOutcome, ...]] | None = None,
) -> None:
    st.header("Ask about the demo data")
    versions = {version.version_id: version for version in queries.versions}
    if not versions:
        _clear_chat_checkpoint()
        st.info("No demo datasets are available.")
        return
    selected = st.selectbox(
        "Demo dataset", list(versions), format_func=lambda value: versions[value].label,
        key="demo_chat_dataset",
    )
    if selected is None:
        _clear_chat_checkpoint()
        return
    version = versions[selected]
    if "demo_chat_conversation" not in st.session_state:
        st.session_state["demo_chat_conversation"] = uuid4()
    scope = (st.session_state["demo_chat_conversation"], demo_query_scope(queries.access, version))
    if scope != st.session_state.get("demo_chat_scope"):
        _clear_chat_checkpoint()
        st.session_state["demo_chat_scope"] = scope
    _render_dataset_brief(version)
    label = version.label
    st.text(f"How many rows are in {label}?")
    question = st.chat_input("Ask for a row count, preview, or both", key="demo_chat_input")
    with st.container():
        outcomes = _run_demo_question(
            version, queries.versions, question, execute or queries.execute, run_plan or queries.run_plan,
        )

    messages = []
    for outcome in outcomes:
        result = outcome.result
        if outcome.reason == "result_limit":
            messages.append(_PREVIEW_LIMIT_MESSAGE)
        elif result is None:
            messages.append(f"The {outcome.task_id} could not finish.")
        elif isinstance(result.request, DemoPreview):
            messages.append(f"Showing {len(result.rows)} rows from {label}.")
        else:
            messages.append(f"{label} has {result.rows[0]['row_count']:,} rows.")
    announcement = " ".join(messages)

    # Keep the same live region mounted when idle, running, and complete.
    st.markdown(
        '<span role="status" aria-atomic="true" '
        'style="position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)">'
        f"{escape(announcement)}</span>",
        unsafe_allow_html=True,
    )
    if outcomes:
        with st.chat_message("assistant"):
            for outcome, message in zip(outcomes, messages):
                if outcome.result is None:
                    st.warning(message)
                elif isinstance(outcome.result.request, DemoPreview):
                    _render_preview(version, outcome.result.rows)
                else:
                    st.write(message)


def _demo_operation(question: str, label: str) -> str | None:
    for suffix in (
        f" are in {label}", f" in {label}", f" of {label}", f" {label}",
        " are in this dataset", " in this dataset", " of this dataset",
    ):
        if question.endswith(suffix):
            question = question.removesuffix(suffix)
            break
    if question == "both":
        return "both"

    def single_operation(phrase: str) -> str | None:
        if _COUNT_QUESTION.fullmatch(phrase):
            return "count"
        if _PREVIEW_QUESTION.fullmatch(phrase):
            return "preview"
        return None

    first, conjunction, second = question.partition(" and ")
    operation = single_operation(first)
    if not conjunction:
        return operation
    if {operation, single_operation(second)} == {"count", "preview"}:
        return "both"
    return None


def _run_demo_question(
    version: DemoVersion, versions: tuple[DemoVersion, ...], question: str | None,
    execute: Callable[[DemoCount], list[dict]], run_plan: Callable[[DemoPlan], tuple[DemoTaskOutcome, ...]],
) -> tuple[DemoTaskOutcome, ...]:
    selected = version.version_id
    label = version.label.casefold()
    retry_requested = st.session_state.pop("demo_chat_retry_requested", False)
    request = st.session_state.get("demo_chat_failed_request")
    if question or (request is not None and request.version_id != selected):
        st.session_state.pop("demo_chat_failed_request", None)
        request = None

    if question:
        normalized = " ".join(question.casefold().split()).rstrip("?.").rstrip()
        previous = st.session_state.pop("demo_chat_last_operation", None)
        if normalized in {"again", "repeat"}:
            if previous is None:
                st.info(f"Ask a new question about {version.label}. You can request a row count, a preview, or both.")
                return ()
            normalized = previous
        operation = _demo_operation(normalized, label)
        if operation == "count":
            request = DemoCount(version_id=selected)
        elif operation == "preview":
            request = DemoPreview(version_id=selected, limit=10)
        elif operation == "both":
            plan = DemoPlan(tasks=(
                DemoCountTask(task_id="count", operation="count", request=DemoCount(version_id=selected)),
                DemoPreviewTask(task_id="preview", operation="preview", request=DemoPreview(version_id=selected, limit=10)),
            ))
            try:
                with st.spinner("Running demo question…"):
                    outcomes = run_plan(plan)
                if len(outcomes) == 2 and all(outcome.status == "success" for outcome in outcomes):
                    st.session_state["demo_chat_last_operation"] = "both"
                return outcomes
            except DemoScopeChanged:
                _show_changed_access()
                return ()
            except (KeyError, TypeError, ValueError, OSError, SQLAlchemyError) as exc:
                logger.warning("Demo plan unavailable: %s", type(exc).__name__)
                st.error("This question could not run. Please submit it again in a moment.")
                return ()
        else:
            for other in versions:
                name = other.label.casefold()
                if name != label and _demo_operation(normalized, name):
                    st.info(f"Choose {other.label} in the dataset picker, then ask again.")
                    return ()
            st.info(f"For {version.label}, would you like a row count, a preview, or both?")
            return ()
    elif request is None:
        return ()
    elif not retry_requested:
        _show_demo_question_retry()
        return ()

    try:
        with st.spinner("Running demo question…"):
            rows = execute(request)
    except DemoScopeChanged:
        _show_changed_access()
        return ()
    except DemoPreviewTooLarge:
        st.session_state.pop("demo_chat_failed_request", None)
        return (DemoTaskOutcome("preview", "failed", reason="result_limit"),)
    except (KeyError, TypeError, ValueError, OSError, SQLAlchemyError) as exc:
        logger.warning("Demo question unavailable: %s", type(exc).__name__)
        st.session_state["demo_chat_failed_request"] = request
        _show_demo_question_retry()
        return ()
    st.session_state.pop("demo_chat_failed_request", None)
    task_id = "preview" if isinstance(request, DemoPreview) else "count"
    st.session_state["demo_chat_last_operation"] = task_id
    return (DemoTaskOutcome(task_id, "success", DemoTaskResult(task_id, request, rows)),)


def render_demo_explorer(queries: DemoQueries) -> None:
    st.header("Demo Data Explorer")
    versions = {version.version_id: version for version in queries.versions}
    if not versions:
        st.info("No demo datasets are available.")
        return
    selected = st.selectbox(
        "Demo dataset", list(versions), format_func=lambda value: versions[value].label,
        key="demo_explorer_dataset",
    )
    if selected is None:
        return
    version = versions[selected]
    _render_dataset_brief(version)
    choices = [None, *version.columns]
    sort = st.selectbox("Sort by", choices, key=f"demo_sort_{selected}")
    filter_column = st.selectbox("Filter column", choices, key=f"demo_filter_{selected}")
    filters = ()
    if filter_column is not None:
        raw = st.text_input("Equals", max_chars=2048, key=f"demo_filter_value_{selected}_{filter_column}")
        if raw:
            try:
                column_type = version.relation.c[filter_column].type.python_type
                value = TypeAdapter(column_type).validate_python(raw)
                filters = (DemoFilter(column=filter_column, value=value),)
            except ValueError:
                st.warning("Enter a value that matches the selected column type.")
                return

    limit = st.selectbox("Preview rows", [25, 50, 100], index=1, key="demo_explorer_limit")
    request = DemoPreview(
        version_id=selected, filters=filters,
        sort=(DemoSort(column=sort),) if sort else (), limit=limit,
    )
    try:
        total = queries.execute(DemoCount(version_id=selected, filters=filters))[0]["row_count"]
        rows = queries.execute(request)
    except DemoPreviewTooLarge:
        rows = None
    except (PermissionError, SQLAlchemyError) as exc:
        render_service_unavailable("Dataset", "demo_explorer", exc)
        return
    unit = "row" if total == 1 else "rows"
    st.caption(f"{total:,} matching {unit}")
    if rows is None:
        st.warning(_PREVIEW_LIMIT_MESSAGE)
    else:
        _render_preview(version, rows)
