"""Streamlit views for the bounded demo reader."""

import logging
from collections.abc import Callable
from html import escape
from uuid import UUID

import streamlit as st
from pydantic import TypeAdapter
from sqlalchemy.exc import SQLAlchemyError

from core.csv_export import csv_bytes
from db.demo_queries import DemoCount, DemoFilter, DemoPreview, DemoQueries, DemoSort, DemoVersion
from ui.csv_download import render_csv_link
from ui.service_status import render_service_unavailable


logger = logging.getLogger(__name__)

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
) -> None:
    st.header("Ask about the demo data")
    versions = {version.version_id: version for version in queries.versions}
    if not versions:
        st.info("No demo datasets are available.")
        return
    selected = st.selectbox(
        "Demo dataset", list(versions), format_func=lambda value: versions[value].label,
        key="demo_chat_dataset",
    )
    if selected is None:
        return
    version = versions[selected]
    _render_dataset_brief(version)
    label = version.label
    st.text(f"How many rows are in {label}?")
    question = st.chat_input("Ask for a row count or preview", key="demo_chat_input")
    with st.container():
        result = _run_demo_question(selected, label, question, execute or queries.execute)

    announcement = ""
    if result is not None:
        request, rows = result
        announcement = (
            f"Showing {len(rows)} rows from {label}."
            if isinstance(request, DemoPreview)
            else f"{label} has {rows[0]['row_count']:,} rows."
        )

    # Keep the same live region mounted when idle, running, and complete.
    st.markdown(
        '<span role="status" aria-atomic="true" '
        'style="position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)">'
        f"{escape(announcement)}</span>",
        unsafe_allow_html=True,
    )
    if result is not None:
        with st.chat_message("assistant"):
            if isinstance(request, DemoPreview):
                _render_preview(version, rows)
            else:
                st.write(announcement)


def _run_demo_question(
    selected: UUID, label: str, question: str | None, execute: Callable[[DemoCount], list[dict]],
) -> tuple[DemoCount, list[dict]] | None:
    retry_requested = st.session_state.pop("demo_chat_retry_requested", False)
    request = st.session_state.get("demo_chat_failed_request")
    if question or (request is not None and request.version_id != selected):
        st.session_state.pop("demo_chat_failed_request", None)
        request = None

    if question:
        normalized = question.strip().rstrip("?.").casefold()
        if normalized in {"row count", "how many rows", f"how many rows are in {label.casefold()}"}:
            request = DemoCount(version_id=selected)
        elif normalized in {"preview", "show rows", f"preview {label.casefold()}"}:
            request = DemoPreview(version_id=selected, limit=10)
        else:
            st.info("Try the row-count question above or ask for a preview.")
            return None
    elif request is None:
        return None
    elif not retry_requested:
        _show_demo_question_retry()
        return None

    try:
        with st.spinner("Running demo question…"):
            rows = execute(request)
    except (KeyError, TypeError, ValueError, OSError, SQLAlchemyError) as exc:
        logger.warning("Demo question unavailable: %s", type(exc).__name__)
        st.session_state["demo_chat_failed_request"] = request
        _show_demo_question_retry()
        return None
    st.session_state.pop("demo_chat_failed_request", None)
    return request, rows


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
    except (PermissionError, SQLAlchemyError) as exc:
        render_service_unavailable("Dataset", "demo_explorer", exc)
        return
    unit = "row" if total == 1 else "rows"
    st.caption(f"{total:,} matching {unit}")
    _render_preview(version, rows)
