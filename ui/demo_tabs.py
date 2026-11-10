"""Streamlit views for the bounded demo reader."""

import streamlit as st
from pydantic import TypeAdapter
from sqlalchemy.exc import SQLAlchemyError

from db.demo_queries import DemoCount, DemoFilter, DemoPreview, DemoQueries, DemoSort
from ui.service_status import render_service_unavailable


def render_demo_chat(queries: DemoQueries) -> None:
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
    label = versions[selected].label
    starter = f"How many rows are in {label}?"
    st.text(starter)
    question = st.chat_input("Ask for a row count or preview", key="demo_chat_input")
    if not question:
        return

    normalized = question.strip().rstrip("?.").casefold()
    if normalized in {"row count", "how many rows", starter.rstrip("?").casefold()}:
        request = DemoCount(version_id=selected)
    elif normalized in {"preview", "show rows", f"preview {label.casefold()}"}:
        request = DemoPreview(version_id=selected, limit=10)
    else:
        st.info("Try the row-count question above or ask for a preview.")
        return

    try:
        rows = queries.execute(request)
    except (PermissionError, SQLAlchemyError) as exc:
        render_service_unavailable("Dataset", "demo_chat", exc)
        return
    with st.chat_message("assistant"):
        if isinstance(request, DemoPreview):
            st.dataframe(rows, hide_index=True)
        else:
            st.write(f"{label} has {rows[0]['row_count']:,} rows.")


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
    st.dataframe(rows, hide_index=True)
