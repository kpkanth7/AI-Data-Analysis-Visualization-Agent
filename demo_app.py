"""Restricted Streamlit entry point for the curated demo release."""

from collections.abc import Iterator
from contextlib import contextmanager
from uuid import UUID

import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import NullPool

from db.demo_catalog import DEMO_READER, DEMO_WORKSPACE, load_demo_versions
from db.demo_queries import DemoAccess, DemoCount, DemoPlan, DemoQueries, DemoTaskOutcome, demo_read_transaction
from ui.demo_tabs import DemoQueryScope, DemoScopeChanged, demo_query_scope, render_demo_chat, render_demo_explorer
from ui.service_status import render_service_unavailable


@contextmanager
def _demo_queries() -> Iterator[DemoQueries]:
    url = make_url(st.secrets["demo_reader"]["url"])
    if url.drivername != "postgresql+psycopg2":
        raise ValueError("The demo reader requires a psycopg2 PostgreSQL URL.")
    engine = create_engine(
        url, poolclass=NullPool, hide_parameters=True, connect_args={"connect_timeout": 5},
    )
    try:
        with demo_read_transaction(engine) as connection:
            versions = load_demo_versions(connection)
        yield DemoQueries(engine, DemoAccess(DEMO_READER, DEMO_WORKSPACE), versions)
    finally:
        engine.dispose()


@contextmanager
def _fresh_chat_reader(rendered: DemoQueries, version_ids: set[UUID]) -> Iterator[DemoQueries]:
    # A fragment rerun skips main(). Check the picker against the newly admitted catalog.
    def scopes(reader: DemoQueries) -> dict[UUID, DemoQueryScope]:
        return {
            version.version_id: demo_query_scope(reader.access, version)
            for version in reader.versions if version.version_id in version_ids
        }

    try:
        with _demo_queries() as fresh:
            current = scopes(fresh)
            if current.keys() != version_ids or current != scopes(rendered):
                raise PermissionError("The selected dataset scope changed.")
            yield fresh
    except PermissionError as exc:
        raise DemoScopeChanged from exc


@st.fragment
def _render_chat(queries: DemoQueries) -> None:
    def execute(request: DemoCount) -> list[dict]:
        with _fresh_chat_reader(queries, {request.version_id}) as reader:
            return reader.execute(request)

    def run_plan(plan: DemoPlan) -> tuple[DemoTaskOutcome, ...]:
        with _fresh_chat_reader(queries, {task.request.version_id for task in plan.tasks}) as reader:
            return reader.run_plan(plan)

    render_demo_chat(queries, execute=execute, run_plan=run_plan)


def main() -> None:
    st.set_page_config(page_title="Data Analysis Agent Demo", layout="wide")
    st.html("""
        <style>
        [data-testid="stTab"][aria-selected="true"],
        [data-testid="stTab"][aria-selected="true"] p { color: inherit; }
        [data-testid="stDataFrame"] button[aria-label="Download as CSV"] { display: none; }
        </style>
    """)
    st.title("Data Analysis Agent Demo")
    st.write("Browse read-only sample datasets in Explorer, or use Chat to count or preview rows.")
    st.caption("Curated samples cannot be deleted.")
    try:
        with _demo_queries() as queries:
            explorer, chat = st.tabs(["Explorer", "Chat"])
            with explorer:
                render_demo_explorer(queries)
            with chat:
                _render_chat(queries)
    except (KeyError, TypeError, ValueError, OSError, SQLAlchemyError) as exc:
        render_service_unavailable("Dataset", "demo_startup", exc)


if __name__ == "__main__":
    main()
