"""Restricted Streamlit entry point for the curated demo release."""

from collections.abc import Iterator
from contextlib import contextmanager

import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import NullPool

from db.demo_catalog import DEMO_READER, DEMO_WORKSPACE, load_demo_versions
from db.demo_queries import DemoAccess, DemoCount, DemoQueries, demo_read_transaction
from ui.demo_tabs import render_demo_chat, render_demo_explorer
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


def _execute_demo_question(request: DemoCount) -> list[dict]:
    # A fragment rerun skips main(), so recheck the reader and catalog here.
    with _demo_queries() as queries:
        return queries.execute(request)


@st.fragment
def _render_chat(queries: DemoQueries) -> None:
    render_demo_chat(queries, execute=_execute_demo_question)


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
