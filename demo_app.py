"""Restricted Streamlit entry point for the curated demo release."""

import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import NullPool

from db.demo_catalog import DEMO_READER, DEMO_WORKSPACE, load_demo_versions
from db.demo_queries import DemoAccess, DemoQueries, demo_read_transaction
from ui.demo_tabs import render_demo_chat, render_demo_explorer
from ui.service_status import render_service_unavailable


def main() -> None:
    st.set_page_config(page_title="Data Analysis Agent Demo", layout="wide")
    st.title("Data Analysis Agent Demo")
    engine = None
    try:
        url = make_url(st.secrets["demo_reader"]["url"])
        if url.drivername != "postgresql+psycopg2":
            raise ValueError("The demo reader requires a psycopg2 PostgreSQL URL.")
        engine = create_engine(
            url, poolclass=NullPool, hide_parameters=True, connect_args={"connect_timeout": 5},
        )
        with demo_read_transaction(engine) as connection:
            versions = load_demo_versions(connection)

        queries = DemoQueries(engine, DemoAccess(DEMO_READER, DEMO_WORKSPACE), versions)
        explorer, chat = st.tabs(["Explorer", "Chat"])
        with explorer:
            render_demo_explorer(queries)
        with chat:
            render_demo_chat(queries)
    except (KeyError, TypeError, ValueError, OSError, SQLAlchemyError) as exc:
        render_service_unavailable("Dataset", "demo_startup", exc)
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    main()
