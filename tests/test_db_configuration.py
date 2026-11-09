"""Construct real SQLAlchemy engines without connecting to a database."""
import importlib

import psycopg2
import pytest
from sqlalchemy import create_engine

from db import postgres


@pytest.fixture(autouse=True)
def no_database_connections(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("Engine smoke tests must not connect to a database")

    monkeypatch.setattr(psycopg2, "connect", blocked)
    postgres.get_engine.cache_clear()
    yield
    postgres.get_engine.cache_clear()


@pytest.mark.parametrize("scheme", ["postgresql", "postgres"])
def test_bare_postgres_url_uses_installed_driver(monkeypatch, scheme):
    monkeypatch.setattr(postgres, "get_db_url", lambda: f"{scheme}://test:synthetic@localhost/example")
    engine = postgres.get_engine()
    try:
        assert engine.dialect.driver == "psycopg2"
        assert engine.dialect.dbapi is psycopg2
    finally:
        engine.dispose()


def test_url_normalization_preserves_credentials_and_options():
    url = postgres.normalize_db_url(
        "postgresql://test:p%40ss%2Fword%25@localhost:5433/example?sslmode=require&application_name=smoke"
    )
    assert url.password == "p@ss/word%"
    assert url.username == "test"
    assert url.host == "localhost" and url.port == 5433 and url.database == "example"
    assert dict(url.query) == {"sslmode": "require", "application_name": "smoke"}


@pytest.mark.parametrize("driver", ["psycopg2", "psycopg", "pg8000"])
def test_explicit_driver_is_preserved(driver):
    assert postgres.normalize_db_url(f"postgresql+{driver}://localhost/example").drivername == f"postgresql+{driver}"


def test_setup_uses_same_driver_without_running_setup(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:synthetic@localhost/example")
    setup = importlib.import_module("scripts.setup_db")
    setup = importlib.reload(setup)
    engine = create_engine(setup.DB_URL)
    try:
        assert engine.dialect.driver == "psycopg2"
    finally:
        engine.dispose()
