"""Report redacted DB runtime evidence without starting the application."""

import argparse
import json
import platform
from importlib.metadata import PackageNotFoundError, version

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.pool import NullPool

from db.postgres import get_db_url, normalize_db_url


_SUPPORTED_SCHEMES = {"postgres", "postgresql", "postgresql+psycopg2", "postgresql+psycopg"}
_ROLE_SQL = """
    SELECT current_user = 'postgres' AS is_postgres,
           rolsuper AS superuser, rolcreaterole AS create_role,
           rolcreatedb AS create_database, rolbypassrls AS bypass_rls,
           has_schema_privilege('public', 'CREATE') AS create_in_public,
           coalesce(has_table_privilege(to_regclass('datasets'), 'SELECT'), false)
               AS can_select_catalog,
           coalesce(has_table_privilege(to_regclass('guest_usage'), 'SELECT'), false)
               AS can_select_guest_usage
    FROM pg_roles
    WHERE rolname = current_user
"""
_TABLE_SQL = """
    SELECT count(*) AS tables,
           count(*) FILTER (WHERE pg_has_role(c.relowner, 'USAGE')) AS ownership_access,
           count(*) FILTER (WHERE has_table_privilege(c.oid, 'SELECT')) AS select_grant,
           count(*) FILTER (
               WHERE has_table_privilege(c.oid, 'INSERT, UPDATE, DELETE, TRUNCATE')
           ) AS write_grant,
           count(*) FILTER (WHERE c.relrowsecurity) AS rls_enabled
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p')
"""


def _installed_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _probe(connect: bool) -> dict:
    report = {
        "python": platform.python_version(),
        "packages": {
            package: _installed_version(package)
            for package in ("streamlit", "SQLAlchemy", "psycopg2-binary", "psycopg")
        },
        "database_ready": None,
    }
    stage = "configuration"
    try:
        configured = make_url(get_db_url())
        if configured.drivername not in _SUPPORTED_SCHEMES:
            report["error"] = {"stage": stage, "type": "UnsupportedDatabaseScheme"}
            return report
        report["configured_scheme"] = configured.drivername
        url = normalize_db_url(configured)
        report["effective_scheme"] = url.drivername

        stage = "engine"
        engine = create_engine(url, poolclass=NullPool, connect_args={"connect_timeout": 5})
        report["loaded_driver"] = engine.dialect.driver
        if connect:
            report["database_ready"] = False
            stage = "connection"
            with engine.connect() as connection:
                connection.exec_driver_sql("SET TRANSACTION READ ONLY")
                connection.exec_driver_sql("SET LOCAL statement_timeout = '5s'")
                report["database_ready"] = connection.exec_driver_sql("SELECT 1").scalar_one() == 1
                stage = "privileges"
                report["role"] = dict(connection.exec_driver_sql(_ROLE_SQL).mappings().one())
                report["public_table_grants"] = dict(
                    connection.exec_driver_sql(_TABLE_SQL).mappings().one()
                )
    except Exception as exc:
        report["error"] = {"stage": stage, "type": type(exc).__name__}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connect", action="store_true", help="Run read-only PostgreSQL checks")
    report = _probe(parser.parse_args().connect)
    print(json.dumps(report, indent=2))
    return int("error" in report or report["database_ready"] is False)


if __name__ == "__main__":
    raise SystemExit(main())
