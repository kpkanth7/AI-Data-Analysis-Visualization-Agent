"""The approved demo versions for this release and their reader contract."""

import csv
from pathlib import Path
from uuid import UUID

from sqlalchemy import MetaData, text
from sqlalchemy.engine import Connection

from db.demo_queries import DemoVersion


DEMO_DIR = Path(__file__).resolve().parents[1] / "data" / "demo"
DEMO_READER = "w1_demo_reader_v2"
DEMO_WORKSPACE = "shared_demo"
ROW_KEY = "_demo_row_id"
DEMO_IDS = {
    "titanic": UUID("a0f905b5-951d-45c5-9865-d3c12bc75dfb"),
    "sp500_financials": UUID("b18883cd-f289-4ea0-a864-b704d67c1a59"),
    "world_co2_emissions": UUID("fc6d0a4f-74e4-4197-a4bb-78a0224d394b"),
}


def load_demo_versions(connection: Connection) -> tuple[DemoVersion, ...]:
    is_reader = connection.execute(text("""
        SELECT session_user = :reader AND current_user = :reader AND rolcanlogin
               AND NOT (rolsuper OR rolbypassrls OR rolcreaterole OR rolcreatedb OR rolreplication)
               AND NOT EXISTS (SELECT 1 FROM pg_catalog.pg_auth_members WHERE member = r.oid)
        FROM pg_catalog.pg_roles AS r WHERE rolname = current_user
    """), {"reader": DEMO_READER}).scalar_one()
    if not is_reader:
        raise PermissionError("Demo queries require the dedicated reader login.")

    metadata = MetaData()
    metadata.reflect(bind=connection, schema="demo", only=tuple(DEMO_IDS), resolve_fks=False)
    versions = []
    for slug, version_id in DEMO_IDS.items():
        can_read = connection.execute(text("""
            SELECT pg_catalog.has_schema_privilege('demo', 'USAGE')
                   AND pg_catalog.has_table_privilege(:relation, 'SELECT')
        """), {"relation": f"demo.{slug}"}).scalar_one()
        if not can_read:
            raise PermissionError("A required demo relation is unavailable to the reader.")
        with (DEMO_DIR / f"{slug}.csv").open(encoding="utf-8", newline="") as demo:
            columns = tuple(next(csv.reader(demo), ()))
        versions.append(DemoVersion(
            version_id, DEMO_WORKSPACE, slug, metadata.tables[f"demo.{slug}"], columns, ROW_KEY,
        ))
    return tuple(versions)
