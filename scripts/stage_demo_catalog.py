"""Prepare demo copies and change the dedicated reader's access in a transaction."""

import pandas as pd
from sqlalchemy.engine import Connection

from db.demo_catalog import DEMO_DIR, DEMO_IDS, DEMO_READER, ROW_KEY


def stage_demo_catalog(connection: Connection) -> None:
    """Create a fresh schema. An existing demo schema aborts the staging transaction."""
    connection.exec_driver_sql("CREATE SCHEMA demo")
    for slug in DEMO_IDS:
        data = pd.read_csv(DEMO_DIR / f"{slug}.csv")
        data.insert(0, ROW_KEY, range(1, len(data) + 1))
        data.to_sql(slug, connection, schema="demo", index=False, if_exists="fail")
        connection.exec_driver_sql(f'ALTER TABLE demo."{slug}" ADD PRIMARY KEY ({ROW_KEY})')


def grant_demo_access(connection: Connection) -> None:
    connection.exec_driver_sql(f"GRANT USAGE ON SCHEMA demo TO {DEMO_READER}")
    for slug in DEMO_IDS:
        connection.exec_driver_sql(f'GRANT SELECT ON demo."{slug}" TO {DEMO_READER}')


def revoke_demo_access(connection: Connection) -> None:
    for slug in DEMO_IDS:
        connection.exec_driver_sql(f'REVOKE SELECT ON demo."{slug}" FROM {DEMO_READER}')
    connection.exec_driver_sql(f"REVOKE USAGE ON SCHEMA demo FROM {DEMO_READER}")
