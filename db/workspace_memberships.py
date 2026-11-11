"""Durable membership for verified OIDC identities."""

from dataclasses import dataclass
from uuid import UUID, uuid4

from sqlalchemy import Boolean, Column, MetaData, Table, Text, or_, select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.schema import CreateSchema
from sqlalchemy.types import Uuid


MEMBERSHIP_READER = "w1_membership_reader"
_SCHEMA = "workspace_auth"
_MEMBERSHIPS = Table(
    "memberships", MetaData(),
    Column("issuer", Text, primary_key=True),
    Column("subject", Text, primary_key=True),
    Column("principal_id", Uuid, nullable=False, unique=True),
    Column("workspace_id", Uuid, nullable=False),
    Column("revision", Uuid, nullable=False),
    Column("can_manage", Boolean, nullable=False),
    Column("active", Boolean, nullable=False),
    schema=_SCHEMA,
)


@dataclass(frozen=True)
class WorkspaceMembership:
    principal_id: str
    workspace_id: UUID
    revision: UUID
    can_manage: bool


def create_membership_registry(connection: Connection) -> None:
    """Create the registry with the migration connection."""
    connection.execute(CreateSchema(_SCHEMA))
    _MEMBERSHIPS.create(connection)


def grant_membership(
    connection: Connection, issuer: str, subject: str, workspace_id: UUID, *, can_manage: bool,
) -> None:
    """Set access from an operator connection. An identical grant keeps its revision."""
    values = {
        "workspace_id": workspace_id, "can_manage": can_manage, "revision": uuid4(), "active": True,
    }
    columns = _MEMBERSHIPS.c
    connection.execute(
        insert(_MEMBERSHIPS)
        .values(issuer=issuer, subject=subject, principal_id=uuid4(), **values)
        .on_conflict_do_update(
            index_elements=[columns.issuer, columns.subject],
            set_=values,
            where=or_(~columns.active, columns.workspace_id != workspace_id, columns.can_manage != can_manage),
        )
    )


def revoke_membership(connection: Connection, issuer: str, subject: str) -> None:
    columns = _MEMBERSHIPS.c
    connection.execute(
        update(_MEMBERSHIPS)
        .where(columns.issuer == issuer, columns.subject == subject, columns.active)
        .values(active=False, revision=uuid4())
    )


class MembershipRegistry:
    def __init__(self, reader: Engine) -> None:
        self._reader = reader

    def lookup(self, issuer: str, subject: str) -> WorkspaceMembership | None:
        columns = _MEMBERSHIPS.c
        with self._reader.begin() as connection:
            connection.exec_driver_sql("SET TRANSACTION READ ONLY")
            connection.exec_driver_sql("SET LOCAL statement_timeout = '3s'")
            connection.exec_driver_sql("SET LOCAL search_path = pg_catalog")
            restricted = connection.execute(text("""
                SELECT session_user = :role AND current_user = :role AND rolcanlogin
                       AND NOT (rolsuper OR rolbypassrls OR rolcreaterole OR rolcreatedb OR rolreplication)
                       AND NOT EXISTS (SELECT 1 FROM pg_catalog.pg_auth_members WHERE member = r.oid)
                FROM pg_catalog.pg_roles AS r WHERE rolname = current_user
            """), {"role": MEMBERSHIP_READER}).scalar_one()
            if not restricted:
                raise PermissionError("Workspace membership requires the registry reader.")
            row = connection.execute(
                select(columns.principal_id, columns.workspace_id, columns.revision, columns.can_manage)
                .where(columns.issuer == issuer, columns.subject == subject, columns.active)
            ).one_or_none()
        if row is None:
            return None
        return WorkspaceMembership(str(row.principal_id), row.workspace_id, row.revision, row.can_manage)
