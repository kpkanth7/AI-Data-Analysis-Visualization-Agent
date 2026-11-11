"""Streamlit entry point for the invite-only private workspace pilot."""

import re
from urllib.parse import urlsplit
from uuid import UUID

import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, URL, make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.pool import NullPool

from db.workspace_memberships import MEMBERSHIP_READER, MembershipRegistry
from ui.private_workspace import render_private_workspace
from ui.service_status import render_service_unavailable


_LOOPBACK = {"localhost", "127.0.0.1", "::1"}


def _role_url(value: str, role: str) -> tuple[URL, str]:
    url = make_url(value)
    if url.drivername != "postgresql+psycopg2" or not url.database or not url.password:
        raise ValueError("Use an explicit psycopg2 URL with a scoped database login.")
    if set(url.query) - {"host", "sslmode", "sslrootcert"} or any(
        not isinstance(option, str) for option in url.query.values()
    ):
        raise ValueError("Use supported database connection options.")
    if url.host is None:
        socket = url.query.get("host")
        if not isinstance(socket, str) or not socket.startswith("/"):
            raise ValueError("Specify a database host or Unix socket directory.")
    elif "host" in url.query:
        raise ValueError("Specify the database host once.")
    if url.host and url.host not in _LOOPBACK and url.query.get("sslmode") != "verify-full":
        raise ValueError("Remote database connections require TLS hostname verification.")

    project = ""
    if url.host and url.host.endswith(".pooler.supabase.com"):
        username, _, project = (url.username or "").partition(".")
        if username != role or not re.fullmatch(r"[a-z0-9]{20}", project) or url.port != 5432:
            raise ValueError("Use the scoped login on the Supabase session pooler.")
    elif url.username != role:
        raise ValueError("Use the scoped database login for this connection.")
    return url, project


def _database_target(url: URL, project: str) -> tuple:
    return (url.host, url.port or 5432, url.database, url.query.get("host"), project)


def _engine(url: URL) -> Engine:
    return create_engine(url, poolclass=NullPool, hide_parameters=True, connect_args={"connect_timeout": 5})


def _configuration() -> tuple[str, URL, dict[UUID, tuple[URL, URL]]]:
    auth = st.secrets["auth"]
    for key in ("redirect_uri", "cookie_secret", "client_id", "client_secret", "server_metadata_url"):
        if not isinstance(auth[key], str) or not auth[key].strip():
            raise ValueError("Configure the default Streamlit OIDC provider.")

    settings = st.secrets["private_pilot"]
    issuer = settings["issuer"]
    parsed = urlsplit(issuer)
    if not parsed.hostname or (
        parsed.scheme != "https" and (parsed.scheme != "http" or parsed.hostname not in _LOOPBACK)
    ):
        raise ValueError("Set the expected OIDC issuer.")

    registry_url, project = _role_url(settings["registry_url"], MEMBERSHIP_READER)
    target = _database_target(registry_url, project)
    workspaces = {}
    for key, credentials in settings["workspaces"].items():
        workspace = UUID(key)
        if workspace in workspaces:
            raise ValueError("Each workspace requires a distinct credential entry.")
        schema = f"w_{workspace.hex}"
        reader, reader_project = _role_url(credentials["reader_url"], f"{schema}_reader")
        writer, writer_project = _role_url(credentials["writer_url"], f"{schema}_writer")
        if _database_target(reader, reader_project) != target or _database_target(writer, writer_project) != target:
            raise ValueError("All pilot roles must connect to the same database.")
        workspaces[workspace] = (reader, writer)
    if not workspaces:
        raise ValueError("Configure at least one workspace.")
    return issuer, registry_url, workspaces


def main() -> None:
    st.set_page_config(page_title="Private workspace", layout="wide")
    try:
        issuer, registry_url, workspaces = _configuration()
    except (ArgumentError, AttributeError, KeyError, OSError, TypeError, ValueError):
        st.title("Private workspace")
        render_service_unavailable("Dataset", "private_pilot_configuration")
        return

    registry = MembershipRegistry(_engine(registry_url))

    def lookup(claim_issuer, subject):
        if claim_issuer != issuer:
            return None
        membership = registry.lookup(claim_issuer, subject)
        if membership is None or membership.workspace_id not in workspaces:
            return None
        return membership

    def engines(workspace_id: UUID) -> tuple[Engine, Engine]:
        reader, writer = workspaces[workspace_id]
        return _engine(reader), _engine(writer)

    render_private_workspace(lookup, engines)


if __name__ == "__main__":
    main()
