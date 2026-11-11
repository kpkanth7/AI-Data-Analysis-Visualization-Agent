"""Streamlit flow for a verified identity and server-supplied workspace membership."""

import time
from base64 import b64encode
from collections.abc import Callable
from uuid import UUID, uuid4

import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as iframe_html
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from db.workspace_datasets import (
    CATALOG_SEARCH_LIMIT, CATALOG_SEARCH_MAX_CHARS, PREVIEW_ROWS,
    UploadLimitReached, UploadRequestConflict, WorkspaceAccess, WorkspaceColumnSummary,
    WorkspaceDataset, WorkspaceDatasets, WorkspaceRowCount,
)
from db.workspace_memberships import WorkspaceMembership
from ui.service_status import render_service_unavailable


_PREFIX = "private_workspace_"
_BINDING = f"{_PREFIX}binding"
_SESSION = f"{_PREFIX}session"


MembershipLookup = Callable[[str, str], WorkspaceMembership | None]
WorkspaceEngines = Callable[[UUID], tuple[Engine, Engine]]
WorkspaceBinding = tuple[str, str, WorkspaceMembership]


def _current_binding(lookup: MembershipLookup) -> WorkspaceBinding | None:
    claims = st.user.to_dict()
    issuer, subject, expires = claims.get("iss"), claims.get("sub"), claims.get("exp")
    if (
        claims.get("is_logged_in") is not True
        or not isinstance(issuer, str) or not issuer
        or not isinstance(subject, str) or not subject
        or type(expires) is not int or expires <= time.time()
    ):
        return None
    membership = lookup(issuer, subject)
    return (issuer, subject, membership) if membership is not None else None


def _clear_workspace_state() -> None:
    for key in list(st.session_state):
        if key.startswith(_PREFIX):
            del st.session_state[key]


def _sign_out() -> None:
    _clear_workspace_state()
    st.logout()


@st.fragment(run_every=2)
def _watch_membership(lookup: MembershipLookup, binding: WorkspaceBinding) -> None:
    try:
        if _current_binding(lookup) == binding:
            return
    except (PermissionError, SQLAlchemyError):
        pass
    _clear_workspace_state()
    st.rerun()


def render_private_workspace(lookup: MembershipLookup, engines: WorkspaceEngines) -> None:
    """Resolve claims supplied by Streamlit OIDC. Neither lookup accepts browser workspace IDs."""
    st.title("Private workspace")
    signed_in = getattr(st.user, "is_logged_in", False)
    if signed_in:
        st.button("Sign out", on_click=_sign_out)
    try:
        binding = _current_binding(lookup)
    except (PermissionError, SQLAlchemyError) as exc:
        _clear_workspace_state()
        render_service_unavailable("Dataset", "private_membership", exc)
        return

    if st.session_state.get(_BINDING) != binding:
        _clear_workspace_state()
        if binding is not None:
            st.session_state[_BINDING] = binding
            st.session_state[_SESSION] = uuid4()

    if binding is None:
        st.info("Sign in with an account that has workspace access.")
        if not signed_in:
            st.button("Sign in", on_click=st.login)
        return

    _, _, membership = binding
    _watch_membership(lookup, binding)
    session_id = st.session_state[_SESSION]
    access = WorkspaceAccess(membership.principal_id, session_id, membership.workspace_id, membership.can_manage)

    def resolve_access() -> WorkspaceAccess | None:
        if st.session_state.get(_SESSION) == session_id and _current_binding(lookup) == binding:
            return access
        return None

    content = st.empty()
    try:
        service = WorkspaceDatasets(access, resolve_access, *engines(access.workspace_id))
        with content.container():
            _render_datasets(service, access)
    except PermissionError:
        _clear_workspace_state()
        content.warning("Your workspace or dataset is no longer available. Refresh to check again.")
    except SQLAlchemyError as exc:
        content.empty()
        render_service_unavailable("Dataset", "private_workspace", exc)


def _render_upload(service: WorkspaceDatasets, prefix: str) -> None:
    request_key = f"{prefix}upload_request"
    if request_key not in st.session_state:
        st.session_state[request_key] = uuid4()
    request_id = st.session_state[request_key]
    with st.form(f"{prefix}upload_{request_id.hex}"):
        uploaded = st.file_uploader("Upload CSV", type=["csv"], max_upload_size=1)
        submit = st.form_submit_button("Upload dataset")
    if not submit:
        return
    if uploaded is None:
        st.warning("Choose a CSV file up to 1 MB.")
        return

    try:
        service.upload_csv(uploaded.getvalue(), uploaded.name, request_id)
    except UploadLimitReached:
        st.warning("This workspace has reached its daily upload limit. Try again after midnight UTC.")
    except UploadRequestConflict as exc:
        st.warning(str(exc))
        st.session_state[request_key] = uuid4()
    except ValueError:
        st.warning("Choose a UTF-8 CSV up to 1 MB with valid columns and up to 1,000 rows and 64 columns.")
    except SQLAlchemyError:
        st.warning("Upload status is unavailable. Submit the same file again to check or retry.")
    else:
        st.session_state[request_key] = uuid4()
        st.rerun()


def _render_analysis(service: WorkspaceDatasets, dataset: WorkspaceDataset, prefix: str) -> None:
    with st.expander("Analyze dataset"):
        operation = st.radio("Operation", ("Count rows", "Summarize column"), key=f"{prefix}operation")
        request = WorkspaceRowCount(version_id=dataset.version_id)
        if operation == "Summarize column":
            column = st.selectbox("Column", dataset.columns, key=f"{prefix}column")
            request = WorkspaceColumnSummary(version_id=dataset.version_id, column=column)
        if st.button("Run analysis", key=f"{prefix}run"):
            try:
                result = service.summarize(request)
            except ValueError:
                st.warning("This dataset exceeds the analysis row limit.")
                return
            labels = {
                "row_count": "Rows", "non_null_count": "Non-null values", "null_count": "Missing values",
                "minimum": "Minimum", "maximum": "Maximum", "total": "Sum", "mean": "Mean",
            }
            for key, value in result.items():
                st.text(f"{labels[key]}: {value if value is not None else 'No values'}")


def _render_datasets(service: WorkspaceDatasets, access: WorkspaceAccess) -> None:
    prefix = f"{_PREFIX}{access.session_id.hex}_"
    if access.can_manage:
        _render_upload(service, prefix)

    query = st.text_input(
        "Search by filename or column", key=f"{prefix}search", max_chars=CATALOG_SEARCH_MAX_CHARS,
    )
    try:
        datasets = {dataset.version_id: dataset for dataset in service.list_datasets(query)}
    except ValueError:
        st.warning("Enter a valid search term.")
        return

    selection_key = f"{prefix}selection"
    pending_key = f"{prefix}delete"
    if not datasets:
        st.session_state.pop(pending_key, None)
        st.info("No matching datasets." if query.strip() else "This workspace has no datasets yet.")
        return
    if query.strip() and len(datasets) == CATALOG_SEARCH_LIMIT:
        st.caption(f"Showing the first {CATALOG_SEARCH_LIMIT} matches.")
    # Streamlit serializes the displayed option. Include the version so
    # identical filenames cannot resolve to a different dataset.
    choices = {f"{dataset.filename} · {version_id}": version_id for version_id, dataset in datasets.items()}
    choice = st.selectbox("Dataset", tuple(choices), index=None, key=selection_key)
    selection = choices.get(choice)
    if selection is None:
        st.session_state.pop(pending_key, None)
        return
    if st.session_state.get(pending_key) != selection:
        st.session_state.pop(pending_key, None)

    dataset = datasets[selection]
    unit = "row" if dataset.row_count == 1 else "rows"
    st.caption(f"{dataset.row_count:,} {unit}")
    if dataset.row_count > PREVIEW_ROWS:
        st.caption(f"Showing the first {PREVIEW_ROWS} rows.")
    rows = service.preview(selection)
    st.html(pd.DataFrame(rows, columns=dataset.columns).to_html(index=False, escape=True))
    _render_analysis(service, dataset, f"{prefix}analysis_{selection.hex}_")
    if st.button("Prepare preview CSV", key=f"{prefix}csv_{selection.hex}"):
        # Deliver the bytes to this authenticated Streamlit session without a media URL.
        encoded = b64encode(service.preview_csv(selection)).decode("ascii")
        iframe_html(
            '<meta name="color-scheme" content="light dark">'
            '<a download="workspace-preview.csv" href="data:text/csv;charset=utf-8;base64,'
            f'{encoded}">Save preview CSV</a>',
            height=30,
        )
    if access.can_manage:
        def request_deletion() -> None:
            st.session_state[pending_key] = selection

        st.button("Delete dataset", key=f"{prefix}request_{selection.hex}", on_click=request_deletion)
        if st.session_state.get(pending_key) == selection:
            st.warning("Delete this dataset and its rows?")
            if st.button("Cancel deletion", key=f"{prefix}cancel_{selection.hex}"):
                del st.session_state[pending_key]
                st.rerun()
            if st.button("Confirm deletion", key=f"{prefix}confirm_{selection.hex}"):
                service.delete(selection)
                del st.session_state[pending_key]
                st.rerun()
