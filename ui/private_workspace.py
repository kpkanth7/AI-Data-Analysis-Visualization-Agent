"""Streamlit flow for a verified identity and server-supplied workspace membership."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from uuid import UUID, uuid4

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from db.workspace_datasets import (
    CATALOG_SEARCH_LIMIT, CATALOG_SEARCH_MAX_CHARS, PREVIEW_ROWS, CleaningPreviewChanged,
    UploadLimitReached, UploadRequestConflict, WorkspaceAccess, WorkspaceAnalysis, WorkspaceCleaningPreview,
    WorkspaceColumnSummary, WorkspaceCSV, WorkspaceDataset, WorkspaceDatasets, WorkspaceRowCount,
    WorkspaceTrimWhitespace,
)
from db.workspace_memberships import WorkspaceMembership
from ui.csv_download import render_csv_link
from ui.service_status import render_service_unavailable


_PREFIX = "private_workspace_"
_BINDING = f"{_PREFIX}binding"
_SESSION = f"{_PREFIX}session"


MembershipLookup = Callable[[str, str], WorkspaceMembership | None]
WorkspaceEngines = Callable[[UUID], tuple[Engine, Engine]]
WorkspaceBinding = tuple[str, str, WorkspaceMembership]


@dataclass(frozen=True)
class CleaningReview:
    preview: WorkspaceCleaningPreview
    request_id: UUID | None = None


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
def _watch_access(
    lookup: MembershipLookup, binding: WorkspaceBinding, service: WorkspaceDatasets, selection_key: str,
) -> None:
    try:
        if _current_binding(lookup) == binding:
            selection = st.session_state.get(selection_key)
            if selection is not None:
                try:
                    service.metadata(selection)
                except PermissionError:
                    st.session_state.pop(selection_key, None)
                    st.rerun()
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
    session_id = st.session_state[_SESSION]
    access = WorkspaceAccess(membership.principal_id, session_id, membership.workspace_id, membership.can_manage)
    prefix = f"{_PREFIX}{session_id.hex}_"

    def resolve_access() -> WorkspaceAccess | None:
        if st.session_state.get(_SESSION) == session_id and _current_binding(lookup) == binding:
            return access
        return None

    content = st.empty()
    try:
        service = WorkspaceDatasets(access, resolve_access, *engines(access.workspace_id))
        _watch_access(lookup, binding, service, f"{prefix}selection")
        with content.container():
            _render_datasets(service, access, prefix)
    except PermissionError:
        _clear_workspace_state()
        content.warning("Your workspace or dataset is no longer available. Refresh to check again.")
    except SQLAlchemyError as exc:
        _clear_workspace_state()
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
        st.warning(
            "Choose a UTF-8 CSV up to 1 MB with distinct, nonempty column names, "
            "up to 1,000 rows and 64 columns, and the same number of fields on each row."
        )
    except SQLAlchemyError:
        st.warning("Upload status is unavailable. Submit the same file again to check or retry.")
    else:
        st.session_state[request_key] = uuid4()
        st.rerun()


def _save_analysis(service: WorkspaceDatasets, result: WorkspaceAnalysis, feedback_key: str) -> None:
    try:
        service.save_analysis(result)
    except PermissionError:
        return
    except SQLAlchemyError:
        st.session_state[feedback_key] = (result, False)
    else:
        st.session_state[feedback_key] = (result, True)


def _render_analysis(
    service: WorkspaceDatasets, dataset: WorkspaceDataset, access: WorkspaceAccess, prefix: str,
) -> None:
    with st.expander("Analyze dataset", expanded=True):
        operation = st.radio("Operation", ("Count rows", "Summarize column"), key=f"{prefix}operation")
        request = WorkspaceRowCount(version_id=dataset.version_id)
        if operation == "Summarize column":
            column = st.selectbox("Column", dataset.columns, key=f"{prefix}column")
            request = WorkspaceColumnSummary(version_id=dataset.version_id, column=column)
        feedback_key = f"{prefix}save_feedback"
        feedback = st.session_state.pop(feedback_key, None)
        result = None
        saved = False
        if feedback is not None and feedback[0].request == request:
            result, saved = feedback
            if saved:
                st.success(f"Saved analysis result {result.result_id}.")
            else:
                st.warning("The result could not be saved. You can retry or download the analysis.")
        if st.button("Run analysis", key=f"{prefix}run"):
            try:
                result = service.summarize(request)
            except ValueError:
                st.warning("This dataset exceeds the analysis row limit.")
                return
        with st.form(f"{prefix}restore"):
            saved_id = st.text_input("Saved result ID", max_chars=36)
            restore = st.form_submit_button("Restore result")
        if restore:
            try:
                result_id = UUID(saved_id.strip())
            except ValueError:
                st.warning("Enter a valid saved result ID.")
                return
            try:
                result = service.restore_analysis(dataset.version_id, result_id)
            except PermissionError:
                st.warning("This saved result is unavailable for the selected dataset version.")
                return
            except SQLAlchemyError:
                st.warning("Saved results are unavailable. Try again.")
                return
            saved = True
        if result is None:
            return
        csv_data = service.analysis_csv(result)
        executed = result.request
        st.caption(f"Analysis scope: Entire dataset. Version: {executed.version_id}")
        if isinstance(executed, WorkspaceColumnSummary):
            st.text(f"Column: {executed.column}")
        labels = {
            "row_count": "Rows", "non_null_count": "Non-null values", "null_count": "Missing values",
            "minimum": "Minimum", "maximum": "Maximum", "total": "Sum", "mean": "Mean",
        }
        for key, value in result.values.items():
            st.text(f"{labels[key]}: {value if value is not None else 'No values'}")
        _render_summary_chart(result)
        st.caption(f"Analysis CSV result: {result.result_id}")
        render_csv_link(csv_data, "Analysis CSV", "workspace-analysis.csv")
        if access.can_manage and not saved:
            st.button(
                "Save result", key=f"{prefix}save_{result.result_id}", on_click=_save_analysis,
                args=(service, result, feedback_key),
            )


def _render_summary_chart(result: WorkspaceAnalysis) -> None:
    values = [result.values.get(name) for name in ("minimum", "mean", "maximum")]
    if any(value is None for value in values):
        return
    coordinates = [float(value) for value in values]
    if (
        not all(isfinite(value) for value in coordinates)
        or any(value != 0 and coordinate == 0 for value, coordinate in zip(values, coordinates, strict=True))
        or any(isinstance(value, int) and int(coordinate) != value
               for value, coordinate in zip(values, coordinates, strict=True))
        or len(set(coordinates)) < len(set(values))
    ):
        st.caption("A chart is unavailable for these values. The numeric summary is shown above.")
        return
    figure = go.Figure(go.Bar(
        x=["Minimum", "Mean", "Maximum"],
        y=coordinates,
        customdata=[str(value) for value in values],
        hovertemplate="%{x}: %{customdata}<extra></extra>",
    ))
    figure.update_layout(
        meta={"result_id": str(result.result_id), "version_id": str(result.request.version_id)},
        yaxis_rangemode="tozero",
    )
    st.plotly_chart(figure, key=f"workspace_summary_{result.result_id}")


def _render_cleaning_preview(
    service: WorkspaceDatasets, dataset: WorkspaceDataset, access: WorkspaceAccess, prefix: str,
) -> None:
    preview_key = f"{prefix}cleaning_preview"
    widget_prefix = f"{prefix}cleaning_{dataset.version_id.hex}_"
    with st.expander("Preview cleaning"):
        column = st.selectbox("Column to trim", dataset.columns, key=f"{widget_prefix}column")
        plan = WorkspaceTrimWhitespace(version_id=dataset.version_id, column=column)
        review = st.session_state.get(preview_key)
        if review is not None and review.preview.plan != plan:
            st.session_state.pop(preview_key, None)
            review = None
        if st.button("Preview whitespace trim", key=f"{widget_prefix}run"):
            st.session_state.pop(preview_key, None)
            try:
                result = service.preview_cleaning(plan)
            except ValueError:
                st.warning("This dataset exceeds the cleaning preview row limit.")
                return
            review = CleaningReview(result)
            st.session_state[preview_key] = review
        if review is None:
            return
        preview = st.empty()
        replay = st.empty()
        if access.can_manage and review.request_id is None and replay.button(
            "Approve and verify replay", key=f"{widget_prefix}replay",
        ):
            try:
                result = service.replay_cleaning(review.preview)
            except CleaningPreviewChanged as exc:
                st.session_state.pop(preview_key, None)
                replay.empty()
                st.warning(str(exc))
                return
            except ValueError:
                st.session_state.pop(preview_key, None)
                replay.empty()
                st.warning("This dataset exceeds the cleaning preview row limit.")
                return
            review = CleaningReview(result, uuid4())
            st.session_state[preview_key] = review
            replay.empty()
            st.success("Replay matched the reviewed preview. No dataset was changed.")
        if access.can_manage and review.request_id is not None:
            publish = st.empty()
            if publish.button(
                "Publish cleaned version", key=f"{widget_prefix}publish",
                help="Creates a separate version and counts toward the workspace’s daily upload limit.",
            ):
                try:
                    cleaned = service.publish_cleaning(review.preview, review.request_id)
                except UploadLimitReached:
                    st.warning("This workspace has reached its daily upload limit. Try again after midnight UTC.")
                except (CleaningPreviewChanged, UploadRequestConflict) as exc:
                    st.session_state.pop(preview_key, None)
                    publish.empty()
                    st.warning(str(exc))
                    return
                except ValueError:
                    st.session_state.pop(preview_key, None)
                    publish.empty()
                    st.warning("This dataset exceeds the cleaning preview row limit.")
                    return
                except SQLAlchemyError:
                    st.warning("Publication status is unavailable. Select Publish cleaned version again to retry.")
                else:
                    st.session_state.pop(preview_key, None)
                    st.session_state[f"{prefix}published_cleaning"] = cleaned.version_id
                    publish.empty()
                    st.rerun()
        result = review.preview
        with preview.container():
            st.caption(
                f"Whitespace trim preview for version {result.plan.version_id}. "
                f"{result.changed_count:,} of {result.row_count:,} rows would change."
            )
            st.text(f"Column: {result.plan.column}")
            if len(result.rows) < result.row_count:
                st.caption(f"Showing the first {len(result.rows)} before and after values.")
            values = [tuple(repr(value) for value in row) for row in result.rows]
            st.html(pd.DataFrame(values, columns=["Before", "After"]).to_html(index=False, escape=True))


def _render_csv_download(export: WorkspaceCSV, label: str, filename: str, status: str) -> None:
    st.caption(f"{label}: {export.row_count} data rows ({status}). Version: {export.version_id}")
    render_csv_link(export.data, label, filename)


def _render_datasets(service: WorkspaceDatasets, access: WorkspaceAccess, prefix: str) -> None:
    if access.can_manage:
        _render_upload(service, prefix)

    query = st.text_input(
        "Search by filename or column", key=f"{prefix}search", max_chars=CATALOG_SEARCH_MAX_CHARS,
    )
    preview_key = f"{prefix}cleaning_preview"
    try:
        datasets = {dataset.version_id: dataset for dataset in service.list_datasets(query)}
    except ValueError:
        st.session_state.pop(preview_key, None)
        st.warning("Enter a valid search term.")
        return

    selection_key = f"{prefix}selection"
    pending_key = f"{prefix}delete"
    published = st.session_state.pop(f"{prefix}published_cleaning", None)
    if published in datasets:
        st.session_state[selection_key] = published
        st.success(f"Published cleaned version {published}.")
    elif published is not None:
        st.success("Published a cleaned version. Clear the search to refresh the list.")
    if st.session_state.get(selection_key) not in datasets:
        st.session_state.pop(selection_key, None)
        st.session_state.pop(pending_key, None)
        st.session_state.pop(preview_key, None)
    if not datasets:
        st.info("No matching datasets." if query.strip() else "This workspace has no datasets yet.")
        return
    if query.strip() and len(datasets) == CATALOG_SEARCH_LIMIT:
        st.caption(f"Showing the first {CATALOG_SEARCH_LIMIT} matches.")
    selection = st.selectbox(
        "Dataset", tuple(datasets), index=None,
        format_func=lambda version_id: f"{datasets[version_id].filename} ({version_id})",
        key=selection_key,
    )
    if st.session_state.get(pending_key) != selection:
        st.session_state.pop(pending_key, None)
    if selection is None:
        st.session_state.pop(preview_key, None)
        return

    dataset = datasets[selection]
    st.caption(f"Dataset version: {dataset.version_id}")
    unit = "row" if dataset.row_count == 1 else "rows"
    st.caption(f"{dataset.row_count:,} {unit}")
    if dataset.row_count > PREVIEW_ROWS:
        st.caption(f"Showing the first {PREVIEW_ROWS} rows.")
    rows = service.preview(selection)
    st.html(pd.DataFrame(rows, columns=dataset.columns).to_html(index=False, escape=True))
    _render_analysis(service, dataset, access, f"{prefix}analysis_{selection.hex}_")
    _render_cleaning_preview(service, dataset, access, prefix)
    if st.button("Prepare preview CSV", key=f"{prefix}csv_{selection.hex}"):
        preview = service.preview_csv(selection)
        status = "truncated" if preview.truncated else "not truncated"
        _render_csv_download(preview, "Preview CSV", "workspace-preview.csv", status)
    if st.button("Prepare full dataset CSV", key=f"{prefix}full_csv_{selection.hex}"):
        try:
            export = service.full_csv(selection)
        except ValueError:
            st.warning("This dataset exceeds the full CSV row limit.")
        else:
            _render_csv_download(export, "Full dataset CSV", "workspace-dataset.csv", "complete")
    if access.can_manage:
        def request_deletion() -> None:
            st.session_state[pending_key] = selection

        st.button("Delete dataset", key=f"{prefix}request_{selection.hex}", on_click=request_deletion)
        if st.session_state.get(pending_key) == selection:
            st.warning("Delete this dataset version? This permanently removes its rows and saved analysis results.")
            if st.button("Cancel deletion", key=f"{prefix}cancel_{selection.hex}"):
                del st.session_state[pending_key]
                st.rerun()
            if st.button("Confirm deletion", key=f"{prefix}confirm_{selection.hex}"):
                service.delete(selection)
                del st.session_state[pending_key]
                st.rerun()
