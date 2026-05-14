"""Data tab: demo datasets, your datasets, upload + clean wizard."""
import json
import pandas as pd
import streamlit as st

from db.postgres import list_datasets_from_db, delete_dataset
from core.dataset_profiler import read_uploaded_file, ingest_dataframe
from core.rate_limiter import (
    record_guest_upload,
    validate_upload_file,
    GUEST_UPLOAD_LIMIT,
)
from ui.auth import is_owner, get_session_id, invalidate_limits_cache
from ui.cleaning_wizard_ui import (
    render_cleaning_wizard,
    start_wizard,
    cancel_wizard,
)


def _delete_with_confirm(slug: str, name: str, key_prefix: str) -> None:
    confirm_key = f"_del_confirm_{key_prefix}_{slug}"
    if st.session_state.get(confirm_key):
        if st.button("Confirm delete", key=f"{key_prefix}_confirm_{slug}", type="primary"):
            try:
                delete_dataset(slug)
                invalidate_limits_cache()
                st.session_state.pop(confirm_key, None)
                st.rerun()
            except Exception as e:
                st.error(f"Delete failed: {e}")
        if st.button("Cancel", key=f"{key_prefix}_cancelbtn_{slug}"):
            st.session_state.pop(confirm_key, None)
            st.rerun()
    else:
        if st.button("🗑", key=f"{key_prefix}_btn_{slug}", help=f"Remove {name}"):
            st.session_state[confirm_key] = True
            st.rerun()


def _dataset_card(ds: dict, allow_delete: bool, key_prefix: str) -> None:
    cols_raw = ds.get("columns_json") or []
    if isinstance(cols_raw, str):
        cols_raw = json.loads(cols_raw)
    row_count = ds.get("row_count") or 0

    head_col, btn_col = st.columns([6, 1])
    with head_col:
        badge = "★ " if ds.get("is_demo") else ""
        with st.expander(f"{badge}**{ds['name']}** · {row_count:,} rows · `{ds['slug']}`", expanded=False):
            if cols_raw:
                st.dataframe(pd.DataFrame(cols_raw), hide_index=True, use_container_width=True)
    if allow_delete:
        with btn_col:
            _delete_with_confirm(ds["slug"], ds["name"], key_prefix)


def _section_header(label: str) -> None:
    st.markdown(
        f'<div class="section-head"><span class="dot"></span><h4>{label}</h4></div>',
        unsafe_allow_html=True,
    )


def _render_demo_section(demo_ds: list[dict]) -> None:
    _section_header("🗄 Demo Datasets")
    if not demo_ds:
        st.caption("No demo datasets loaded.")
        return
    st.caption("Test the product with these — head over to **Explorer** to browse rows, "
               "or to **Chat** to ask questions.")
    for ds in demo_ds:
        _dataset_card(ds, allow_delete=False, key_prefix="demo")


def _render_user_section(user_ds: list[dict], owner: bool, sid: str) -> None:
    _section_header("📂 Your Datasets" if owner else "📂 Session Datasets")
    if not user_ds:
        st.caption("No datasets uploaded yet. Drop a file below to get started.")
        return
    for ds in user_ds:
        can_del = owner or ds.get("session_id") == sid
        _dataset_card(ds, allow_delete=can_del, key_prefix="user")
    if not owner:
        st.caption("⚠️ Removing a dataset wipes its rows and vector embeddings immediately.")


def _render_upload_section(owner: bool, sid: str, active_count: int) -> None:
    _section_header("⬆️ Upload Dataset")

    if not owner:
        if active_count >= GUEST_UPLOAD_LIMIT:
            st.warning(
                f"Session file limit reached ({GUEST_UPLOAD_LIMIT} files). "
                "Remove one above to upload a new one."
            )
            return
        st.caption(
            f"{active_count}/{GUEST_UPLOAD_LIMIT} session files · "
            f"10 MB max per file · CSV or Excel"
        )
    else:
        st.caption("Upload directly, or use **Clean & Upload** for a guided cleaning wizard.")

    uploaded = st.file_uploader(
        "CSV or Excel",
        type=["csv", "xlsx", "xls"],
        key="data_uploader",
        label_visibility="collapsed",
    )

    if not uploaded:
        return

    ok, err = validate_upload_file(uploaded)
    if not ok:
        st.error(err)
        return

    file_size = uploaded.size if hasattr(uploaded, "size") else len(uploaded.getvalue())

    b1, b2 = st.columns(2)
    with b1:
        if st.button("⬆️ Upload", key="btn_plain_upload", type="primary", use_container_width=True):
            with st.status(f"Ingesting {uploaded.name}…"):
                try:
                    df = read_uploaded_file(uploaded)
                    slug = ingest_dataframe(
                        df, uploaded.name,
                        owner_only=owner,
                        session_id=None if owner else sid,
                    )
                    if not owner:
                        record_guest_upload(file_size)
                        invalidate_limits_cache()
                    st.success(f"✅ `{slug}` — {len(df):,} rows")
                    st.rerun()
                except Exception as e:
                    st.error(f"Upload failed: {e}")

    with b2:
        if owner:
            if st.button("🧹 Clean & Upload", key="btn_clean_upload", use_container_width=True):
                try:
                    start_wizard(uploaded)
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not start wizard: {e}")
        else:
            st.button(
                "🔒 Clean & Upload",
                key="btn_clean_locked",
                disabled=True,
                use_container_width=True,
                help="Owner only — sign in to clean datasets",
            )
            st.caption(
                "🔒 *Sign in as owner to unlock guided cleaning — duplicates, nulls, "
                "type detection, and outlier handling, all in a few clicks.*"
            )


def render_data_tab() -> None:
    owner = is_owner()
    sid = get_session_id()

    # If wizard active, take over the tab
    if st.session_state.get("_cleaning"):
        render_cleaning_wizard()
        return

    try:
        all_ds = list_datasets_from_db(include_owner_only=owner, session_id=sid)
    except Exception as e:
        st.error(f"Could not load datasets: {e}")
        return

    demo_ds = [d for d in all_ds if d.get("is_demo")]
    user_ds = [d for d in all_ds if not d.get("is_demo")]
    active_count = len([d for d in user_ds if d.get("session_id") == sid]) if not owner else len(user_ds)

    # Hero
    st.markdown(
        '<div class="hero-card">'
        '<div class="hero-eyebrow">Data workspace</div>'
        '<div class="hero-title">Bring your data — or play with <em>demos</em>.</div>'
        '<div class="hero-body">Upload a CSV / Excel file, explore demo datasets, '
        'and head to Chat for natural-language analysis with charts, SQL, and PDF export.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    _render_user_section(user_ds, owner, sid)
    st.markdown("---")
    _render_demo_section(demo_ds)
    st.markdown("---")
    _render_upload_section(owner, sid, active_count)
