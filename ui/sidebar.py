"""Sidebar: auth toggle, dataset management, session controls."""
import json
import uuid
import streamlit as st
import pandas as pd

from db.postgres import (
    list_datasets_from_db,
    delete_dataset,
    delete_guest_session_datasets,
    cleanup_old_guest_datasets,
)
from core.dataset_profiler import read_uploaded_file, ingest_dataframe
from core.rate_limiter import (
    get_guest_limits,
    record_guest_upload,
    validate_upload_file,
    GUEST_UPLOAD_MAX_BYTES,
    GUEST_UPLOAD_LIMIT,
)
from core.session_manager import save_session, auto_save_if_nonempty


# ── Session state bootstrap ────────────────────────────────────────────────────

def init_session_state() -> None:
    defaults = {
        "is_owner": False,
        "chat_history": [],
        "lc_history": [],
        "active_dataset": None,
        "guest_session_id": str(uuid.uuid4()),
        "session_label": "",
        "_owner_pw_attempt": "",
        "_auth_mode": "guest",      # "owner" | "guest"
        "_login_error": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Public helpers used by chat.py ─────────────────────────────────────────────

def is_owner() -> bool:
    return st.session_state.get("is_owner", False)


def get_session_id() -> str:
    return st.session_state.get("guest_session_id", "")


def can_query() -> bool:
    if is_owner():
        return True
    try:
        return get_guest_limits()["can_query"]
    except Exception:
        return st.session_state.get("_fallback_queries", 0) < 10


def record_query() -> None:
    if is_owner():
        return
    try:
        from core.rate_limiter import record_guest_query
        record_guest_query()
    except Exception:
        st.session_state["_fallback_queries"] = st.session_state.get("_fallback_queries", 0) + 1
    invalidate_limits_cache()


# ── Internal auth ──────────────────────────────────────────────────────────────

def _try_owner_login(pw: str) -> bool:
    try:
        expected = st.secrets.get("owner_password", "")
    except Exception:
        expected = ""
    return bool(pw and pw == expected)


def _render_auth_toggle() -> None:
    """Owner / Guest toggle + login form."""
    mode = st.session_state.get("_auth_mode", "guest")

    col_owner, col_guest = st.sidebar.columns(2)
    owner_style = "primary" if mode == "owner" else "secondary"
    guest_style = "primary" if mode == "guest" else "secondary"

    if col_owner.button("👑 Owner", use_container_width=True, type=owner_style, key="btn_owner_mode"):
        st.session_state["_auth_mode"] = "owner"
        st.session_state["_login_error"] = ""
        st.rerun()

    if col_guest.button("👤 Guest", use_container_width=True, type=guest_style, key="btn_guest_mode"):
        if st.session_state.get("is_owner"):
            # Auto-save before switching away
            auto_save_if_nonempty(st.session_state.get("chat_history", []))
        st.session_state["_auth_mode"] = "guest"
        st.session_state["is_owner"] = False
        st.rerun()

    if mode == "owner" and not st.session_state.get("is_owner"):
        st.sidebar.markdown('<div class="login-card">', unsafe_allow_html=True)
        pw = st.sidebar.text_input(
            "Owner password",
            type="password",
            key="owner_pw_field",
            placeholder="Enter password…",
            label_visibility="collapsed",
        )
        if st.sidebar.button("Unlock  →", use_container_width=True, key="btn_login", type="primary"):
            if _try_owner_login(pw):
                st.session_state["is_owner"] = True
                st.session_state["_login_error"] = ""
                st.rerun()
            else:
                st.session_state["_login_error"] = "Incorrect password."
        if st.session_state.get("_login_error"):
            st.sidebar.error(st.session_state["_login_error"])
        st.sidebar.markdown("</div>", unsafe_allow_html=True)


def _get_cached_limits() -> dict:
    """Fetch guest limits once per session; invalidated by record_query/record_upload."""
    if "guest_session_id" not in st.session_state:
        return {"queries_remaining": 10, "uploads_remaining": 5, "can_query": True, "can_upload": True}
    if "_limits_cache" not in st.session_state:
        try:
            st.session_state["_limits_cache"] = get_guest_limits()
        except Exception:
            st.session_state["_limits_cache"] = {
                "queries_remaining": 10, "uploads_remaining": 5,
                "can_query": True, "can_upload": True,
            }
    return st.session_state["_limits_cache"]


def invalidate_limits_cache() -> None:
    st.session_state.pop("_limits_cache", None)


def _render_role_badge() -> None:
    if is_owner():
        st.sidebar.markdown(
            '<div class="role-badge owner-badge">👑 Owner · Unlimited queries</div>',
            unsafe_allow_html=True,
        )
        return

    limits = _get_cached_limits()
    q_rem = limits.get("queries_remaining", 10)
    color = "#16a34a" if q_rem > 4 else ("#d97706" if q_rem > 1 else "#dc2626")
    st.sidebar.markdown(
        f'<div class="role-badge guest-badge" style="border-color:{color}">'
        f'👤 Guest · {q_rem}/10 queries left today · 5 uploads · 10 MB each'
        f'</div>',
        unsafe_allow_html=True,
    )
    if q_rem == 0:
        st.sidebar.warning("Daily query limit reached. Resets at midnight UTC.")


# ── Session controls (owner only) ──────────────────────────────────────────────

def _render_session_controls() -> None:
    if not is_owner():
        return

    st.sidebar.markdown("#### 🗂 Session")
    label_input = st.sidebar.text_input(
        "Session label (optional)",
        value=st.session_state.get("session_label", ""),
        key="session_label_input",
        placeholder="e.g. Q1 Sales Review",
        label_visibility="collapsed",
    )
    st.session_state["session_label"] = label_input

    c1, c2, c3 = st.sidebar.columns(3)

    if c1.button("💾 Save", use_container_width=True, key="btn_save_session"):
        history = st.session_state.get("chat_history", [])
        if history:
            path = save_session(history, label=label_input)
            st.sidebar.success(f"Saved!")
        else:
            st.sidebar.info("Nothing to save yet.")

    if c2.button("✨ New", use_container_width=True, key="btn_new_session"):
        auto_save_if_nonempty(st.session_state.get("chat_history", []))
        st.session_state["chat_history"] = []
        st.session_state["lc_history"] = []
        st.session_state["session_label"] = ""
        st.rerun()

    if c3.button("🗑 Clear", use_container_width=True, key="btn_clear_session"):
        st.session_state["chat_history"] = []
        st.session_state["lc_history"] = []
        st.rerun()

    st.sidebar.markdown("---")


# ── Upload ─────────────────────────────────────────────────────────────────────

def _render_upload() -> None:
    owner = is_owner()
    sid = get_session_id()

    st.sidebar.markdown("#### 📁 Upload Dataset")

    if not owner:
        # Count active session files (not daily counter — guest can replace files freely)
        try:
            active = list_datasets_from_db(include_owner_only=False, session_id=sid)
            session_files = [d for d in active if not d.get("is_demo") and d.get("session_id") == sid]
            active_count = len(session_files)
        except Exception:
            active_count = 0

        if active_count >= GUEST_UPLOAD_LIMIT:
            st.sidebar.caption(
                f"Session file limit reached ({GUEST_UPLOAD_LIMIT} files). "
                "Remove a dataset below to upload a new one."
            )
            st.sidebar.markdown("---")
            return
        st.sidebar.caption(
            f"{active_count}/{GUEST_UPLOAD_LIMIT} session files · 10 MB each"
        )

    uploaded = st.sidebar.file_uploader(
        "CSV or Excel",
        type=["csv", "xlsx", "xls"],
        key="file_uploader",
        label_visibility="collapsed",
    )

    if uploaded:
        ok, err = validate_upload_file(uploaded)
        if not ok:
            st.sidebar.error(err)
            return

        file_size = uploaded.size if hasattr(uploaded, "size") else len(uploaded.getvalue())

        with st.sidebar.status(f"Ingesting {uploaded.name}…"):
            try:
                df = read_uploaded_file(uploaded)
                ingest_kwargs = {"owner_only": owner, "session_id": None if owner else sid}
                slug = ingest_dataframe(df, uploaded.name, **ingest_kwargs)
                if not owner:
                    record_guest_upload(file_size)
                    invalidate_limits_cache()
                st.sidebar.success(f"✅ `{slug}` — {len(df):,} rows")
            except Exception as e:
                st.sidebar.error(f"Upload failed: {e}")

    st.sidebar.markdown("---")


# ── Dataset list ───────────────────────────────────────────────────────────────

def _render_datasets() -> None:
    owner = is_owner()
    sid = get_session_id()

    try:
        all_ds = list_datasets_from_db(include_owner_only=owner, session_id=sid)
    except Exception:
        all_ds = []

    demo_ds = [d for d in all_ds if d.get("is_demo")]
    user_ds = [d for d in all_ds if not d.get("is_demo")]

    # ── Session / uploaded datasets first ─────────────────────────────────────
    section_label = "#### 📂 Your Datasets" if owner else "#### 📂 Session Datasets"
    st.sidebar.markdown(section_label)
    if not user_ds:
        st.sidebar.caption("No datasets uploaded yet.")
    for ds in user_ds:
        can_del = owner or ds.get("session_id") == sid
        _dataset_row(ds, allow_delete=can_del)

    st.sidebar.markdown("---")

    # ── Demo datasets ──────────────────────────────────────────────────────────
    st.sidebar.markdown("#### 🗄 Demo Datasets")
    if not demo_ds:
        st.sidebar.caption("No demo datasets loaded.")
    for ds in demo_ds:
        _dataset_row(ds, allow_delete=False)

    # ── Suggested queries (guest) ──────────────────────────────────────────────
    if not owner and demo_ds:
        st.sidebar.markdown("---")
        st.sidebar.markdown("#### 💡 Try These")
        hints = [
            "Total revenue by region?",
            "Show sales trend over time",
            "Detect anomalies in revenue",
            "Correlation between profit and sales?",
            "Which country has highest CO2 emissions?",
        ]
        for h in hints:
            st.sidebar.caption(f"→ *{h}*")


def _dataset_row(ds: dict, allow_delete: bool = False) -> None:
    """Render a dataset as an expandable row with optional inline 🗑 delete."""
    row_count = ds.get("row_count") or 0
    slug = ds["slug"]

    if allow_delete:
        # Two columns: expander label | 🗑 button
        col_exp, col_del = st.sidebar.columns([5, 1])
        with col_exp:
            with st.expander(f"{ds['name']} · {row_count:,} rows", expanded=False):
                _dataset_detail(ds)
        with col_del:
            st.markdown("<div style='margin-top:6px'>", unsafe_allow_html=True)
            if st.button("🗑", key=f"del_ds_{slug}", help=f"Remove {ds['name']}", use_container_width=True):
                try:
                    delete_dataset(slug)
                    invalidate_limits_cache()
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Delete failed: {e}")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        with st.sidebar.expander(f"★ {ds['name']} · {row_count:,} rows", expanded=False):
            _dataset_detail(ds)


def _dataset_detail(ds: dict) -> None:
    st.caption(f"Table: `{ds['slug']}`")
    cols = ds.get("columns_json") or []
    if isinstance(cols, str):
        cols = json.loads(cols)
    if cols:
        col_df = pd.DataFrame(cols)
        st.dataframe(col_df, hide_index=True, use_container_width=True)


# ── Main entry point ───────────────────────────────────────────────────────────

def render_sidebar() -> None:
    init_session_state()

    # Run cleanup on startup (non-blocking: catches errors silently)
    try:
        cleanup_old_guest_datasets(max_age_hours=24)
    except Exception:
        pass

    st.sidebar.markdown(
        '<div class="sidebar-title">📊 Data Analysis Agent</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    _render_auth_toggle()
    st.sidebar.markdown("---")
    _render_role_badge()
    st.sidebar.markdown("---")
    _render_session_controls()
    _render_upload()
    _render_datasets()
