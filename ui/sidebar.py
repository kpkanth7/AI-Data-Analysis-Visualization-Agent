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


def _render_role_badge() -> None:
    if is_owner():
        st.sidebar.markdown(
            '<div class="role-badge owner-badge">👑 Owner · Unlimited queries</div>',
            unsafe_allow_html=True,
        )
        return

    try:
        limits = get_guest_limits()
        q_rem = limits["queries_remaining"]
        u_rem = limits["uploads_remaining"]
        color = "#16a34a" if q_rem > 4 else ("#d97706" if q_rem > 1 else "#dc2626")
        st.sidebar.markdown(
            f'<div class="role-badge guest-badge" style="border-color:{color}">'
            f'👤 Guest · {q_rem}/10 queries left today · {u_rem}/5 uploads left'
            f'</div>',
            unsafe_allow_html=True,
        )
        if q_rem == 0:
            st.sidebar.warning("Daily query limit reached. Resets at midnight UTC.")
    except Exception:
        st.sidebar.info("👤 Guest mode")


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
        try:
            limits = get_guest_limits()
            if not limits["can_upload"]:
                st.sidebar.caption(
                    f"Upload limit reached ({GUEST_UPLOAD_LIMIT} files/day). Resets at midnight UTC."
                )
                return
            bytes_rem_mb = limits["upload_bytes_remaining"] / 1024 / 1024
            st.sidebar.caption(
                f"Up to {GUEST_UPLOAD_LIMIT} files · 10 MB each · {bytes_rem_mb:.0f} MB total remaining today"
            )
        except Exception:
            pass

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

        if not owner:
            try:
                limits = get_guest_limits()
                if file_size + limits["upload_bytes_used"] > 50 * 1024 * 1024:
                    st.sidebar.error("Daily upload byte quota exceeded.")
                    return
            except Exception:
                pass

        with st.sidebar.status(f"Ingesting {uploaded.name}…"):
            try:
                df = read_uploaded_file(uploaded)
                ingest_kwargs = {"owner_only": owner, "session_id": None if owner else sid}
                slug = ingest_dataframe(df, uploaded.name, **ingest_kwargs)
                if not owner:
                    record_guest_upload(file_size)
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

    # ── Demo datasets ──────────────────────────────────────────────────────────
    st.sidebar.markdown("#### 🗄 Demo Datasets")
    if not demo_ds:
        st.sidebar.caption("No demo datasets loaded.")
    for ds in demo_ds:
        _dataset_expander(ds, allow_delete=False)

    # ── User datasets ──────────────────────────────────────────────────────────
    if user_ds or owner:
        label = "#### 📂 Your Datasets" if owner else "#### 📂 Session Datasets"
        st.sidebar.markdown(label)
        if not user_ds:
            st.sidebar.caption("No datasets uploaded yet.")
        for ds in user_ds:
            _dataset_expander(ds, allow_delete=owner or ds.get("session_id") == sid)

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


def _dataset_expander(ds: dict, allow_delete: bool = False) -> None:
    is_demo = ds.get("is_demo", False)
    icon = "★ " if is_demo else ""
    row_count = ds.get("row_count") or 0
    with st.sidebar.expander(f"{icon}{ds['name']} · {row_count:,} rows"):
        st.caption(f"Table: `{ds['slug']}`")
        cols = ds.get("columns_json") or []
        if isinstance(cols, str):
            cols = json.loads(cols)
        if cols:
            col_df = pd.DataFrame(cols)
            st.dataframe(col_df, hide_index=True, use_container_width=True)
        if allow_delete:
            if st.button("🗑 Remove dataset", key=f"del_ds_{ds['slug']}", type="secondary"):
                try:
                    delete_dataset(ds["slug"])
                    st.success(f"Removed `{ds['slug']}`")
                    st.rerun()
                except Exception as e:
                    st.error(f"Delete failed: {e}")


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
