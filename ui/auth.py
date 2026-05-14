"""Auth + session helpers. Single import surface for is_owner / get_session_id / limits."""
import streamlit as st

from core.rate_limiter import get_guest_limits, get_browser_fingerprint
from core.session_manager import auto_save_if_nonempty


def _new_guest_session_id() -> str:
    return get_browser_fingerprint()


def init_session_state() -> None:
    if "guest_session_id" not in st.session_state:
        st.session_state["guest_session_id"] = _new_guest_session_id()

    defaults = {
        "is_owner": False,
        "chat_history": [],
        "lc_history": [],
        "active_dataset": None,
        "session_label": "",
        "_owner_pw_attempt": "",
        "_auth_mode": "guest",
        "_login_error": "",
        "chart_store": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def is_owner() -> bool:
    return st.session_state.get("is_owner", False)


def get_session_id() -> str:
    return st.session_state.get("guest_session_id", "")


def can_query() -> bool:
    if is_owner():
        return True
    try:
        return get_cached_limits()["can_query"]
    except Exception:
        return st.session_state.get("_fallback_queries", 0) < 10


def can_upload() -> bool:
    if is_owner():
        return True
    try:
        return get_cached_limits()["can_upload"]
    except Exception:
        return True


def record_query() -> None:
    if is_owner():
        return
    try:
        from core.rate_limiter import record_guest_query
        record_guest_query()
    except Exception:
        st.session_state["_fallback_queries"] = st.session_state.get("_fallback_queries", 0) + 1
    invalidate_limits_cache()


def get_cached_limits() -> dict:
    """Always-fresh DB read. No session-state caching — external resets propagate immediately."""
    if "guest_session_id" not in st.session_state:
        return {"queries_remaining": 10, "uploads_remaining": 5, "can_query": True, "can_upload": True}
    try:
        return get_guest_limits()
    except Exception:
        return {
            "queries_remaining": 10, "uploads_remaining": 5,
            "can_query": True, "can_upload": True,
        }


def invalidate_limits_cache() -> None:
    pass


def try_owner_login(pw: str) -> bool:
    try:
        expected = st.secrets.get("owner_password", "")
    except Exception:
        expected = ""
    return bool(pw and pw == expected)


def switch_to_owner(pw: str) -> bool:
    """Password gate. Returns True on success. Preserves chat + datasets."""
    if try_owner_login(pw):
        st.session_state["is_owner"] = True
        st.session_state["_auth_mode"] = "owner"
        st.session_state["_login_error"] = ""
        return True
    st.session_state["_login_error"] = "Incorrect password."
    return False


def switch_to_guest() -> None:
    """One-click drop owner. Preserves chat + datasets."""
    if st.session_state.get("is_owner"):
        auto_save_if_nonempty(st.session_state.get("chat_history", []))
    st.session_state["is_owner"] = False
    st.session_state["_auth_mode"] = "guest"
    st.session_state["_login_error"] = ""


def reset_chat_state() -> None:
    st.session_state["chat_history"] = []
    st.session_state["lc_history"] = []
    st.session_state["chart_store"] = {}
    st.session_state.pop("_last_sub_questions", None)
    for k in list(st.session_state.keys()):
        if k.startswith("_pdf_cache_"):
            del st.session_state[k]
