"""IP-hash based daily rate limiting for guest users on Streamlit Cloud."""
import hashlib

import streamlit as st

from db.postgres import get_guest_usage, increment_guest_query, increment_guest_upload

GUEST_QUERY_LIMIT = 10
GUEST_UPLOAD_LIMIT = 5
GUEST_UPLOAD_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per file
GUEST_UPLOAD_TOTAL_BYTES = 50 * 1024 * 1024  # 50 MB total per day


def get_client_ip() -> str:
    """Extract real client IP from Streamlit Cloud forwarded headers."""
    try:
        headers = st.context.headers
        forwarded = headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return headers.get("X-Real-Ip", headers.get("Remote-Addr", "unknown"))
    except Exception:
        return "unknown"


def get_ip_hash() -> str:
    """Return a stable daily hash of the client IP (stored in session_state for the session)."""
    if "ip_hash" not in st.session_state:
        ip = get_client_ip()
        # Salt with a constant to prevent rainbow-table enumeration
        raw = f"daa-salt-2024:{ip}"
        st.session_state["ip_hash"] = hashlib.sha256(raw.encode()).hexdigest()[:32]
    return st.session_state["ip_hash"]


def get_guest_limits() -> dict:
    """Return current usage + remaining for the calling guest."""
    ip_hash = get_ip_hash()
    usage = get_guest_usage(ip_hash)
    return {
        "queries_used": usage["queries"],
        "queries_remaining": max(0, GUEST_QUERY_LIMIT - usage["queries"]),
        "uploads_used": usage["uploads"],
        "uploads_remaining": max(0, GUEST_UPLOAD_LIMIT - usage["uploads"]),
        "upload_bytes_used": usage["upload_bytes"],
        "upload_bytes_remaining": max(0, GUEST_UPLOAD_TOTAL_BYTES - usage["upload_bytes"]),
        "can_query": usage["queries"] < GUEST_QUERY_LIMIT,
        "can_upload": (
            usage["uploads"] < GUEST_UPLOAD_LIMIT
            and usage["upload_bytes"] < GUEST_UPLOAD_TOTAL_BYTES
        ),
    }


def record_guest_query() -> None:
    increment_guest_query(get_ip_hash())


def record_guest_upload(file_bytes: int) -> None:
    increment_guest_upload(get_ip_hash(), file_bytes)


def validate_upload_file(file) -> tuple[bool, str]:
    """Return (ok, error_message). file is a Streamlit UploadedFile."""
    if file is None:
        return False, "No file provided."
    size = file.size if hasattr(file, "size") else len(file.getvalue())
    if size > GUEST_UPLOAD_MAX_BYTES:
        mb = size / 1024 / 1024
        return False, f"File too large ({mb:.1f} MB). Max 10 MB per file."
    # Validate MIME type via extension (Streamlit already enforces type= list)
    name = file.name.lower()
    if not (name.endswith(".csv") or name.endswith(".xlsx") or name.endswith(".xls")):
        return False, "Only CSV and Excel files are accepted."
    return True, ""
