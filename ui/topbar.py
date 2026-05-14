"""Top bar: title, countdown chip, avatar popover with auth switcher."""
import datetime
import streamlit as st

from ui.auth import (
    init_session_state,
    is_owner,
    get_cached_limits,
    switch_to_owner,
    switch_to_guest,
)
from db.postgres import cleanup_old_guest_datasets


def _utc_today() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d")


def _countdown_chip() -> str:
    if is_owner():
        return '<span class="tb-chip tb-chip-owner">👑 Owner · Unlimited</span>'
    limits = get_cached_limits()
    q = limits.get("queries_remaining", 10)
    tone = "good" if q > 4 else ("warn" if q > 1 else "bad")
    return (
        f'<span class="tb-chip tb-chip-{tone}">👤 Guest · '
        f'{q}/10 queries · Max 5 uploads</span>'
    )


def _render_avatar_popover() -> None:
    owner = is_owner()
    initial = "O" if owner else "G"
    label = f"  {initial}  "
    with st.popover(label, use_container_width=False):
        if owner:
            st.markdown("**👑 Owner**")
            limits_caption = "Unlimited queries · cleaning enabled · history visible"
            st.caption(limits_caption)
            st.markdown("---")
            if st.button("Switch to Guest", key="pop_to_guest", use_container_width=True):
                switch_to_guest()
                st.rerun()
        else:
            limits = get_cached_limits()
            st.markdown("**👤 Guest**")
            st.caption(
                f"{limits.get('queries_remaining', 10)}/10 queries left · "
                f"{limits.get('uploads_remaining', 5)}/5 uploads · 10 MB each"
            )
            st.markdown("---")
            st.markdown("**Switch to Owner**")
            pw = st.text_input(
                "Owner password",
                type="password",
                key="topbar_owner_pw",
                placeholder="Enter password…",
                label_visibility="collapsed",
            )
            if st.button("Unlock →", key="topbar_unlock", type="primary", use_container_width=True):
                if switch_to_owner(pw):
                    st.rerun()
            if st.session_state.get("_login_error"):
                st.error(st.session_state["_login_error"])


def render_topbar() -> None:
    init_session_state()

    # Run cleanup once per UTC day: purges guest datasets > 24h old.
    today = _utc_today()
    if st.session_state.get("_cleanup_day") != today:
        try:
            cleanup_old_guest_datasets(max_age_hours=24)
        except Exception:
            pass
        st.session_state["_cleanup_day"] = today

    title_col, chip_col, avatar_col = st.columns([5, 4, 1])

    with title_col:
        st.markdown(
            '<div class="tb-title">📊 Data Analysis Agent</div>',
            unsafe_allow_html=True,
        )

    with chip_col:
        st.markdown(
            f'<div class="tb-chip-wrap">{_countdown_chip()}</div>',
            unsafe_allow_html=True,
        )

    with avatar_col:
        _render_avatar_popover()
