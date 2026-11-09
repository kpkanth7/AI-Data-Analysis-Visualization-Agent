"""Shared owner sign-in form for the header and History entry points."""
import streamlit as st

from ui.auth import switch_to_owner


def _submit_login(password_key: str) -> None:
    password = st.session_state.get(password_key, "")
    # Callbacks run before the next render, so the password widget can be cleared.
    st.session_state[password_key] = ""
    switch_to_owner(password)


def render_owner_login(key_prefix: str) -> None:
    password_key = f"{key_prefix}_owner_pw"
    with st.form(f"{key_prefix}_login_form"):
        st.text_input("Owner password", type="password", key=password_key)
        st.form_submit_button(
            "Sign in", type="primary",
            on_click=_submit_login, args=(password_key,),
        )
    if st.session_state.get("_login_error"):
        st.error(st.session_state["_login_error"])
