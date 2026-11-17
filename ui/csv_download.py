"""Deliver CSV bytes inline to the current Streamlit session."""

from base64 import b64encode
from html import escape

import streamlit as st


def render_csv_link(data: bytes, label: str, filename: str) -> None:
    markup = (
        "<style>"
        "body { margin: 4px; font: 14px/20px system-ui, sans-serif; }"
        "a { display: inline-block; padding: 3px 7px; color: #eff6ff; background: #1e3a5f; }"
        "a:focus-visible { outline: 2px solid #60a5fa; outline-offset: 1px; }"
        "</style>"
        f'<a download="{escape(filename)}" href="data:text/csv;charset=utf-8;base64,'
        f'{b64encode(data).decode("ascii")}">Save {escape(label)}</a>'
    )
    if hasattr(st, "iframe"):
        st.iframe(markup, height=36, alt=f"{label} download")
    else:
        from streamlit.components.v1 import html as iframe_html

        iframe_html(markup, height=36)
