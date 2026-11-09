"""Consistent recovery messages without exposing connection details to visitors."""
import logging

import streamlit as st

logger = logging.getLogger(__name__)


def render_service_unavailable(service: str, key: str, error: Exception | None = None) -> None:
    if error is not None:
        # Driver errors can contain credentials, SQL, or private table names.
        logger.warning("%s service unavailable: %s", service, type(error).__name__)
    if service == "Usage":
        st.warning("Usage could not be checked. New AI requests and guest uploads are paused. Please retry.")
    else:
        st.error("Datasets are temporarily unavailable. Please retry in a moment.")
    if st.button("Retry connection", key=f"retry_{key}"):
        st.rerun()
