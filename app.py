import streamlit as st


try:
    settings = st.secrets.get("private_pilot", {})
except OSError:
    settings = {}

if hasattr(settings, "get") and settings.get("enabled") is True:
    from private_pilot_app import main
else:
    from demo_app import main

main()
