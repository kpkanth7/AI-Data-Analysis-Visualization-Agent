import json
import streamlit as st
import pandas as pd
from db.postgres import list_datasets_from_db
from core.dataset_profiler import read_uploaded_file, ingest_dataframe

GUEST_QUERY_LIMIT = 5


def init_session_state():
    defaults = {
        "is_owner": False,
        "guest_queries": 0,
        "chat_history": [],
        "lc_history": [],
        "active_dataset": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def check_owner_auth() -> bool:
    if st.session_state.get("is_owner"):
        return True
    with st.sidebar.expander("🔐 Owner Login", expanded=False):
        pw = st.text_input("Password", type="password", key="owner_pw_input")
        if st.button("Login", key="owner_login_btn"):
            try:
                expected = st.secrets.get("owner_password", "")
            except Exception:
                expected = ""
            if pw and pw == expected:
                st.session_state["is_owner"] = True
                st.rerun()
            else:
                st.error("Wrong password")
    return False


def can_query() -> bool:
    if st.session_state.get("is_owner"):
        return True
    return st.session_state.get("guest_queries", 0) < GUEST_QUERY_LIMIT


def increment_query_count():
    if not st.session_state.get("is_owner"):
        st.session_state["guest_queries"] = st.session_state.get("guest_queries", 0) + 1


def render_sidebar():
    init_session_state()

    st.sidebar.title("📊 Data Analyst AI")
    st.sidebar.markdown("---")

    is_owner = check_owner_auth()

    # Role badge
    if is_owner:
        st.sidebar.success("👑 Owner — unlimited queries")
    else:
        used = st.session_state.get("guest_queries", 0)
        remaining = GUEST_QUERY_LIMIT - used
        if remaining > 0:
            st.sidebar.info(f"👤 Guest — {remaining}/{GUEST_QUERY_LIMIT} queries left")
        else:
            st.sidebar.error("⛔ Guest limit reached. Reload for new session.")

    st.sidebar.markdown("---")

    # Upload (owner only)
    if is_owner:
        st.sidebar.subheader("📁 Upload Dataset")
        uploaded = st.sidebar.file_uploader(
            "CSV or Excel",
            type=["csv", "xlsx", "xls"],
            key="file_uploader",
        )
        if uploaded:
            with st.sidebar.status(f"Ingesting {uploaded.name}..."):
                try:
                    df = read_uploaded_file(uploaded)
                    slug = ingest_dataframe(df, uploaded.name)
                    st.sidebar.success(f"✅ Loaded `{slug}` ({len(df):,} rows)")
                except Exception as e:
                    st.sidebar.error(f"Upload failed: {e}")
        st.sidebar.markdown("---")

    # Dataset list
    st.sidebar.subheader("🗃️ Datasets")
    try:
        datasets = list_datasets_from_db()
    except Exception:
        datasets = []

    if not datasets:
        st.sidebar.caption("No datasets loaded yet.")
    else:
        for ds in datasets:
            is_demo = ds.get("is_demo", False)
            label = f"{'★ ' if is_demo else ''}{ds['name']}"
            with st.sidebar.expander(label):
                row_count = ds.get("row_count") or 0
                st.caption(f"Table: `{ds['slug']}` · {row_count:,} rows")
                cols = ds.get("columns_json") or []
                if isinstance(cols, str):
                    cols = json.loads(cols)
                if cols:
                    col_df = pd.DataFrame(cols)
                    st.dataframe(col_df, hide_index=True, use_container_width=True)

    # Guest demo hints
    if not is_owner:
        demo_datasets = [d for d in datasets if d.get("is_demo")]
        if demo_datasets:
            st.sidebar.markdown("---")
            st.sidebar.subheader("💡 Try These Queries")
            hints = [
                "Total revenue by region?",
                "Show sales trend over time",
                "Detect anomalies in revenue AND top 5 products",
                "Correlation between profit and sales?",
                "Which country has highest CO2 emissions?",
            ]
            for h in hints:
                st.sidebar.caption(f"→ *{h}*")
