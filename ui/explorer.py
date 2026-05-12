import json
import streamlit as st
import pandas as pd
from db.postgres import list_datasets_from_db, get_engine
from core.exporter import export_to_excel


def render_explorer_tab():
    st.header("🔍 Data Explorer")

    try:
        datasets = list_datasets_from_db()
    except Exception as e:
        st.error(f"Could not load datasets: {e}")
        return

    if not datasets:
        st.info("No datasets loaded. Upload a CSV from the sidebar (owner only).")
        return

    slugs = [d["slug"] for d in datasets]
    selected = st.selectbox("Select dataset", slugs, key="explorer_table_select")

    if not selected:
        return

    ds_meta = next((d for d in datasets if d["slug"] == selected), {})
    row_count = ds_meta.get("row_count") or 0

    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(f"**{ds_meta.get('name', selected)}** — {row_count:,} rows")
    with col2:
        page_size = st.selectbox("Rows/page", [25, 50, 100, 250], index=1, key="explorer_page_size")

    # Load data
    try:
        engine = get_engine()
        df = pd.read_sql_table(selected, engine)
    except Exception as e:
        st.error(f"Could not load `{selected}`: {e}")
        return

    # Filter
    search = st.text_input(
        "Filter (pandas query expression)",
        placeholder='e.g. region == "North" and revenue > 500',
        key="explorer_filter",
    )
    filtered_df = df
    if search:
        try:
            filtered_df = df.query(search)
            st.caption(f"Filtered: {len(filtered_df):,} rows")
        except Exception as e:
            st.warning(f"Filter error: {e}")

    # Pagination
    total = len(filtered_df)
    max_page = max(1, (total - 1) // page_size + 1)
    page = st.number_input("Page", min_value=1, max_value=max_page, value=1, key="explorer_page")
    start = (page - 1) * page_size
    st.dataframe(
        filtered_df.iloc[start: start + page_size],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"Rows {start + 1}–{min(start + page_size, total):,} of {total:,}")

    # Export
    if st.button("📥 Export to Excel", key="explorer_export_btn"):
        try:
            path = export_to_excel(filtered_df, title=f"{selected} Export")
            with open(path, "rb") as f:
                st.download_button(
                    "Download Excel",
                    data=f.read(),
                    file_name=f"{selected}_export.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="explorer_dl_btn",
                )
        except Exception as e:
            st.error(f"Export failed: {e}")
