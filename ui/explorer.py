import streamlit as st
import pandas as pd
from sqlalchemy import text

from db.postgres import list_datasets_from_db, get_connection
from core.exporter import export_to_excel
from ui.auth import is_owner, get_session_id


def _nl_to_sql_where(nl: str, columns: list[str], table: str) -> str:
    """Convert plain-English filter to a SQL WHERE clause via LLM."""
    import os, json
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0,
                     api_key=os.getenv("OPENAI_API_KEY", ""))
    cols_hint = ", ".join(columns[:30])
    prompt = (
        f"Table `{table}` has columns: {cols_hint}.\n"
        f"Convert this plain-English filter to a PostgreSQL WHERE clause (no WHERE keyword, "
        f"just the condition). Return ONLY the SQL condition, nothing else.\n"
        f"Filter: {nl}"
    )
    resp = llm.invoke(prompt)
    clause = resp.content.strip().strip(";").strip()
    # Safety: reject anything with DML/DDL
    if any(kw in clause.upper() for kw in
           ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", ";")):
        raise ValueError("Unsafe filter generated.")
    return clause


def render_explorer_tab():
    st.header("🔍 Data Explorer")

    try:
        datasets = list_datasets_from_db(
            include_owner_only=is_owner(),
            session_id=get_session_id(),
        )
    except Exception as e:
        st.error(f"Could not load datasets: {e}")
        return

    if not datasets:
        st.info("No datasets loaded yet. Upload a CSV or Excel file from the **Data** tab.")
        return

    slugs = [d["slug"] for d in datasets]
    selected = st.selectbox("Select dataset", slugs, key="explorer_table_select")
    if not selected:
        return

    ds_meta = next((d for d in datasets if d["slug"] == selected), {})
    row_count = ds_meta.get("row_count") or 0

    # Column metadata
    import json as _json
    cols_raw = ds_meta.get("columns_json") or []
    if isinstance(cols_raw, str):
        cols_raw = _json.loads(cols_raw)
    column_names = [c["name"] for c in cols_raw if isinstance(c, dict)]

    col_meta, col_page = st.columns([3, 1])
    with col_meta:
        st.caption(f"**{ds_meta.get('name', selected)}** — {row_count:,} rows")
    with col_page:
        page_size = st.selectbox("Rows/page", [25, 50, 100], index=1, key="explorer_page_size")

    # ── Filter bar ─────────────────────────────────────────────────────────────
    filter_mode = st.radio(
        "Filter mode", ["Plain English", "SQL WHERE"],
        horizontal=True, key="explorer_filter_mode",
        label_visibility="collapsed",
    )

    where_clause = st.session_state.get("_explorer_where", "")
    filter_err = ""

    if filter_mode == "Plain English":
        nl_input = st.text_input(
            "Describe what rows you want",
            placeholder='e.g. movies directed by Christopher Nolan after 2000',
            key="explorer_nl_input",
        )
        c_apply, c_clear = st.columns([1, 4])
        if c_apply.button("Apply", key="explorer_nl_apply"):
            if nl_input.strip():
                with st.spinner("Converting to SQL…"):
                    try:
                        clause = _nl_to_sql_where(nl_input, column_names, selected)
                        st.session_state["_explorer_where"] = clause
                        st.session_state["_explorer_nl_display"] = clause
                        where_clause = clause
                    except Exception as e:
                        filter_err = f"Could not convert: {e}"
            else:
                st.session_state["_explorer_where"] = ""
                where_clause = ""
        if c_clear.button("Clear filter", key="explorer_nl_clear"):
            st.session_state["_explorer_where"] = ""
            st.session_state.pop("_explorer_nl_display", None)
            where_clause = ""
        if st.session_state.get("_explorer_nl_display"):
            st.caption(f"SQL: `{st.session_state['_explorer_nl_display']}`")

    else:  # SQL WHERE mode
        sql_input = st.text_input(
            "SQL WHERE clause",
            placeholder="e.g. release_year > 2010 AND type = 'Movie'",
            key="explorer_sql_input",
        )
        c_apply2, c_clear2 = st.columns([1, 4])
        if c_apply2.button("Apply", key="explorer_sql_apply"):
            raw = sql_input.strip()
            if raw:
                if ";" in raw or any(kw in raw.upper() for kw in
                                     ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE")):
                    filter_err = "Only WHERE-style conditions allowed."
                else:
                    st.session_state["_explorer_where"] = raw
                    where_clause = raw
            else:
                st.session_state["_explorer_where"] = ""
                where_clause = ""
        if c_clear2.button("Clear filter", key="explorer_sql_clear"):
            st.session_state["_explorer_where"] = ""
            where_clause = ""

    if filter_err:
        st.warning(filter_err)
        where_clause = ""

    where_clause = st.session_state.get("_explorer_where", "")
    sql_where = f" WHERE {where_clause}" if where_clause else ""
    table_q = f'"{selected}"'

    # ── Count + paginate ───────────────────────────────────────────────────────
    try:
        with get_connection() as conn:
            total = conn.execute(text(f"SELECT COUNT(*) FROM {table_q}{sql_where}")).scalar() or 0
    except Exception as e:
        st.error(f"Filter error: {e}")
        return

    max_page = max(1, (total - 1) // page_size + 1)
    page = st.number_input("Page", min_value=1, max_value=max_page, value=1, key="explorer_page")
    offset = (page - 1) * page_size

    try:
        with get_connection() as conn:
            result = conn.execute(
                text(f"SELECT * FROM {table_q}{sql_where} LIMIT :lim OFFSET :off"),
                {"lim": page_size, "off": offset},
            )
            page_rows = [dict(r._mapping) for r in result.fetchall()]
    except Exception as e:
        st.error(f"Could not load data: {e}")
        return

    st.dataframe(pd.DataFrame(page_rows), use_container_width=True, hide_index=True)
    end = min(offset + page_size, total)
    st.caption(f"Rows {offset + 1}–{end:,} of {total:,}")

    # ── Export ─────────────────────────────────────────────────────────────────
    if st.button("📥 Export filtered to Excel", key="explorer_export_btn"):
        try:
            with get_connection() as conn:
                full = pd.read_sql(text(f"SELECT * FROM {table_q}{sql_where}"), conn)
            path = export_to_excel(full, title=f"{selected} Export")
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
