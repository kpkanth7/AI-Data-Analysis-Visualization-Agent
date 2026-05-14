"""Multi-step cleaning wizard UI. Owner-only. Stores progress in st.session_state['_cleaning']."""
import pandas as pd
import streamlit as st

from core.cleaning_wizard import (
    profile_for_cleaning,
    apply_duplicates,
    apply_nulls,
    apply_types,
    apply_outliers,
)
from core.dataset_profiler import ingest_dataframe, read_uploaded_file


STEPS = ["profile", "duplicates", "nulls", "types", "outliers", "preview", "commit"]


def _state() -> dict:
    s = st.session_state.get("_cleaning")
    if not s:
        return {}
    return s


def start_wizard(uploaded_file) -> None:
    df = read_uploaded_file(uploaded_file)
    st.session_state["_cleaning"] = {
        "step": "profile",
        "filename": uploaded_file.name,
        "df_original": df,
        "df": df.copy(),
        "plan": {},
    }


def cancel_wizard() -> None:
    st.session_state.pop("_cleaning", None)


def _go(step: str) -> None:
    st.session_state["_cleaning"]["step"] = step


def _next(current: str) -> None:
    i = STEPS.index(current)
    _go(STEPS[min(i + 1, len(STEPS) - 1)])


def _back(current: str) -> None:
    i = STEPS.index(current)
    _go(STEPS[max(i - 1, 0)])


def _progress_bar(current: str) -> None:
    idx = STEPS.index(current)
    pct = int((idx / (len(STEPS) - 1)) * 100)
    st.progress(pct, text=f"Step {idx + 1}/{len(STEPS)} — {current}")


def render_cleaning_wizard() -> None:
    """Renders inline. Call only when _cleaning state exists."""
    s = _state()
    if not s:
        return

    st.markdown(f"### 🧹 Cleaning `{s['filename']}`")
    _progress_bar(s["step"])

    df: pd.DataFrame = s["df"]

    if s["step"] == "profile":
        _step_profile(df)
    elif s["step"] == "duplicates":
        _step_duplicates(df)
    elif s["step"] == "nulls":
        _step_nulls(df)
    elif s["step"] == "types":
        _step_types(df)
    elif s["step"] == "outliers":
        _step_outliers(df)
    elif s["step"] == "preview":
        _step_preview(s)
    elif s["step"] == "commit":
        _step_commit(s)

    st.markdown("---")
    cancel_col, _ = st.columns([1, 5])
    if cancel_col.button("✖ Cancel cleaning", key="clean_cancel"):
        cancel_wizard()
        st.rerun()


# ── Steps ─────────────────────────────────────────────────────────────────────

def _step_profile(df: pd.DataFrame) -> None:
    info = profile_for_cleaning(df)
    st.markdown(f"**{info['rows']:,} rows × {info['n_cols']} columns** · "
                f"{info['dup_count']:,} duplicate rows detected")
    summary_df = pd.DataFrame([
        {
            "column": c["name"],
            "dtype": c["dtype"],
            "nulls": c["null_count"],
            "null %": c["null_pct"],
            "unique": c["unique_count"],
            "sample": ", ".join(c["sample"][:2]),
        }
        for c in info["cols"]
    ])
    st.dataframe(summary_df, hide_index=True, use_container_width=True)
    c1, c2 = st.columns([1, 1])
    if c1.button("Continue →", key="clean_p_next", type="primary"):
        _next("profile"); st.rerun()


def _step_duplicates(df: pd.DataFrame) -> None:
    dup = int(df.duplicated().sum())
    st.markdown(f"**Duplicate rows detected:** {dup:,}")
    if dup == 0:
        st.success("No duplicates to remove.")
    choice = st.radio(
        "How to handle?",
        ["keep_first", "keep_last", "drop_all", "skip"],
        format_func=lambda x: {
            "keep_first": "Keep first occurrence",
            "keep_last": "Keep last occurrence",
            "drop_all": "Drop all duplicates",
            "skip": "Skip — leave as is",
        }[x],
        key="clean_dup_choice",
    )
    c1, c2 = st.columns(2)
    if c1.button("← Back", key="clean_d_back"):
        _back("duplicates"); st.rerun()
    if c2.button("Continue →", key="clean_d_next", type="primary"):
        s = _state()
        s["df"] = apply_duplicates(df, choice)
        s["plan"]["duplicates"] = choice
        _next("duplicates"); st.rerun()


def _step_nulls(df: pd.DataFrame) -> None:
    null_cols = [c for c in df.columns if df[c].isna().any()]
    if not null_cols:
        st.success("No null values found. Skipping.")
        c1, c2 = st.columns(2)
        if c1.button("← Back", key="clean_n_back"):
            _back("nulls"); st.rerun()
        if c2.button("Continue →", key="clean_n_next", type="primary"):
            _next("nulls"); st.rerun()
        return

    st.markdown("**Per-column null handling**")
    plan: dict[str, dict] = {}
    for col in null_cols:
        nulls = int(df[col].isna().sum())
        is_num = pd.api.types.is_numeric_dtype(df[col])
        default_idx = 1 if is_num else 3  # mean for num, mode for non-num
        opts = ["drop_row", "fill_mean", "fill_median", "fill_mode", "fill_const", "leave"]
        if not is_num:
            opts = ["drop_row", "fill_mode", "fill_const", "leave"]
            default_idx = 1
        labels = {
            "drop_row": "Drop rows with null in this column",
            "fill_mean": "Fill with mean",
            "fill_median": "Fill with median",
            "fill_mode": "Fill with mode (most common)",
            "fill_const": "Fill with constant",
            "leave": "Leave as null",
        }
        cols_row = st.columns([3, 3, 2])
        cols_row[0].markdown(f"`{col}` — {nulls} nulls ({df[col].dtype})")
        action = cols_row[1].selectbox(
            "Action", opts, format_func=lambda x: labels[x],
            index=default_idx, key=f"clean_null_{col}", label_visibility="collapsed",
        )
        const_val = ""
        if action == "fill_const":
            const_val = cols_row[2].text_input(
                "Value", key=f"clean_null_const_{col}", label_visibility="collapsed",
                placeholder="e.g. 0 or N/A",
            )
        plan[col] = {"action": action, "const": const_val}

    c1, c2 = st.columns(2)
    if c1.button("← Back", key="clean_n_back2"):
        _back("nulls"); st.rerun()
    if c2.button("Continue →", key="clean_n_next2", type="primary"):
        s = _state()
        s["df"] = apply_nulls(df, plan)
        s["plan"]["nulls"] = plan
        _next("nulls"); st.rerun()


def _step_types(df: pd.DataFrame) -> None:
    st.markdown("**Column type review** — accept current or convert.")
    plan: dict[str, str] = {}
    for col in df.columns:
        current = str(df[col].dtype)
        suggested = "keep"
        if df[col].dtype == object:
            sample = df[col].dropna().astype(str).head(20)
            if sample.str.match(r"^-?\d+\.?\d*$").mean() > 0.8 if len(sample) else False:
                suggested = "numeric"
            elif ("date" in col.lower() or "time" in col.lower()):
                suggested = "datetime"
        opts = ["keep", "datetime", "numeric", "string"]
        idx = opts.index(suggested) if suggested in opts else 0
        cols_row = st.columns([3, 2, 3])
        cols_row[0].markdown(f"`{col}`")
        cols_row[1].caption(f"current: `{current}`")
        choice = cols_row[2].selectbox(
            "Target", opts, index=idx, key=f"clean_type_{col}",
            label_visibility="collapsed",
        )
        plan[col] = choice

    c1, c2 = st.columns(2)
    if c1.button("← Back", key="clean_t_back"):
        _back("types"); st.rerun()
    if c2.button("Continue →", key="clean_t_next", type="primary"):
        s = _state()
        s["df"] = apply_types(df, plan)
        s["plan"]["types"] = plan
        _next("types"); st.rerun()


def _step_outliers(df: pd.DataFrame) -> None:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        st.info("No numeric columns. Skipping outlier handling.")
        c1, c2 = st.columns(2)
        if c1.button("← Back", key="clean_o_back"):
            _back("outliers"); st.rerun()
        if c2.button("Continue →", key="clean_o_next", type="primary"):
            _next("outliers"); st.rerun()
        return

    method = st.radio("Detection method", ["iqr", "zscore"], horizontal=True,
                      format_func=lambda x: "IQR (1.5×)" if x == "iqr" else "Z-score",
                      key="clean_outlier_method")
    z_thresh = 3.0
    if method == "zscore":
        z_thresh = st.number_input("Z-score threshold", 1.0, 6.0, 3.0, 0.5,
                                   key="clean_zthresh")

    plan: dict[str, dict] = {}
    for col in num_cols:
        s = df[col]
        if method == "iqr":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            mask = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)
        else:
            mu, sd = s.mean(), s.std()
            mask = (s < mu - z_thresh * sd) | (s > mu + z_thresh * sd)
        outlier_count = int(mask.sum())
        cols_row = st.columns([3, 2, 2])
        cols_row[0].markdown(f"`{col}`")
        cols_row[1].caption(f"{outlier_count} outliers")
        action = cols_row[2].selectbox(
            "Action", ["leave", "clip", "drop"], key=f"clean_out_{col}",
            label_visibility="collapsed",
        )
        plan[col] = {"action": action}

    c1, c2 = st.columns(2)
    if c1.button("← Back", key="clean_o_back2"):
        _back("outliers"); st.rerun()
    if c2.button("Continue →", key="clean_o_next2", type="primary"):
        st_s = _state()
        st_s["df"] = apply_outliers(st_s["df"], plan, method=method, z_thresh=z_thresh)
        st_s["plan"]["outliers"] = {"cols": plan, "method": method, "z_thresh": z_thresh}
        _next("outliers"); st.rerun()


def _step_preview(s: dict) -> None:
    orig: pd.DataFrame = s["df_original"]
    cleaned: pd.DataFrame = s["df"]

    c1, c2 = st.columns(2)
    c1.metric("Rows", f"{len(cleaned):,}", delta=f"{len(cleaned) - len(orig):,}")
    c2.metric("Total nulls", int(cleaned.isna().sum().sum()),
              delta=int(cleaned.isna().sum().sum() - orig.isna().sum().sum()))

    st.markdown("**Cleaned sample (first 8 rows)**")
    st.dataframe(cleaned.head(8), use_container_width=True, hide_index=True)

    st.markdown("**Plan summary**")
    st.json(s["plan"])

    confirm = st.checkbox(
        "I understand: cleaned data will replace the original table in the database.",
        key="clean_confirm",
    )

    c_back, c_commit = st.columns(2)
    if c_back.button("← Back", key="clean_p_back2"):
        _back("preview"); st.rerun()
    if c_commit.button("✅ Commit & Ingest", key="clean_commit",
                       type="primary", disabled=not confirm):
        _next("preview"); st.rerun()


def _step_commit(s: dict) -> None:
    with st.spinner("Ingesting cleaned dataset…"):
        try:
            slug = ingest_dataframe(
                s["df"], s["filename"],
                owner_only=True, session_id=None,
            )
            st.success(f"✅ Ingested as `{slug}` — {len(s['df']):,} rows.")
            cancel_wizard()
            if st.button("Done", key="clean_done", type="primary"):
                st.rerun()
        except Exception as e:
            st.error(f"Ingest failed: {e}")
            if st.button("← Back to preview", key="clean_commit_back"):
                _back("commit"); st.rerun()
