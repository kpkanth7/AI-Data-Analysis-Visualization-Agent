"""Chat tab: renders messages, runs agent, handles exports."""
import io
import datetime
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from agent.agent import run_agent
from agent.schema import AnalysisOutput, SubQueryResult
from ui.charts import build_chart
from ui.sidebar import can_query, record_query, is_owner, get_session_id
from core.pdf_export import chart_to_png, dataframe_to_csv, session_to_pdf


# ── Export helpers ─────────────────────────────────────────────────────────────

def _csv_download(rows: list[dict], key: str) -> None:
    if not rows:
        return
    data = dataframe_to_csv(rows)
    st.download_button(
        "⬇ Download CSV",
        data=data,
        file_name=f"data_{key}.csv",
        mime="text/csv",
        key=f"csv_{key}",
        use_container_width=True,
        type="secondary",
    )


def _png_download(fig, key: str) -> bool:
    """Returns True if the button was rendered (kaleido available)."""
    try:
        png = chart_to_png(fig)
        st.download_button(
            "⬇ Download PNG",
            data=png,
            file_name=f"chart_{key}.png",
            mime="image/png",
            key=f"png_{key}",
            use_container_width=True,
            type="secondary",
        )
        return True
    except Exception:
        return False


def _render_chart(fig, chart_key: str, chart_store: dict) -> None:
    """Render a chart with its download button in a clean layout."""
    chart_store[chart_key] = fig
    st.plotly_chart(fig, use_container_width=True, key=f"pc_{chart_key}")
    # Download button flush below chart, small width so it doesn't dominate
    dl_col, _ = st.columns([1, 3])
    with dl_col:
        _png_download(fig, key=chart_key)


def _render_table(rows: list[dict], table_key: str) -> None:
    """Render a dataframe with its download button."""
    import pandas as pd
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )
    dl_col, _ = st.columns([1, 3])
    with dl_col:
        _csv_download(rows, key=table_key)


# ── Result renderer ────────────────────────────────────────────────────────────

def _render_result(analysis: AnalysisOutput, msg_key: str, chart_store: dict) -> None:
    """Render answer, data table, charts, and export buttons."""

    # ── Multi-subquery path ────────────────────────────────────────────────────
    if analysis.sub_results:
        for sub in analysis.sub_results:
            with st.container():
                st.markdown(f"**[{sub.index}] {sub.question}**")
                st.markdown(sub.answer)

                if sub.data_preview:
                    _render_table(sub.data_preview, table_key=f"{msg_key}_sub{sub.index}")

                if sub.chart_config:
                    try:
                        fig = build_chart(sub.chart_config)
                        _render_chart(fig, chart_key=f"sub_{msg_key}_{sub.index}", chart_store=chart_store)
                    except Exception as e:
                        st.warning(f"Chart render failed: {e}")

                if sub.sql_used:
                    with st.expander("SQL", expanded=False):
                        st.code(sub.sql_used, language="sql")

                if sub.export_path:
                    try:
                        with open(sub.export_path, "rb") as f:
                            dl_col, _ = st.columns([1, 3])
                            with dl_col:
                                st.download_button(
                                    "⬇ Download Excel",
                                    data=f.read(),
                                    file_name=sub.export_path.split("/")[-1],
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key=f"xl_{msg_key}_sub{sub.index}",
                                    use_container_width=True,
                                    type="secondary",
                                )
                    except Exception:
                        pass

        return  # sub_results handled everything

    # ── Single-question path ───────────────────────────────────────────────────

    if analysis.data_preview:
        _render_table(analysis.data_preview, table_key=f"{msg_key}_main")

    if analysis.chart_config:
        try:
            fig = build_chart(analysis.chart_config)
            _render_chart(fig, chart_key=str(msg_key), chart_store=chart_store)
        except Exception as e:
            st.warning(f"Chart render failed: {e}")

    # SQL used — always show so user can verify every factual claim
    if analysis.sql_used:
        with st.expander("🔍 Verify SQL", expanded=False):
            st.code(analysis.sql_used, language="sql")

    # Excel export
    if analysis.export_path:
        try:
            with open(analysis.export_path, "rb") as f:
                st.download_button(
                    "⬇ Download Excel",
                    data=f.read(),
                    file_name=analysis.export_path.split("/")[-1],
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"xl_{msg_key}_main",
                    use_container_width=True,
                )
        except Exception:
            pass

    # Datasets used
    if analysis.datasets_used:
        st.caption(f"Datasets: {', '.join(f'`{d}`' for d in analysis.datasets_used)}")


def _render_steps_expander(steps: list[dict]) -> None:
    if not steps:
        return
    with st.expander(f"🔍 Agent steps ({len(steps)})", expanded=False):
        for s in steps:
            icon = "✅" if s["status"] == "done" else ("❌" if s["status"] == "error" else "⏳")
            hint = f" → {s['output_hint']}" if s.get("output_hint") else ""
            st.markdown(f"{icon} `{s['tool']}` ← `{s['input'][:100]}`{hint}")


# ── PDF export (header bar) ────────────────────────────────────────────────────

def _render_pdf_export_button(chart_store: dict) -> None:
    history = st.session_state.get("chat_history", [])
    disabled = len(history) == 0
    if st.button(
        "📄 Export PDF",
        key="export_pdf_btn",
        disabled=disabled,
        help="Export this session as a structured PDF" if not disabled else "Start a conversation first",
        use_container_width=True,
    ):
        with st.spinner("Building PDF…"):
            try:
                pdf_bytes = session_to_pdf(history, charts=chart_store)
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    "⬇ Download PDF",
                    data=pdf_bytes,
                    file_name=f"session_{ts}.pdf",
                    mime="application/pdf",
                    key=f"dl_pdf_{ts}",
                )
            except Exception as e:
                st.error(f"PDF export failed: {e}")


# ── Main tab renderer ──────────────────────────────────────────────────────────

def render_chat_tab() -> None:
    if "chart_store" not in st.session_state:
        st.session_state["chart_store"] = {}
    chart_store: dict = st.session_state["chart_store"]

    # ── Header row: title + PDF export ────────────────────────────────────────
    hcol, pcol = st.columns([5, 1])
    hcol.markdown("#### 💬 Ask Your Data")
    with pcol:
        _render_pdf_export_button(chart_store)

    # ── Scrollable message area ────────────────────────────────────────────────
    # Fixed-height container keeps the chat input anchored below it at all times.
    with st.container(height=560, border=False):
        history = st.session_state.get("chat_history", [])

        # Render persisted messages
        for i, msg in enumerate(history):
            role = msg["role"]
            with st.chat_message(role, avatar="🧑" if role == "user" else "🤖"):
                st.markdown(msg["content"])
                if role == "assistant" and msg.get("analysis"):
                    _render_result(msg["analysis"], msg_key=i, chart_store=chart_store)
                if role == "assistant" and msg.get("steps"):
                    _render_steps_expander(msg["steps"])

        # Process any pending query (stored by previous run)
        pending = st.session_state.pop("_pending_query", None)
        if pending:
            with st.chat_message("user", avatar="🧑"):
                st.markdown(pending)
            st.session_state["chat_history"].append({"role": "user", "content": pending})

            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Analysing…"):
                    analysis, steps = run_agent(
                        query=pending,
                        chat_history=st.session_state.get("lc_history", []),
                        is_owner=is_owner(),
                        session_id=get_session_id(),
                    )

                if analysis:
                    st.markdown(analysis.answer)
                    msg_key = len(st.session_state["chat_history"])
                    _render_result(analysis, msg_key=msg_key, chart_store=chart_store)
                    _render_steps_expander(steps)

                    st.session_state["lc_history"].append(HumanMessage(content=pending))
                    st.session_state["lc_history"].append(AIMessage(content=analysis.answer))
                    st.session_state["chat_history"].append({
                        "role": "assistant",
                        "content": analysis.answer,
                        "analysis": analysis,
                        "steps": steps,
                    })
                else:
                    st.error("Agent returned no response. Try again.")

            record_query()

    # ── Chat input — always rendered after the container = always at bottom ────
    if not can_query():
        st.info("Daily query limit reached. Resets at midnight UTC.")
        return

    query = st.chat_input("Ask anything about your data…", key="chat_input")
    if query:
        # Store query and rerun so it's processed inside the container above
        st.session_state["_pending_query"] = query
        st.rerun()
