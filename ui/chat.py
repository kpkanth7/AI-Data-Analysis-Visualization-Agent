import json
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from agent.agent import run_agent
from agent.schema import AnalysisOutput
from ui.charts import build_chart
from ui.sidebar import can_query, increment_query_count


def _render_analysis_result(analysis: AnalysisOutput):
    """Render charts, sub-results, export buttons inline in the chat message."""
    # Multi-subquery results
    if analysis.sub_results:
        for sub in analysis.sub_results:
            with st.container():
                st.markdown(f"**[{sub.index}] {sub.question}**")
                st.markdown(sub.answer)
                if sub.chart_config:
                    try:
                        fig = build_chart(sub.chart_config)
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            key=f"chart_{sub.index}_{id(sub)}",
                        )
                    except Exception as e:
                        st.warning(f"Chart render failed: {e}")
                if sub.sql_used:
                    with st.expander("SQL used"):
                        st.code(sub.sql_used, language="sql")
                if sub.export_path:
                    try:
                        with open(sub.export_path, "rb") as f:
                            st.download_button(
                                "📥 Download Excel",
                                data=f.read(),
                                file_name=sub.export_path.split("/")[-1],
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key=f"dl_{sub.index}_{id(sub)}",
                            )
                    except Exception:
                        pass

    # Top-level export (if no sub-result exports)
    if analysis.export_path and not any(s.export_path for s in analysis.sub_results):
        try:
            with open(analysis.export_path, "rb") as f:
                st.download_button(
                    "📥 Download Excel",
                    data=f.read(),
                    file_name=analysis.export_path.split("/")[-1],
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"dl_main_{id(analysis)}",
                )
        except Exception:
            pass

    if analysis.datasets_used:
        st.caption(f"Datasets: {', '.join(f'`{d}`' for d in analysis.datasets_used)}")


def render_chat_tab():
    st.header("💬 Ask Your Data")

    # Re-render full chat history
    for msg in st.session_state.get("chat_history", []):
        role = msg["role"]
        with st.chat_message(role, avatar="🧑" if role == "user" else "🤖"):
            st.markdown(msg["content"])
            if msg.get("analysis"):
                _render_analysis_result(msg["analysis"])

    # Chat input
    query = st.chat_input("Ask anything about your data...", key="chat_input")
    if not query:
        return

    # Guest quota enforcement
    if not can_query():
        st.error("⛔ Guest session limit (5 queries) reached. Reload the page for a new session.")
        return

    # Render user message immediately
    with st.chat_message("user", avatar="🧑"):
        st.markdown(query)
    st.session_state["chat_history"].append({"role": "user", "content": query})

    # Run agent with streaming
    with st.chat_message("assistant", avatar="🤖"):
        steps_container = st.empty()
        with st.spinner("Thinking..."):
            analysis = run_agent(
                query=query,
                chat_history=st.session_state.get("lc_history", []),
                step_container=steps_container,
            )
        steps_container.empty()

        if analysis:
            st.markdown(analysis.answer)
            _render_analysis_result(analysis)

            # Update LangChain message history
            st.session_state["lc_history"].append(HumanMessage(content=query))
            st.session_state["lc_history"].append(AIMessage(content=analysis.answer))

            # Persist for re-render on next run
            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": analysis.answer,
                "analysis": analysis,
            })
        else:
            st.error("Agent returned no response. Please try again.")

    increment_query_count()
