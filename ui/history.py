"""History tab: browse, view, and delete saved owner sessions."""
import streamlit as st
import pandas as pd

from core.session_manager import list_sessions, delete_session, load_session
from ui.sidebar import is_owner


def render_history_tab() -> None:
    if not is_owner():
        st.info("Session history is available to the owner only.")
        return

    st.markdown("### 📜 Saved Sessions")

    sessions = list_sessions()
    if not sessions:
        st.info("No sessions saved yet. Use **💾 Save** in the sidebar to save a session.")
        return

    st.caption(f"{len(sessions)} saved session(s)")

    for s in sessions:
        label = s["label"] or "Untitled session"
        header = f"**{s['ts_display']}** · {label} · {s['message_count']} messages"

        with st.expander(header, expanded=False):
            data = s.get("data") or {}
            messages = data.get("messages", [])

            # Render a readable transcript
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    st.markdown(
                        f'<div style="background:#1e3a5f;padding:8px 12px;border-radius:8px;'
                        f'margin-bottom:6px;border-left:3px solid #3b82f6">'
                        f'<small style="color:#93c5fd">You</small><br>{content}</div>',
                        unsafe_allow_html=True,
                    )
                elif role == "assistant":
                    st.markdown(
                        f'<div style="background:#1a2535;padding:8px 12px;border-radius:8px;'
                        f'margin-bottom:6px;border-left:3px solid #6366f1">'
                        f'<small style="color:#a5b4fc">Agent</small><br>{content}</div>',
                        unsafe_allow_html=True,
                    )
                    # Show data previews from saved analysis
                    analysis = msg.get("analysis") or {}
                    preview = analysis.get("data_preview") or []
                    if preview:
                        st.dataframe(
                            pd.DataFrame(preview[:8]),
                            use_container_width=True,
                            hide_index=True,
                        )
                    for sub in analysis.get("sub_results") or []:
                        if sub.get("data_preview"):
                            st.caption(f"[{sub['index']}] {sub.get('question', '')}")
                            st.dataframe(
                                pd.DataFrame(sub["data_preview"][:5]),
                                use_container_width=True,
                                hide_index=True,
                            )

            col_dl, col_del = st.columns([3, 1])
            # Download raw JSON
            try:
                import json
                raw_json = json.dumps(data, indent=2, default=str).encode()
                col_dl.download_button(
                    "⬇ Download JSON",
                    data=raw_json,
                    file_name=s["filename"],
                    mime="application/json",
                    key=f"dl_{s['filename']}",
                    use_container_width=True,
                )
            except Exception:
                pass

            if col_del.button("🗑 Delete", key=f"del_{s['filename']}", use_container_width=True):
                try:
                    delete_session(s["path"])
                    st.rerun()
                except Exception as e:
                    st.error(f"Delete failed: {e}")
