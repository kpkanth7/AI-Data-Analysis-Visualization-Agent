import os
import streamlit as st


def render_history_tab():
    st.header("📜 Session History")

    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        st.info("No sessions saved yet.")
        return

    log_files = sorted(
        [f for f in os.listdir(logs_dir) if f.endswith(".txt")],
        reverse=True,
    )

    if not log_files:
        st.info("No sessions saved yet.")
        return

    st.caption(f"{len(log_files)} saved session(s)")

    for fname in log_files[:20]:
        path = os.path.join(logs_dir, fname)
        ts = fname.replace("session_", "").replace(".txt", "")
        try:
            formatted = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]}:{ts[13:15]}"
        except Exception:
            formatted = fname

        with st.expander(f"Session: {formatted}"):
            try:
                with open(path) as f:
                    content = f.read()
                st.text(content[:2000] + ("..." if len(content) > 2000 else ""))
            except Exception as e:
                st.warning(f"Could not read log: {e}")
            if st.button("🗑️ Delete", key=f"del_{fname}"):
                try:
                    os.remove(path)
                    st.rerun()
                except Exception as e:
                    st.error(f"Delete failed: {e}")
