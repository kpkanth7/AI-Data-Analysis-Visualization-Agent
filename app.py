import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark theme polish — tighter padding, styled bubbles, code blocks
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1200px;
    }
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        margin-bottom: 0.5rem;
        padding: 0.75rem 1rem;
    }
    [data-testid="stExpander"] {
        border: 1px solid #2a3a4a;
        border-radius: 8px;
    }
    code {
        background: #1e2a3a !important;
        padding: 2px 6px;
        border-radius: 4px;
    }
    .stDownloadButton > button {
        background-color: #1f6feb;
        color: white;
        border: none;
        border-radius: 6px;
    }
    .stDownloadButton > button:hover {
        background-color: #388bfd;
    }
    h1, h2, h3 { color: #e6edf3; }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

from ui.sidebar import render_sidebar
from ui.chat import render_chat_tab
from ui.explorer import render_explorer_tab
from ui.history import render_history_tab

render_sidebar()

tab_chat, tab_explore, tab_history = st.tabs(["💬 Chat", "🔍 Explorer", "📜 History"])

with tab_chat:
    render_chat_tab()

with tab_explore:
    render_explorer_tab()

with tab_history:
    render_history_tab()
