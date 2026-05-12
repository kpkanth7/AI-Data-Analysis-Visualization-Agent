import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Data Analysis Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Base & fonts ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Main container ───────────────────────────────────────────── */
.main .block-container {
    padding-top: 1.25rem;
    padding-bottom: 5rem;   /* room for pinned chat input */
    max-width: 1100px;
}

/* ── Sidebar ──────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stCaption {
    color: #94a3b8;
}

.sidebar-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.02em;
    padding: 4px 0;
}

/* ── Role badges ──────────────────────────────────────────────── */
.role-badge {
    padding: 8px 12px;
    border-radius: 8px;
    border: 1px solid #334155;
    font-size: 0.82rem;
    margin: 4px 0;
}
.owner-badge {
    background: #1e3a5f;
    color: #bfdbfe;
    border-color: #3b82f6;
}
.guest-badge {
    background: #1a2535;
    color: #d1d5db;
}

/* ── Login card ───────────────────────────────────────────────── */
.login-card {
    background: #1e293b;
    border-radius: 10px;
    padding: 12px;
    border: 1px solid #334155;
    margin: 8px 0;
}

/* ── Auth toggle buttons ──────────────────────────────────────── */
[data-testid="stSidebar"] .stButton button {
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.85rem;
    transition: all 0.15s ease;
}

/* ── Chat messages ────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: 14px;
    margin-bottom: 0.6rem;
    padding: 0.75rem 1rem;
}
[data-testid="stChatMessage"][data-testid*="user"] {
    background: #1e2d3d;
}

/* ── Expanders ────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #1e293b;
    border-radius: 8px;
    background: #0f172a;
}
[data-testid="stExpander"] summary {
    color: #64748b;
    font-size: 0.82rem;
}

/* ── Download buttons ─────────────────────────────────────────── */
.stDownloadButton > button {
    background: #1e3a5f;
    color: #93c5fd;
    border: 1px solid #3b82f6;
    border-radius: 7px;
    font-size: 0.8rem;
    padding: 4px 12px;
}
.stDownloadButton > button:hover {
    background: #2d4f7c;
    color: #bfdbfe;
}

/* ── Dataframe ────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 8px;
    border: 1px solid #1e293b;
    overflow: hidden;
}

/* ── Tabs ─────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 1px solid #1e293b;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.9rem;
    font-weight: 500;
    color: #64748b;
    border-radius: 8px 8px 0 0;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    color: #f1f5f9 !important;
    background: #1e293b !important;
}

/* ── Code blocks ──────────────────────────────────────────────── */
code {
    background: #1e2a3a !important;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.85em;
}

/* ── Headings ─────────────────────────────────────────────────── */
h1, h2, h3, h4 { color: #f1f5f9; letter-spacing: -0.02em; }

/* ── Chat input: keep at bottom, styled ──────────────────────── */
[data-testid="stChatInput"] {
    border-radius: 14px !important;
    border: 1px solid #334155 !important;
    background: #1e293b !important;
    box-shadow: 0 2px 20px rgba(0,0,0,0.3) !important;
}
[data-testid="stChatInput"] textarea {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    color: #f1f5f9 !important;
    background: transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #475569 !important;
}

/* ── Spinner ──────────────────────────────────────────────────── */
[data-testid="stSpinner"] {
    color: #6366f1;
}

/* ── Buttons – small session controls ────────────────────────── */
[data-testid="stSidebar"] .stButton button[kind="secondary"] {
    background: #1e293b;
    color: #94a3b8;
    border: 1px solid #334155;
}
[data-testid="stSidebar"] .stButton button[kind="secondary"]:hover {
    background: #273548;
    color: #f1f5f9;
}
</style>
""", unsafe_allow_html=True)

from ui.sidebar import render_sidebar
from ui.chat import render_chat_tab
from ui.explorer import render_explorer_tab
from ui.history import render_history_tab
from core.session_manager import auto_save_if_nonempty

render_sidebar()

tab_chat, tab_explore, tab_history = st.tabs(["💬 Chat", "🔍 Explorer", "📜 History"])

with tab_chat:
    render_chat_tab()

with tab_explore:
    render_explorer_tab()

with tab_history:
    render_history_tab()
