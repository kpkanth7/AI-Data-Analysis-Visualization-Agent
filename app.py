import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Data Analysis Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ─────────────────────────────────────────────────────── */
:root {
    --bg-0:#0a0e1a;
    --bg-1:#0f1525;
    --bg-2:#161e33;
    --bg-3:#1c263f;
    --line:#222d49;
    --line-strong:#2f3d62;
    --text-0:#f1f5f9;
    --text-1:#cbd5e1;
    --text-2:#94a3b8;
    --text-3:#64748b;
    --accent:#7c5cff;
    --accent-2:#ec4899;
    --accent-3:#22d3ee;
    --grad-1: linear-gradient(135deg, #7c5cff 0%, #ec4899 100%);
    --grad-2: linear-gradient(135deg, #22d3ee 0%, #7c5cff 100%);
    --grad-3: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
    --glow: 0 0 32px rgba(124, 92, 255, 0.35);
}

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
    background:
        radial-gradient(ellipse 80% 60% at 0% 0%, rgba(124,92,255,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 100% 0%, rgba(236,72,153,0.05) 0%, transparent 60%),
        radial-gradient(ellipse 70% 50% at 50% 100%, rgba(34,211,238,0.04) 0%, transparent 60%),
        #0a0e1a !important;
    color: var(--text-1);
}

/* ── Sidebar OFF ──────────────────────────────────────────────── */
[data-testid="stSidebar"], [data-testid="collapsedControl"] {
    display: none !important;
}

/* ── Kill outer frame padding ─────────────────────────────────── */
.main .block-container,
.appview-container .main .block-container,
[data-testid="stAppViewContainer"] > .main .block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    margin-top: 0 !important;
    max-width: 1260px;
}
.stApp { padding-top: 0 !important; }
header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {
    display: none !important;
    height: 0 !important;
    visibility: hidden !important;
}
[data-testid="stAppViewContainer"] { padding-top: 0 !important; }
[data-testid="stBottomBlockContainer"] {
    padding-top: 0 !important;
    padding-bottom: 4px !important;
    background: transparent !important;
}

/* ── Custom scrollbar ─────────────────────────────────────────── */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #2f3d62, #1c263f);
    border-radius: 999px;
    border: 2px solid #0a0e1a;
}
::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #7c5cff, #ec4899);
}

/* ── Typography ───────────────────────────────────────────────── */
h1, h2, h3, h4 {
    color: var(--text-0);
    letter-spacing: -0.025em;
    font-weight: 700;
}
h3 { font-size: 1.25rem; }
h4 { font-size: 1.05rem; }
p, .stMarkdown { color: var(--text-1); line-height: 1.55; }

/* ── Top bar ──────────────────────────────────────────────────── */
.tb-title {
    font-size: 1.65rem;
    font-weight: 800;
    background: var(--grad-3);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.03em;
    padding: 8px 0 4px 0;
    display: inline-block;
    line-height: 1.15;
}
.tb-chip-wrap {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100%;
    padding-top: 6px;
}
.tb-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 7px 14px;
    border-radius: 999px;
    border: 1px solid var(--line-strong);
    background: rgba(22, 30, 51, 0.7);
    backdrop-filter: blur(10px);
    color: var(--text-1);
    font-size: 0.82rem;
    font-weight: 500;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
}
.tb-chip-owner {
    background: linear-gradient(135deg, rgba(124,92,255,0.25), rgba(236,72,153,0.18));
    color: #f5d0fe;
    border-color: rgba(168,85,247,0.6);
    box-shadow: 0 0 24px rgba(168,85,247,0.35);
}
.tb-chip-good { border-color: rgba(34,197,94,0.55); color: #bbf7d0; box-shadow: 0 0 18px rgba(34,197,94,0.18); }
.tb-chip-warn { border-color: rgba(245,158,11,0.55); color: #fed7aa; box-shadow: 0 0 18px rgba(245,158,11,0.18); }
.tb-chip-bad  { border-color: rgba(239,68,68,0.6);  color: #fecaca; box-shadow: 0 0 18px rgba(239,68,68,0.22); }

/* Avatar (popover trigger) */
[data-testid="stPopover"] button {
    border-radius: 50% !important;
    width: 42px !important;
    height: 42px !important;
    padding: 0 !important;
    background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899) !important;
    color: #fff !important;
    border: 2px solid rgba(255,255,255,0.12) !important;
    font-weight: 800 !important;
    font-size: 0.95rem !important;
    box-shadow: 0 6px 18px rgba(124,92,255,0.45) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="stPopover"] button:hover {
    transform: translateY(-1px) scale(1.04) !important;
    box-shadow: 0 8px 24px rgba(236,72,153,0.5) !important;
}

/* ── Tabs ─────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 1px solid var(--line);
    gap: 6px;
    padding: 6px 0 0 0;
    margin-top: 4px;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.92rem;
    font-weight: 600;
    color: var(--text-3);
    border-radius: 10px 10px 0 0;
    padding: 10px 18px;
    background: rgba(22, 30, 51, 0.35);
    border: 1px solid transparent;
    transition: all 0.18s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-0);
    background: rgba(28, 38, 63, 0.7);
}
.stTabs [aria-selected="true"] {
    color: #fff !important;
    background: linear-gradient(180deg, rgba(124,92,255,0.22), rgba(28, 38, 63, 0.9)) !important;
    border-color: var(--line-strong) !important;
    border-bottom-color: transparent !important;
    box-shadow: inset 0 2px 0 0 #7c5cff;
}
.stTabs [data-baseweb="tab-highlight"] { background: transparent !important; }
.stTabs [data-baseweb="tab-border"] { background: var(--line) !important; }

/* ── Buttons ──────────────────────────────────────────────────── */
.stButton > button, .stDownloadButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
    padding: 8px 16px !important;
    transition: all 0.18s ease !important;
    border: 1px solid var(--line-strong) !important;
    background: rgba(22, 30, 51, 0.6) !important;
    color: var(--text-1) !important;
    backdrop-filter: blur(8px) !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-1px);
    border-color: #7c5cff !important;
    color: #fff !important;
    box-shadow: 0 6px 18px rgba(124,92,255,0.25) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #7c5cff 0%, #ec4899 100%) !important;
    border: 1px solid transparent !important;
    color: #fff !important;
    box-shadow: 0 6px 20px rgba(124,92,255,0.4) !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #8b6dff 0%, #f472b6 100%) !important;
    box-shadow: 0 8px 26px rgba(236,72,153,0.5) !important;
}
.stButton > button:disabled {
    background: rgba(15, 21, 37, 0.6) !important;
    color: var(--text-3) !important;
    border-color: var(--line) !important;
    cursor: not-allowed !important;
}

/* ── Chat messages ────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: 16px;
    margin-bottom: 0.7rem;
    padding: 0.85rem 1.1rem;
    border: 1px solid var(--line);
    background: rgba(22, 30, 51, 0.5);
    backdrop-filter: blur(6px);
    box-shadow: 0 4px 14px rgba(0,0,0,0.18);
}
[data-testid="stChatMessage"][data-testid*="user"] {
    background: linear-gradient(135deg, rgba(124,92,255,0.18), rgba(28, 38, 63, 0.85));
    border-color: rgba(124,92,255,0.35);
}

/* ── Chat input ───────────────────────────────────────────────── */
[data-testid="stBottom"] { padding: 0 !important; background: transparent !important; }
[data-testid="stBottom"] > div { padding-bottom: 0 !important; }
[data-testid="stChatInput"] {
    border-radius: 16px !important;
    border: 1px solid var(--line-strong) !important;
    background: rgba(15, 21, 37, 0.92) !important;
    backdrop-filter: blur(14px) !important;
    box-shadow:
        0 -6px 30px rgba(0,0,0,0.5),
        0 0 0 1px rgba(124,92,255,0.0) !important;
    transition: box-shadow 0.2s ease, border-color 0.2s ease !important;
    margin-bottom: 0 !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #7c5cff !important;
    box-shadow:
        0 -6px 30px rgba(0,0,0,0.5),
        0 0 0 3px rgba(124,92,255,0.18) !important;
}
[data-testid="stChatInput"] textarea {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    color: var(--text-0) !important;
    background: transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-3) !important;
}

/* ── Expanders (cards) ────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid var(--line) !important;
    border-radius: 12px !important;
    background: rgba(22, 30, 51, 0.55) !important;
    backdrop-filter: blur(6px) !important;
    transition: border-color 0.2s ease, transform 0.15s ease !important;
    overflow: hidden;
}
[data-testid="stExpander"]:hover {
    border-color: var(--line-strong) !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-0) !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    padding: 12px 14px !important;
}
[data-testid="stExpander"] summary:hover {
    background: rgba(28, 38, 63, 0.6) !important;
}

/* ── Dataframe ────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    border: 1px solid var(--line);
    overflow: hidden;
    background: rgba(15, 21, 37, 0.6);
}

/* ── Text inputs / select ─────────────────────────────────────── */
[data-baseweb="input"] input,
[data-baseweb="select"] > div,
.stTextInput input, .stTextArea textarea, .stNumberInput input {
    background: rgba(15, 21, 37, 0.85) !important;
    border: 1px solid var(--line-strong) !important;
    border-radius: 10px !important;
    color: var(--text-0) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {
    border-color: #7c5cff !important;
    box-shadow: 0 0 0 3px rgba(124,92,255,0.18) !important;
}

/* ── Radio + selectbox label tone ─────────────────────────────── */
.stRadio label, .stSelectbox label, .stTextInput label, .stFileUploader label {
    color: var(--text-1) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
}

/* ── File uploader ────────────────────────────────────────────── */
[data-testid="stFileUploader"] section {
    background: linear-gradient(135deg, rgba(124,92,255,0.05), rgba(236,72,153,0.04)) !important;
    border: 1.5px dashed var(--line-strong) !important;
    border-radius: 14px !important;
    padding: 18px !important;
    transition: all 0.18s ease !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #7c5cff !important;
    background: linear-gradient(135deg, rgba(124,92,255,0.1), rgba(236,72,153,0.08)) !important;
}

/* ── Progress bar ─────────────────────────────────────────────── */
[data-testid="stProgress"] > div > div > div > div {
    background: linear-gradient(90deg, #7c5cff, #ec4899) !important;
}

/* ── Metrics ──────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: rgba(22, 30, 51, 0.55);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 12px 16px;
}
[data-testid="stMetricValue"] {
    color: var(--text-0) !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-2) !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* ── Alerts (info/warn/success/error) ─────────────────────────── */
.stAlert {
    border-radius: 12px !important;
    border: 1px solid var(--line) !important;
    backdrop-filter: blur(6px) !important;
}

/* ── Code blocks ──────────────────────────────────────────────── */
code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}
code {
    background: rgba(22, 30, 51, 0.85) !important;
    padding: 2px 7px;
    border-radius: 5px;
    font-size: 0.85em;
    color: #c4b5fd !important;
    border: 1px solid var(--line);
}

/* ── Captions ─────────────────────────────────────────────────── */
.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--text-2) !important;
}

/* ── Spinner ──────────────────────────────────────────────────── */
[data-testid="stSpinner"] > div {
    border-top-color: #7c5cff !important;
    border-right-color: #ec4899 !important;
    border-bottom-color: #22d3ee !important;
}

/* ── Hero card on empty states ─────────────────────────────────── */
.hero-card {
    margin: 14px 0 18px 0;
    padding: 22px 26px;
    border-radius: 18px;
    background:
        radial-gradient(ellipse at top left, rgba(124,92,255,0.18) 0%, transparent 55%),
        radial-gradient(ellipse at bottom right, rgba(236,72,153,0.14) 0%, transparent 55%),
        rgba(22, 30, 51, 0.6);
    border: 1px solid var(--line-strong);
    backdrop-filter: blur(12px);
    box-shadow: 0 10px 36px rgba(0,0,0,0.35);
}
.hero-eyebrow {
    text-transform: uppercase;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    color: #c4b5fd;
    margin-bottom: 6px;
}
.hero-title {
    font-size: 1.4rem;
    font-weight: 800;
    line-height: 1.2;
    color: var(--text-0);
    margin-bottom: 6px;
}
.hero-title em {
    font-style: normal;
    background: var(--grad-3);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-body {
    color: var(--text-1);
    font-size: 0.95rem;
    line-height: 1.55;
    max-width: 80%;
}

/* ── Section header chip ─────────────────────────────────────── */
.section-head {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 14px 0 10px 0;
}
.section-head .dot {
    width: 8px; height: 8px;
    border-radius: 999px;
    background: var(--grad-1);
    box-shadow: 0 0 12px rgba(124,92,255,0.55);
}
.section-head h4 {
    margin: 0 !important;
    font-size: 1rem;
    color: var(--text-0);
}

/* ── Locked badge (used by guest history banner) ─────────────── */
.locked-banner {
    margin-top: 18px;
    padding: 26px 28px;
    border-radius: 18px;
    background:
        radial-gradient(ellipse at top left, rgba(168,85,247,0.35) 0%, transparent 60%),
        radial-gradient(ellipse at bottom right, rgba(236,72,153,0.3) 0%, transparent 60%),
        linear-gradient(135deg, #312e81 0%, #1e1b4b 100%);
    color: #fef9c3;
    font-weight: 600;
    font-size: 1.05rem;
    box-shadow: 0 12px 40px rgba(124,58,237,0.4);
    border: 1px solid rgba(168,85,247,0.5);
}
.locked-banner .sub {
    font-weight: 400;
    font-size: 0.92rem;
    color: #e9d5ff;
    margin-top: 6px;
    display: block;
}

/* ── Plotly chart container ──────────────────────────────────── */
.js-plotly-plot, .plotly {
    border-radius: 12px;
    background: rgba(15, 21, 37, 0.4);
    padding: 4px;
}

/* ── Misc polish ─────────────────────────────────────────────── */
hr { border-color: var(--line) !important; opacity: 0.6; margin: 14px 0 !important; }
::selection { background: rgba(124,92,255,0.4); color: #fff; }
</style>
""", unsafe_allow_html=True)

from ui.topbar import render_topbar
from ui.data_tab import render_data_tab
from ui.chat import render_chat_tab
from ui.explorer import render_explorer_tab
from ui.history import render_history_tab

render_topbar()

tab_data, tab_explore, tab_chat, tab_history = st.tabs(
    ["📁 Data", "🔍 Explorer", "💬 Chat", "📜 History"]
)
with tab_data:
    render_data_tab()
with tab_explore:
    render_explorer_tab()
with tab_chat:
    render_chat_tab()
with tab_history:
    render_history_tab()
