# AI Data Analyst Agent — Design Spec
**Date:** 2026-05-11  
**Stack:** OpenAI gpt-4o-mini · LangChain · PostgreSQL · Chroma · Streamlit · Plotly

---

## Overview

A production-quality AI data analyst agent. Users upload CSV/Excel datasets, ask natural-language questions, and receive SQL-backed answers with interactive Plotly charts. Designed to feel like a real product: streaming agent steps, multi-dataset awareness, anomaly detection, Excel export, guest rate-limiting, and pre-loaded demo datasets for showcasing.

Deployed target: Streamlit Cloud.

---

## Architecture

Single Streamlit monolith — no separate backend server. LangChain agent runs inline. PostgreSQL (local PG18, database `data_analyst`) stores all datasets as tables. Chroma (local, embedded) indexes metadata for semantic search. All state in Streamlit session state.

```
data-analysis-agent/
├── app.py                    # Streamlit entry point
├── agent/
│   ├── agent.py              # LangChain agent + streaming callbacks
│   ├── tools.py              # All 10 agent tools
│   ├── prompts.py            # System prompt + multi-subquery instructions
│   └── schema.py             # Pydantic output models
├── db/
│   ├── postgres.py           # PostgreSQL: dataset CRUD, SQL execution
│   └── vector_store.py       # Chroma: column/table semantic search
├── ui/
│   ├── sidebar.py            # Dataset manager, upload, guest quota display
│   ├── chat.py               # Chat UI + streaming agent steps
│   └── charts.py             # Plotly chart builders
├── core/
│   ├── dataset_profiler.py   # Auto-profile on upload
│   ├── anomaly.py            # Z-score + IQR anomaly/trend detection
│   └── exporter.py           # Excel/CSV export
├── data/
│   └── demo/                 # Pre-loaded demo CSVs
├── setup_db.py               # One-time DB + Chroma init
├── requirements.txt
└── .env
```

---

## Model

**OpenAI `gpt-4o-mini`**
- Temperature: 0 (deterministic for data analysis)
- Max tokens: 4096
- Cost: ~$0.15/1M input, $0.60/1M output
- Typical query: ~2000 input + 500 output ≈ $0.0006

---

## Database

**PostgreSQL 18** (existing local instance, user `pradhyumnakasula`)  
New database: `data_analyst`

Tables:
- `datasets` catalog: `(id, name, slug, row_count, columns_json, uploaded_at, is_demo)`
- One table per uploaded dataset, named by slugified filename
- Read-only SQL enforced in tools (no INSERT/UPDATE/DELETE/DROP allowed)

**Chroma** (embedded, no server)  
Collection: `dataset_metadata`  
Documents: one per column — `"table: sales_2024, column: revenue, type: float, description: inferred"`  
Used by `search_metadata` tool before writing SQL.

---

## Agent Tools

| Tool | Input | Output |
|---|---|---|
| `list_datasets` | none | All tables + schemas |
| `search_metadata` | natural language query | Matching columns/tables |
| `query_sql` | SQL string | Rows as list of dicts |
| `query_pandas` | pandas query expression | Filtered DataFrame rows |
| `compute_stats` | table, column, metrics list | Stats dict |
| `detect_anomalies` | table, column | Flagged rows + severity |
| `detect_trends` | table, date_col, value_col | Rolling avg, growth rates |
| `create_visualization` | chart type, data, config | Plotly figure JSON |
| `export_data` | DataFrame or SQL result | Excel file path |
| `save_session` | conversation + results | JSON log path |

SQL safety: whitelist-only — queries must be SELECT only. Blocked keywords: INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, TRUNCATE.

---

## Multi-Subquery Handling

System prompt instructs agent: detect compound questions (AND, ALSO, PLUS, multiple question marks). Decompose into numbered sub-tasks. Execute each sequentially. Label outputs `[1]`, `[2]`, etc. Generate separate charts per sub-question where appropriate.

Example:  
*"Show total revenue by region AND detect anomalies in daily sales AND export top 10 products"*  
→ 3 tool chains, 2 charts, 1 Excel download, all in one response.

---

## Streaming

`LangChainStreamingCallbackHandler` emits events per tool invocation:
- `on_tool_start` → shows "▶ Running: query_sql..."
- `on_tool_end` → shows row count / result preview
- `on_agent_finish` → renders final answer + charts

Streamlit: `st.empty()` containers updated in-place. Steps shown in a collapsible expander above the final answer bubble.

---

## Visualizations (Plotly)

All charts: dark theme, hover tooltips, zoom/pan/reset, PNG download button.

| Chart Type | Triggered When |
|---|---|
| Line + range selector | time series, trends, "over time" |
| Grouped/stacked bar | category comparison, "by region/product" |
| Histogram + box (side by side) | distribution, "spread of", "distribution" |
| Scatter + regression line | correlation, "relationship between" |
| Heatmap | cross-tab, correlation matrix |
| Pie/donut | composition, "percentage of", "share" |
| Scatter with red outliers | anomaly detection results |

Agent picks chart type based on query semantics. Can be overridden by user ("show as bar chart").

---

## UI Layout

### Sidebar
- Owner login (password from `st.secrets`, session persists until browser close)
- Dataset list: uploaded + demo datasets with row counts
- Upload widget (owner only / hidden for guest)
- Guest quota: "Guest session: X/5 queries used"
- Demo datasets section: expandable preview + schema for each

### Main Area — 3 tabs

**Chat tab**
- ChatGPT-style message bubbles
- Agent streaming steps in collapsible expander
- Plotly charts rendered inline below each answer
- Excel export button on results that have tabular data
- Input box pinned to bottom

**Explorer tab**
- Dropdown: select any loaded table
- Paginated data table (100 rows/page)
- Column filter + sort
- Export full table as Excel

**History tab**
- List of saved sessions (timestamp + first query)
- Click to re-load full conversation
- Re-run any past query against current data

---

## Guest Access Control

- No login required for guest
- Guest identified by Streamlit session state (resets on browser refresh = new session)
- Limits: 5 queries per session, demo datasets only, no upload, no history tab
- Owner: password in `st.secrets["owner_password"]`, stored in session state on match
- Guest quota counter shown in sidebar
- On quota exceeded: friendly message + invite to reload for new session

---

## Demo Datasets

Pre-loaded on `setup_db.py` run, flagged `is_demo=True` in catalog:

1. **ecommerce_sales** — orders, revenue, product category, region, date (~10k rows). Classic for trend + comparison queries.
2. **global_superstore** — Tableau's public dataset. Rich: sales, profit, discount, shipping, segment, category, country (~10k rows). Complex enough for tricky multi-part queries.

Both fully profiled + indexed in Chroma. Schema + sample rows visible to guests in sidebar before querying.

---

## Enhancements Beyond PDF

| Enhancement | Implementation |
|---|---|
| Multi-dataset support | All tables in Postgres, agent discovers via `list_datasets` |
| Vector semantic search | Chroma on column metadata before SQL generation |
| Interactive charts | Plotly (vs. static Matplotlib PNGs) |
| Streaming agent steps | LangChain callbacks → Streamlit live updates |
| Anomaly detection | Z-score + IQR with severity scoring |
| Trend detection | Rolling avg, period-over-period growth rate |
| Excel export | openpyxl, styled output |
| Guest rate limiting | Session state quota, demo-only mode |
| Dataset auto-profiler | Null %, unique counts, dtype inference, date detection |
| Data Explorer tab | Browse/filter/export any table without asking agent |
| Session history | Saved JSON logs, re-loadable |
| Owner/guest modes | Secrets-based auth, no external auth library |
| Deployment-ready | `.env` + `st.secrets` pattern, Streamlit Cloud compatible |

---

## Requirements

```
streamlit
langchain
langchain-openai
langchain-community
openai
psycopg2-binary
sqlalchemy
pandas
numpy
plotly
chromadb
sentence-transformers
openpyxl
python-dotenv
pydantic
scipy
```

---

## Environment Variables

```
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://pradhyumnakasula@localhost:5432/data_analyst
CHROMA_PERSIST_DIR=./chroma_db
```

Streamlit Cloud: stored in `st.secrets` / Streamlit secrets manager.
