import os
import re
import json
import threading

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

from agent.tools import ALL_TOOLS, set_session_context
from agent.prompts import SYSTEM_PROMPT, DECOMPOSE_PROMPT, FOLLOWUP_PROMPT
from agent.schema import AnalysisOutput, SubQueryResult, DecomposedQuestions, FollowUpResult

load_dotenv()

# Thread-local storage for step accumulation (safe for concurrent Streamlit sessions)
_tl = threading.local()


class StepAccumulator(BaseCallbackHandler):
    """Collects agent tool steps silently; caller renders them after the response."""

    def __init__(self):
        self.steps: list[dict] = []

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs):
        self.steps.append({
            "tool": serialized.get("name", "tool"),
            "input": str(input_str)[:200],
            "status": "running",
            "output_hint": "",
        })

    def on_tool_end(self, output: str, **kwargs):
        if not self.steps:
            return
        self.steps[-1]["status"] = "done"
        try:
            parsed = json.loads(output)
            if "total_rows" in parsed:
                self.steps[-1]["output_hint"] = f"{parsed['total_rows']} rows"
            elif "anomaly_count" in parsed:
                self.steps[-1]["output_hint"] = f"{parsed['anomaly_count']} anomalies"
            elif "export_path" in parsed:
                self.steps[-1]["output_hint"] = "exported"
            elif "log_path" in parsed:
                self.steps[-1]["output_hint"] = "saved"
        except Exception:
            pass

    def on_tool_error(self, error, **kwargs):
        if self.steps:
            self.steps[-1]["status"] = "error"
            self.steps[-1]["output_hint"] = str(error)[:100]


def _get_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("OPENAI_API_KEY", "")
        except Exception:
            pass
    return key


def build_agent() -> AgentExecutor:
    parser = PydanticOutputParser(pydantic_object=AnalysisOutput)
    format_instructions = parser.get_format_instructions()

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=_get_openai_key(),
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]).partial(format_instructions=format_instructions)

    agent = create_tool_calling_agent(llm, ALL_TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=False,
        return_intermediate_steps=True,
        max_iterations=15,
        handle_parsing_errors=True,
    )


def decompose_query(query: str) -> list[str]:
    """Split multi-question query into independent sub-questions via LLM. Returns [query] if single."""
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=_get_openai_key())
        structured = llm.with_structured_output(DecomposedQuestions)
        result = structured.invoke(DECOMPOSE_PROMPT.format(query=query))
        if result.is_multi and len(result.questions) >= 2:
            return result.questions
    except Exception:
        pass
    return [query]


def detect_followup(query: str, last_sub_questions: list[str]) -> FollowUpResult:
    """Detect if query is a follow-up to a specific previous sub-question."""
    if not last_sub_questions:
        return FollowUpResult(is_followup=False, target_index=-1, rewritten_query=query)
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=_get_openai_key())
        structured = llm.with_structured_output(FollowUpResult)
        sub_q_list = "\n".join(f"[{i + 1}] {q}" for i, q in enumerate(last_sub_questions))
        result = structured.invoke(FOLLOWUP_PROMPT.format(sub_q_list=sub_q_list, query=query))
        if result.is_followup and not (1 <= result.target_index <= len(last_sub_questions)):
            return FollowUpResult(is_followup=False, target_index=-1, rewritten_query=query)
        return result
    except Exception:
        return FollowUpResult(is_followup=False, target_index=-1, rewritten_query=query)


def run_agent(
    query: str,
    chat_history: list,
    is_owner: bool = False,
    session_id: str | None = None,
    last_sub_questions: list[str] | None = None,
) -> tuple[AnalysisOutput | None, list[dict], list[str]]:
    """
    Returns (analysis, steps, used_sub_questions).

    used_sub_questions: list of sub-question strings dispatched this turn.
      - Empty list for single-question queries and follow-ups.
      - Non-empty list for multi-question queries (caller stores for follow-up detection).
    """
    set_session_context(is_owner=is_owner, session_id=session_id)

    # ── Follow-up detection ────────────────────────────────────────────────────
    followup = detect_followup(query, last_sub_questions or [])
    if followup.is_followup:
        sub_result, steps = run_sub_agent(
            followup.rewritten_query,
            followup.target_index,
            chat_history,
            is_owner,
            session_id,
        )
        merged = AnalysisOutput(
            answer=f"Follow-up on sub-question {followup.target_index}.",
            sub_results=[sub_result],
        )
        return merged, steps, []

    # ── Decompose ──────────────────────────────────────────────────────────────
    sub_questions = decompose_query(query)

    # ── Single-question path ───────────────────────────────────────────────────
    if len(sub_questions) == 1:
        executor = build_agent()
        accumulator = StepAccumulator()
        result = executor.invoke(
            {"input": sub_questions[0], "chat_history": chat_history},
            config={"callbacks": [accumulator]},
        )
        output_text = result.get("output", "")
        parser = PydanticOutputParser(pydantic_object=AnalysisOutput)
        analysis = _extract_analysis(output_text, parser)
        if analysis:
            _coerce_chart_configs(analysis)
            _auto_visualize_single(analysis, sub_questions[0])
        return analysis, accumulator.steps, []

    # ── Multi-question path ────────────────────────────────────────────────────
    all_steps: list[dict] = []
    sub_results: list[SubQueryResult] = []
    for i, sub_q in enumerate(sub_questions, start=1):
        sub_result, steps = run_sub_agent(sub_q, i, chat_history, is_owner, session_id)
        sub_results.append(sub_result)
        all_steps.extend(steps)

    merged = AnalysisOutput(
        answer=f"Answered {len(sub_results)} questions independently.",
        sub_results=sub_results,
    )
    return merged, all_steps, sub_questions


_CHART_KEYWORDS = [
    ("stacked_bar", ["stacked bar", "stacked-bar", "stack bar"]),
    ("pie", ["pie chart", "pie graph", "donut", "as a pie", "pie of"]),
    ("line", ["line chart", "line graph", "as a line", "line plot", "trend line"]),
    ("scatter", ["scatter plot", "scatter chart", "scatterplot"]),
    ("histogram", ["histogram", "distribution plot"]),
    ("heatmap", ["heatmap", "heat map", "correlation matrix"]),
    ("box", ["box plot", "boxplot", "box-and-whisker"]),
    ("bar", ["bar chart", "bar graph", "as a bar", "bar plot"]),
]


def _detect_chart_preference(query: str) -> str | None:
    """Extract explicit chart-type request from text. None if none."""
    if not query:
        return None
    q = query.lower()
    for ctype, keywords in _CHART_KEYWORDS:
        for kw in keywords:
            if kw in q:
                return ctype
    return None



def _infer_chart_config(rows: list[dict], forced_type: str | None = None) -> dict | None:
    """Infer chart_type/x/y from row shape. Return None if not chartable."""
    if not rows or len(rows) < 2:
        return None
    cols = list(rows[0].keys())
    if len(cols) < 2:
        return None

    def is_num(v):
        if isinstance(v, bool):
            return False
        if isinstance(v, (int, float)):
            return True
        try:
            float(str(v).replace(",", ""))
            return True
        except (ValueError, TypeError):
            return False

    numeric_cols, cat_cols, date_cols = [], [], []
    for c in cols:
        vals = [r.get(c) for r in rows if r.get(c) is not None]
        if not vals:
            continue
        name_l = c.lower()
        if any(k in name_l for k in ("date", "year", "month", "time", "_at", "day")):
            date_cols.append(c)
        elif sum(is_num(v) for v in vals) / len(vals) >= 0.8:
            numeric_cols.append(c)
        else:
            cat_cols.append(c)

    if not numeric_cols:
        return None

    y = numeric_cols[0]
    x_default = date_cols[0] if date_cols else (cat_cols[0] if cat_cols else (numeric_cols[1] if len(numeric_cols) >= 2 else None))

    if forced_type:
        cfg = {"chart_type": forced_type, "data": rows[:500], "y": y, "title": ""}
        if forced_type == "histogram":
            cfg["x"] = y
        elif forced_type == "heatmap":
            pass
        elif forced_type == "scatter" and len(numeric_cols) >= 2:
            cfg["x"], cfg["y"] = numeric_cols[0], numeric_cols[1]
        else:
            cfg["x"] = x_default
        return cfg

    if date_cols:
        return {"chart_type": "line", "data": rows[:500], "x": date_cols[0], "y": y, "title": ""}
    if cat_cols:
        x = cat_cols[0]
        if len(rows) <= 5 and len(numeric_cols) == 1 and len(cat_cols) == 1:
            return {"chart_type": "pie", "data": rows[:500], "x": x, "y": y, "title": ""}
        return {"chart_type": "bar", "data": rows[:500], "x": x, "y": y, "title": ""}
    if len(numeric_cols) >= 2:
        return {"chart_type": "scatter", "data": rows[:500], "x": numeric_cols[0], "y": numeric_cols[1], "title": ""}
    return None


def _auto_visualize_single(analysis: AnalysisOutput, sub_question: str) -> None:
    """Apply chart preference from sub_question text and fill in missing chart_config.

    Precedence:
    1. Agent produced chart_config + user specified type → override type only, keep rest
    2. Agent produced chart_config, no user pref → keep as-is
    3. User specified type, no agent chart → infer from data with forced type
    4. No pref, no chart, data present → auto-infer from data shape
    5. No data → nothing
    """
    pref = _detect_chart_preference(sub_question)
    existing = analysis.chart_config
    rows = analysis.data_preview

    if pref:
        if existing and isinstance(existing, dict):
            cfg = dict(existing)
            cfg["chart_type"] = pref
            if not cfg.get("data") and rows:
                cfg["data"] = rows[:500]
            analysis.chart_config = cfg
        else:
            analysis.chart_config = _infer_chart_config(rows or [], forced_type=pref)
    elif not existing and rows:
        analysis.chart_config = _infer_chart_config(rows)


def run_sub_agent(
    sub_question: str,
    index: int,
    lc_history: list,
    is_owner: bool,
    session_id: str | None,
) -> tuple[SubQueryResult, list[dict]]:
    """Run one independent agent invocation for a single sub-question."""
    set_session_context(is_owner=is_owner, session_id=session_id)
    executor = build_agent()
    accumulator = StepAccumulator()

    result = executor.invoke(
        {"input": sub_question, "chat_history": lc_history},
        config={"callbacks": [accumulator]},
    )

    output_text = result.get("output", "")
    parser = PydanticOutputParser(pydantic_object=AnalysisOutput)
    analysis = _extract_analysis(output_text, parser)

    if analysis:
        _coerce_chart_configs(analysis)
        _auto_visualize_single(analysis, sub_question)

    return SubQueryResult(
        index=index,
        question=sub_question,
        answer=analysis.answer if analysis else "No answer returned for this sub-question.",
        chart_config=analysis.chart_config if analysis else None,
        data_preview=analysis.data_preview if analysis else [],
        sql_used=analysis.sql_used if analysis else None,
        export_path=analysis.export_path if analysis else None,
    ), accumulator.steps


def _coerce_chart_configs(analysis: AnalysisOutput) -> None:
    """If the LLM embedded chart_config as a JSON string instead of a dict, decode it."""
    if isinstance(analysis.chart_config, str):
        try:
            analysis.chart_config = json.loads(analysis.chart_config)
        except Exception:
            analysis.chart_config = None
    for sub in analysis.sub_results or []:
        if isinstance(sub.chart_config, str):
            try:
                sub.chart_config = json.loads(sub.chart_config)
            except Exception:
                sub.chart_config = None


def _extract_analysis(output_text: str, parser: PydanticOutputParser) -> AnalysisOutput:
    # 1. Direct parse
    try:
        return parser.parse(output_text)
    except Exception:
        pass

    # 2. JSON from ```json ... ``` block — use bracket matching, not regex, for nested objects
    fence = re.search(r"```(?:json)?\s*(\{)", output_text, re.DOTALL)
    if fence:
        start = fence.start(1)
        depth = 0
        for i, c in enumerate(output_text[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return parser.parse(output_text[start : i + 1])
                    except Exception:
                        break

    # 3. Outermost { } in text
    depth, start = 0, -1
    for i, c in enumerate(output_text):
        if c == "{":
            depth += 1
            if start == -1:
                start = i
        elif c == "}":
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    return parser.parse(output_text[start : i + 1])
                except Exception:
                    start = -1

    # 4. Salvage chart_config with proper bracket matching + strip code fences from prose
    chart_config = None
    m2 = re.search(r'"chart_config"\s*:\s*(\{)', output_text)
    if m2:
        try:
            start = m2.start(1)
            depth = 0
            for i, c in enumerate(output_text[start:], start):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        chart_config = json.loads(output_text[start : i + 1])
                        break
        except Exception:
            pass
    clean = re.sub(r"```.*?```", "", output_text, flags=re.DOTALL).strip()
    return AnalysisOutput(answer=clean, chart_config=chart_config)
