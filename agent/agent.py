import os
import json
import threading

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

from agent.tools import ALL_TOOLS, set_session_context
from agent.prompts import SYSTEM_PROMPT
from agent.schema import AnalysisOutput

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


def run_agent(
    query: str,
    chat_history: list,
    is_owner: bool = False,
    session_id: str | None = None,
) -> tuple[AnalysisOutput | None, list[dict]]:
    """
    Returns (AnalysisOutput, steps).
    steps is a list of {tool, input, status, output_hint} dicts.
    Caller decides how to display them (collapsed expander).
    """
    set_session_context(is_owner=is_owner, session_id=session_id)

    executor = build_agent()
    accumulator = StepAccumulator()

    result = executor.invoke(
        {"input": query, "chat_history": chat_history},
        config={"callbacks": [accumulator]},
    )

    output_text = result.get("output", "")
    parser = PydanticOutputParser(pydantic_object=AnalysisOutput)
    try:
        analysis = parser.parse(output_text)
    except Exception:
        analysis = AnalysisOutput(answer=output_text)

    return analysis, accumulator.steps
