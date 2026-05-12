import os
import json

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

from agent.tools import ALL_TOOLS
from agent.prompts import SYSTEM_PROMPT
from agent.schema import AnalysisOutput

load_dotenv()


class StreamlitStepHandler(BaseCallbackHandler):
    """Streams agent tool calls live into a Streamlit st.empty() container."""

    def __init__(self, container):
        self.container = container
        self.steps: list[str] = []

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs):
        name = serialized.get("name", "tool")
        self.steps.append(f"⏳ `{name}` ← {str(input_str)[:120]}")
        self.container.markdown("\n\n".join(self.steps))

    def on_tool_end(self, output: str, **kwargs):
        if self.steps:
            self.steps[-1] = self.steps[-1].replace("⏳", "✅")
            try:
                parsed = json.loads(output)
                if "total_rows" in parsed:
                    self.steps[-1] += f" → {parsed['total_rows']} rows"
                elif "anomaly_count" in parsed:
                    self.steps[-1] += f" → {parsed['anomaly_count']} anomalies"
                elif "export_path" in parsed:
                    self.steps[-1] += f" → saved to `{parsed['export_path']}`"
                elif "log_path" in parsed:
                    self.steps[-1] += f" → logged"
            except Exception:
                pass
        self.container.markdown("\n\n".join(self.steps))

    def on_agent_finish(self, finish, **kwargs):
        self.steps.append("🏁 Done")
        self.container.markdown("\n\n".join(self.steps))


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
    step_container,
) -> AnalysisOutput | None:
    executor = build_agent()
    handler = StreamlitStepHandler(step_container)

    result = executor.invoke(
        {"input": query, "chat_history": chat_history},
        config={"callbacks": [handler]},
    )

    output_text = result.get("output", "")
    parser = PydanticOutputParser(pydantic_object=AnalysisOutput)
    try:
        return parser.parse(output_text)
    except Exception:
        return AnalysisOutput(answer=output_text)
