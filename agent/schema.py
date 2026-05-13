import json
from pydantic import BaseModel, Field, field_validator
from typing import Any, Optional


def _coerce_chart(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return None
    return v


class SubQueryResult(BaseModel):
    index: int
    question: str
    answer: str
    chart_type: Optional[str] = None
    chart_config: Optional[dict[str, Any]] = None
    sql_used: Optional[str] = None
    export_path: Optional[str] = None
    data_preview: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("chart_config", mode="before")
    @classmethod
    def coerce_chart_config(cls, v):
        return _coerce_chart(v)


class AnalysisOutput(BaseModel):
    answer: str
    chart_config: Optional[dict[str, Any]] = None
    data_preview: list[dict[str, Any]] = Field(default_factory=list)
    sql_used: Optional[str] = None
    sub_results: list[SubQueryResult] = Field(default_factory=list)
    datasets_used: list[str] = Field(default_factory=list)
    export_path: Optional[str] = None
    queries_used: list[str] = Field(default_factory=list)

    @field_validator("chart_config", mode="before")
    @classmethod
    def coerce_chart_config(cls, v):
        return _coerce_chart(v)


class DecomposedQuestions(BaseModel):
    questions: list[str]
    is_multi: bool


class FollowUpResult(BaseModel):
    is_followup: bool
    target_index: int   # 1-based index into last_sub_questions; -1 if not a follow-up
    rewritten_query: str  # self-contained query combining follow-up intent + original context
