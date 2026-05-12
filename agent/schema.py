from pydantic import BaseModel, Field
from typing import Any, Optional


class SubQueryResult(BaseModel):
    index: int
    question: str
    answer: str
    chart_type: Optional[str] = None
    chart_config: Optional[dict[str, Any]] = None
    sql_used: Optional[str] = None
    export_path: Optional[str] = None
    data_preview: list[dict[str, Any]] = Field(default_factory=list)


class AnalysisOutput(BaseModel):
    answer: str
    chart_config: Optional[dict[str, Any]] = None  # top-level chart for single-question responses
    data_preview: list[dict[str, Any]] = Field(default_factory=list)  # top-level table preview
    sql_used: Optional[str] = None
    sub_results: list[SubQueryResult] = Field(default_factory=list)
    datasets_used: list[str] = Field(default_factory=list)
    export_path: Optional[str] = None
    queries_used: list[str] = Field(default_factory=list)
