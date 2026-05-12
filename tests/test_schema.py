from agent.schema import AnalysisOutput, SubQueryResult


def test_analysis_output_defaults():
    out = AnalysisOutput(answer="Revenue is $1M")
    assert out.answer == "Revenue is $1M"
    assert out.sub_results == []
    assert out.export_path is None
    assert out.queries_used == []
    assert out.datasets_used == []


def test_sub_query_result():
    sub = SubQueryResult(index=1, question="total revenue", answer="$1M")
    assert sub.index == 1
    assert sub.chart_type is None
    assert sub.chart_config is None
    assert sub.data_preview == []


def test_analysis_output_with_sub_results():
    subs = [SubQueryResult(index=1, question="q1", answer="a1", chart_type="bar")]
    out = AnalysisOutput(answer="Combined answer", sub_results=subs)
    assert len(out.sub_results) == 1
    assert out.sub_results[0].chart_type == "bar"


def test_sub_query_result_with_chart_config():
    sub = SubQueryResult(
        index=2,
        question="trend",
        answer="revenue grew",
        chart_type="line",
        chart_config={"x": "date", "y": "revenue"},
    )
    assert sub.chart_config["x"] == "date"


def test_analysis_output_serializable():
    import json
    out = AnalysisOutput(
        answer="test",
        datasets_used=["sales"],
        queries_used=["SELECT * FROM sales"],
    )
    # must be JSON serializable
    d = json.loads(out.model_dump_json())
    assert d["answer"] == "test"
    assert d["datasets_used"] == ["sales"]
