"""Desired safety contracts, not assertions that vulnerabilities should persist.

Known gaps use strict xfail limited to AssertionError. Run with --runxfail to
see the failing acceptance assertions. SQL and file sinks are always mocked.
"""
from unittest.mock import MagicMock, mock_open

import pandas as pd
import pytest

from agent import tools
from agent.schema import AnalysisOutput, SubQueryResult
from core import dataset_profiler
from db import postgres, vector_store
from ui import auth, chat, cleaning_wizard_ui


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="C01: SQL bypasses the session catalog")
@pytest.mark.parametrize("sql", [
    'SELECT * FROM owner_private',
    'SELECT * FROM guest_b_sales',
    'SELECT * FROM guest_a_sales JOIN guest_b_sales USING (id)',
    'WITH hidden AS (SELECT * FROM guest_b_sales) SELECT * FROM hidden',
    'SELECT * FROM guest_a_sales WHERE id IN (SELECT id FROM guest_b_sales)',
])
def test_foreign_sql_never_reaches_executor(monkeypatch, guest_catalog, sql):
    execute = MagicMock(return_value=[{"value": "FOREIGN_CANARY"}])
    monkeypatch.setattr(tools, "execute_sql", execute)
    result = tools.query_sql.invoke({"sql": sql})
    execute.assert_not_called()
    assert "FOREIGN_CANARY" not in result


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="C01: export guards table but executes unrelated SQL")
def test_export_cannot_read_a_foreign_query(monkeypatch, guest_catalog):
    read = MagicMock(return_value=pd.DataFrame({"secret": ["FOREIGN_CANARY"]}))
    export = MagicMock(return_value="synthetic.xlsx")
    monkeypatch.setattr(tools, "get_engine", lambda: object())
    monkeypatch.setattr(pd, "read_sql", read)
    monkeypatch.setattr(tools, "export_to_excel", export)
    tools.export_data.invoke({"table": "guest_a_sales", "sql": "SELECT * FROM guest_b_sales"})
    read.assert_not_called()
    export.assert_not_called()


def test_guarded_stats_reject_foreign_table(monkeypatch, guest_catalog):
    read = MagicMock()
    monkeypatch.setattr(pd, "read_sql_table", read)
    result = tools.compute_stats.invoke({"table": "guest_b_sales", "column": "amount"})
    assert result.startswith("Access denied:")
    read.assert_not_called()
    guest_catalog.assert_called_once_with(include_owner_only=False, session_id="guest_a")


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="C02: regex admits non-read-only SQL")
@pytest.mark.parametrize("sql", [
    "/* comment */ DELETE FROM synthetic_target",
    "WITH removed AS (DELETE FROM synthetic_target RETURNING *) SELECT * FROM removed",
    "SELECT * INTO synthetic_copy FROM synthetic_target",
    "SELECT * FROM synthetic_target FOR UPDATE",
])
def test_unsafe_sql_is_rejected_before_connection(monkeypatch, sql):
    connection = MagicMock()
    connection.return_value.__enter__.return_value.execute.return_value.fetchmany.return_value = []
    monkeypatch.setattr(postgres, "get_connection", connection)
    try:
        postgres.execute_sql(sql)
    except ValueError:
        pass
    connection.assert_not_called()


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="C03: filenames choose shared replacement tables")
@pytest.mark.parametrize("filename", ["sales.csv", "datasets.csv", "guest_usage.csv", "titanic.csv"])
def test_duplicate_filenames_do_not_replace_existing_data(monkeypatch, filename):
    writes = []
    monkeypatch.setattr(dataset_profiler, "get_engine", lambda: object())
    monkeypatch.setattr(pd.DataFrame, "to_sql", lambda df, name, engine, **kw: writes.append((name, kw)))
    monkeypatch.setattr(dataset_profiler, "register_dataset", MagicMock())
    monkeypatch.setattr(dataset_profiler, "index_dataset_in_chroma", MagicMock())
    for sid in ("guest_a", "guest_b"):
        dataset_profiler.ingest_dataframe(pd.DataFrame({"id": [1]}), filename, session_id=sid)
    assert writes[0][0] != writes[1][0], "Different workspaces selected the same physical table"
    assert all(name not in {"datasets", "guest_usage", "titanic"} for name, _ in writes)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="C04: retrieval has no authorization filter")
def test_foreign_metadata_is_not_returned(monkeypatch, guest_catalog):
    collection = MagicMock()
    collection.count.return_value = 1
    collection.query.return_value = {
        "documents": [["FOREIGN_CANARY column: salary"]],
        "metadatas": [[{"table": "guest_b_payroll", "column": "salary"}]],
    }
    monkeypatch.setattr(vector_store, "_get_collection", lambda: collection)
    result = tools.search_metadata_tool.invoke({"query": "salary"})
    assert "FOREIGN_CANARY" not in result


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="C06: model expressions reach DataFrame.query")
def test_model_filter_does_not_reach_python_expression_sink(monkeypatch, guest_catalog):
    df = pd.DataFrame({"amount": [1, 2]})
    query = MagicMock(return_value=df)
    monkeypatch.setattr(tools, "get_engine", lambda: object())
    monkeypatch.setattr(pd, "read_sql_table", lambda *args: df)
    monkeypatch.setattr(pd.DataFrame, "query", query)
    tools.query_pandas.invoke({"table": "guest_a_sales", "expression": "amount > 1"})
    query.assert_not_called()  # Harmless expression; no exploit is evaluated.


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="C07: model paths reach local file-open sink")
@pytest.mark.parametrize("path", ["/synthetic/unregistered.xlsx", "../../synthetic.xlsx"])
@pytest.mark.parametrize("multi", [False, True], ids=["single", "subquery"])
def test_unregistered_artifact_is_not_opened(monkeypatch, fake_st, path, multi):
    file_open = mock_open(read_data=b"SYNTHETIC_ONLY")
    monkeypatch.setattr(chat, "st", fake_st)
    monkeypatch.setattr(chat, "open", file_open, raising=False)
    analysis = AnalysisOutput(answer="Synthetic result", export_path=None if multi else path)
    if multi:
        analysis.sub_results = [SubQueryResult(index=1, question="Q", answer="A", export_path=path)]
    chat._render_result(analysis, "fixture", {})
    file_open.assert_not_called()


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="C08: sign-out retains privileged state")
def test_signout_discards_owner_results(monkeypatch, fake_st):
    monkeypatch.setattr(auth, "st", fake_st)
    monkeypatch.setattr(auth, "auto_save_if_nonempty", MagicMock())
    fake_st.session_state.update({
        "is_owner": True, "chat_history": [{"role": "assistant", "content": "OWNER_CANARY"}],
        "lc_history": ["OWNER_CANARY"], "chart_store": {"owner": "OWNER_CANARY"},
        "_cleaning": {"step": "commit"}, "_pdf_cache_1": b"OWNER_CANARY",
    })
    auth.switch_to_guest()
    assert not auth.is_owner()
    assert not fake_st.session_state.get("chat_history")
    assert not fake_st.session_state.get("lc_history")
    assert not fake_st.session_state.get("chart_store")
    assert not fake_st.session_state.get("_cleaning")
    assert "_pdf_cache_1" not in fake_st.session_state


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="C08: commit does not reauthorize owner")
def test_guest_cannot_commit_stale_owner_wizard(monkeypatch, fake_st):
    monkeypatch.setattr(auth, "st", fake_st)
    monkeypatch.setattr(cleaning_wizard_ui, "st", fake_st)
    ingest = MagicMock(return_value="synthetic")
    monkeypatch.setattr(cleaning_wizard_ui, "ingest_dataframe", ingest)
    state = {"df": pd.DataFrame({"id": [1]}), "filename": "synthetic.csv"}
    cleaning_wizard_ui._step_commit(state)
    ingest.assert_not_called()


def test_usage_outage_denies_new_query(monkeypatch, fake_st):
    monkeypatch.setattr(auth, "st", fake_st)
    fake_st.session_state["_fallback_queries"] = 100
    monkeypatch.setattr(auth, "get_guest_limits", MagicMock(side_effect=RuntimeError("synthetic outage")))
    assert auth.can_query() is False
