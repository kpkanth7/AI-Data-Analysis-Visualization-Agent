"""Outages must remain recoverable without admitting new paid work."""
import io
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from streamlit.testing.v1 import AppTest

from ui import auth, chat, data_tab, explorer, topbar, history


APP_PATH = Path(__file__).resolve().parents[2] / "app.py"
CHAT_APP = """
from ui.auth import init_session_state
from ui.chat import render_chat_tab
init_session_state()
render_chat_tab()
"""
LIMITS = {"queries_remaining": 10, "uploads_remaining": 5, "can_query": True, "can_upload": True}
CATALOG = [{"name": "Synthetic sales", "slug": "synthetic_sales", "row_count": 2,
            "columns_json": [{"name": "amount", "dtype": "int64"}], "is_demo": True}]


@pytest.fixture
def services(monkeypatch):
    limits = MagicMock(return_value=LIMITS.copy())
    catalog = MagicMock(return_value=[])
    monkeypatch.setattr(auth, "get_browser_fingerprint", lambda: "synthetic_guest")
    monkeypatch.setattr(auth, "get_guest_limits", limits)
    monkeypatch.setattr(topbar, "cleanup_old_guest_datasets", MagicMock())
    for module in (data_tab, explorer, chat):
        monkeypatch.setattr(module, "list_datasets_from_db", catalog)
    return limits, catalog


def test_missing_identity_denies_queries_and_uploads(monkeypatch, fake_st):
    fake_st.session_state.pop("guest_session_id")
    monkeypatch.setattr(auth, "st", fake_st)
    usage = MagicMock()
    monkeypatch.setattr(auth, "get_guest_limits", usage)
    assert not auth.can_query() and not auth.can_upload()
    assert auth.get_cached_limits()["queries_remaining"] is None
    usage.assert_not_called()


def test_app_outage_is_sanitized_and_retry_recovers(services):
    limits, catalog = services
    catalog.side_effect = ModuleNotFoundError("PRIVATE_CONNECTION_CANARY")
    limits.side_effect = RuntimeError("PRIVATE_USAGE_CANARY")
    app = AppTest.from_file(str(APP_PATH)).run(timeout=15)
    assert not app.exception
    assert len(app.error) == 3
    assert all("Datasets are temporarily unavailable" in item.value for item in app.error)
    visible = " ".join(item.value for group in (app.markdown, app.caption, app.error, app.warning) for item in group)
    assert "CANARY" not in visible and "10/10" not in visible
    assert "Usage unavailable" in visible
    assert len(app.get("file_uploader")) == 0
    assert app.chat_input[0].disabled

    catalog.side_effect = None
    limits.side_effect = None
    app.button(key="retry_data").click().run(timeout=15)
    assert not app.exception and not app.error
    assert len(app.get("file_uploader")) == 1


@pytest.mark.parametrize("outage", ["usage", "datasets", "admission"])
def test_queued_question_is_preserved_and_never_auto_replayed(monkeypatch, services, outage):
    limits, catalog = services
    catalog.return_value = CATALOG
    if outage == "usage":
        limits.side_effect = RuntimeError("synthetic outage")
    elif outage == "datasets":
        catalog.side_effect = RuntimeError("synthetic outage")
    admission = MagicMock(return_value=outage != "admission")
    monkeypatch.setattr(chat, "can_query", admission)
    agent = MagicMock()
    record = MagicMock()
    monkeypatch.setattr(chat, "run_agent", agent)
    monkeypatch.setattr(chat, "record_query", record)
    monkeypatch.setattr(chat, "_render_pdf_export_button", lambda _: None)
    app = AppTest.from_string(CHAT_APP)
    saved = [{"role": "assistant", "content": "Previously computed result"}]
    app.session_state["chat_history"] = saved.copy()
    app.session_state["_pending_query"] = "Summarize synthetic sales"
    app.run(timeout=15)
    assert not app.exception
    assert app.session_state["chat_history"] == saved
    assert app.session_state["_unsent_query"] == "Summarize synthetic sales"
    assert app.button(key="retry_question").disabled
    assert app.chat_input[0].disabled
    agent.assert_not_called()
    record.assert_not_called()

    limits.side_effect = None
    catalog.side_effect = None
    admission.return_value = True
    app.run(timeout=15)
    assert not app.exception
    assert not app.button(key="retry_question").disabled
    agent.assert_not_called()  # Recovery requires an explicit Retry question click.


def test_reset_chat_discards_queued_and_unsent_questions(monkeypatch, fake_st):
    monkeypatch.setattr(auth, "st", fake_st)
    fake_st.session_state.update({"_pending_query": "queued", "_unsent_query": "held"})
    auth.reset_chat_state()
    assert "_pending_query" not in fake_st.session_state
    assert "_unsent_query" not in fake_st.session_state


def test_guest_upload_rechecks_usage_before_ingestion(monkeypatch, fake_st):
    upload = io.BytesIO(b"amount\n1\n")
    upload.name = "synthetic.csv"
    monkeypatch.setattr(data_tab, "st", fake_st)
    monkeypatch.setattr(data_tab, "get_cached_limits", lambda: {"available": True})
    monkeypatch.setattr(data_tab, "can_upload", lambda: False)
    status = MagicMock()
    monkeypatch.setattr(data_tab, "render_service_unavailable", status)
    ingest = MagicMock()
    monkeypatch.setattr(data_tab, "ingest_dataframe", ingest)
    fake_st.file_uploader.return_value = upload
    fake_st.button.return_value = True
    data_tab._render_upload_section(False, "synthetic_guest", 0)
    ingest.assert_not_called()
    status.assert_called_once_with("Usage", "upload_admission")


def test_explorer_outage_blocks_model_construction(monkeypatch):
    model = MagicMock()
    monkeypatch.setattr("langchain_openai.ChatOpenAI", model)
    monkeypatch.setattr(explorer, "can_query", lambda: False)
    with pytest.raises(PermissionError, match="usage"):
        explorer._nl_to_sql_where("above ten", ["amount"], "synthetic_sales")
    model.assert_not_called()


@pytest.mark.parametrize("fails", [False, True])
def test_explorer_records_attempted_model_call(monkeypatch, fails):
    model = MagicMock()
    model.return_value.invoke.return_value.content = "amount > 10"
    if fails:
        model.return_value.invoke.side_effect = RuntimeError("synthetic provider outage")
    monkeypatch.setattr("langchain_openai.ChatOpenAI", model)
    monkeypatch.setattr(explorer, "can_query", lambda: True)
    record = MagicMock()
    monkeypatch.setattr(explorer, "record_query", record)
    if fails:
        with pytest.raises(RuntimeError, match="synthetic provider"):
            explorer._nl_to_sql_where("above ten", ["amount"], "synthetic_sales")
    else:
        assert explorer._nl_to_sql_where("above ten", ["amount"], "synthetic_sales") == "amount > 10"
    record.assert_called_once()


@pytest.mark.parametrize("entry", ["topbar", "history"])
def test_signin_entry_authenticates_and_clears_password(monkeypatch, services, entry):
    monkeypatch.setattr(auth, "try_owner_login", lambda password: password == "synthetic_password")
    monkeypatch.setattr(history, "list_sessions", lambda: [])
    app = AppTest.from_file(str(APP_PATH)).run(timeout=15)
    assert not app.exception
    password = app.text_input(key=f"{entry}_owner_pw")
    password.set_value("incorrect")
    submit = next(button for button in app.button if entry in (button.key or "") and button.label == "Sign in")
    submit.click().run(timeout=15)
    assert not app.exception
    assert not app.session_state["is_owner"]
    assert any("Incorrect password" in item.value for item in app.error)
    assert app.session_state[f"{entry}_owner_pw"] == ""

    app.text_input(key=f"{entry}_owner_pw").set_value("synthetic_password")
    next(button for button in app.button if entry in (button.key or "") and button.label == "Sign in").click().run(timeout=15)
    assert not app.exception
    assert app.session_state["is_owner"]
    assert any("No sessions saved yet" in item.value for item in app.info)
    assert app.session_state.filtered_state.get(f"{entry}_owner_pw", "") == ""


def test_chat_empty_state_has_grounded_context_and_enabled_composer(services):
    _, catalog = services
    catalog.return_value = CATALOG
    app = AppTest.from_string(CHAT_APP).run(timeout=15)
    assert not app.exception
    assert not app.chat_input[0].disabled
    visible_text = " ".join(item.value for item in app.text)
    assert "Synthetic sales" in visible_text
    assert "How many rows are in synthetic_sales?" in visible_text


def test_chat_without_datasets_explains_disabled_composer(services):
    app = AppTest.from_string(CHAT_APP).run(timeout=15)
    assert not app.exception
    assert app.chat_input[0].disabled
    assert any("No datasets are available yet" in item.value for item in app.info)
