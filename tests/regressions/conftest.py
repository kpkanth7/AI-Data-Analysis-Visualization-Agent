"""Offline W0 fixtures. Unstubbed external I/O must fail, including under xfail."""
import socket
from unittest.mock import MagicMock

import pandas as pd
import pytest

from agent import tools
from core import dataset_profiler
from db import postgres, vector_store


@pytest.fixture(autouse=True)
def offline_boundaries(monkeypatch, tmp_path):
    def blocked(*args, **kwargs):
        pytest.fail("W0 fixture reached unstubbed external I/O", pytrace=False)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(postgres, "get_connection", blocked)
    monkeypatch.setattr(postgres, "get_engine", blocked)
    monkeypatch.setattr(vector_store, "_get_collection", blocked)
    monkeypatch.setattr(tools, "get_engine", blocked)
    monkeypatch.setattr(dataset_profiler, "get_engine", blocked)
    monkeypatch.setattr(pd, "read_sql", blocked)
    monkeypatch.setattr(pd, "read_sql_table", blocked)
    monkeypatch.setattr(pd.DataFrame, "to_sql", blocked)
    # Thread-local context is shared with the original test suite; restore it.
    old_context = tools._get_ctx()
    tools.set_session_context(session_id="guest_a")
    yield
    tools.set_session_context(**old_context)


@pytest.fixture
def fake_st():
    st = MagicMock()
    st.session_state = {"is_owner": False, "guest_session_id": "guest_a"}
    st.columns.side_effect = lambda spec, **kw: [
        MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))
    ]
    st.button.return_value = False
    return st


@pytest.fixture
def guest_catalog(monkeypatch):
    catalog = MagicMock(return_value=[{
        "slug": "guest_a_sales", "session_id": "guest_a", "is_demo": False,
        "owner_only": False,
    }])
    monkeypatch.setattr(tools, "list_datasets_from_db", catalog)
    return catalog
