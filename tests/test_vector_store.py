import pytest
from unittest.mock import patch, MagicMock

def test_build_document_string():
    from db.vector_store import _build_doc
    profile_entry = {"dtype": "float64", "null_pct": 5.0, "unique_count": 100}
    doc = _build_doc("sales", "revenue", profile_entry)
    assert "sales" in doc
    assert "revenue" in doc
    assert "float64" in doc
    assert "5.0" in doc

def test_search_returns_empty_when_no_data():
    with patch("db.vector_store._get_collection") as mock_col:
        mock_instance = MagicMock()
        mock_instance.count.return_value = 0
        mock_col.return_value = mock_instance
        from db.vector_store import search_metadata
        results = search_metadata("revenue columns")
        assert results == []

def test_search_returns_results():
    with patch("db.vector_store._get_collection") as mock_col:
        mock_instance = MagicMock()
        mock_instance.count.return_value = 5
        mock_instance.query.return_value = {
            "documents": [["table: sales, column: revenue, type: float64, null_pct: 0%, unique_values: 100"]],
            "metadatas": [[{"table": "sales", "column": "revenue"}]],
        }
        mock_col.return_value = mock_instance
        from db.vector_store import search_metadata
        results = search_metadata("revenue")
        assert len(results) == 1
        assert results[0]["meta"]["table"] == "sales"

def test_index_dataset_calls_upsert():
    with patch("db.vector_store._get_collection") as mock_col:
        mock_instance = MagicMock()
        mock_col.return_value = mock_instance
        from db.vector_store import index_dataset_in_chroma
        profile = {
            "revenue": {"dtype": "float64", "null_pct": 0.0, "unique_count": 50},
            "region": {"dtype": "object", "null_pct": 0.0, "unique_count": 4},
        }
        index_dataset_in_chroma("sales", ["revenue", "region"], profile)
        mock_instance.upsert.assert_called_once()
        call_kwargs = mock_instance.upsert.call_args
        ids = call_kwargs[1]["ids"] if call_kwargs[1] else call_kwargs[0][0]
        assert "sales::revenue" in ids

def test_search_returns_empty_on_exception():
    with patch("db.vector_store._get_collection") as mock_col:
        mock_instance = MagicMock()
        mock_instance.count.return_value = 5
        mock_instance.query.side_effect = Exception("chroma error")
        mock_col.return_value = mock_instance
        from db.vector_store import search_metadata
        results = search_metadata("anything")
        assert results == []
