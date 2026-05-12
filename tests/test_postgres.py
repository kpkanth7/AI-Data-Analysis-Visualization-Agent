import pytest
from unittest.mock import patch, MagicMock
from db.postgres import execute_sql, slugify

def test_execute_sql_blocks_write_statements():
    with pytest.raises(ValueError, match="Only SELECT"):
        execute_sql("INSERT INTO foo VALUES (1)")

def test_execute_sql_blocks_drop():
    with pytest.raises(ValueError, match="Only SELECT"):
        execute_sql("DROP TABLE users")

def test_execute_sql_blocks_update():
    with pytest.raises(ValueError, match="Only SELECT"):
        execute_sql("UPDATE users SET name='x'")

def test_slugify_basic():
    assert slugify("My Sales Data 2024.csv") == "my_sales_data_2024"
    assert slugify("revenue-report.xlsx") == "revenue_report"

def test_slugify_long_name():
    long = "a" * 100 + ".csv"
    assert len(slugify(long)) <= 63

def test_execute_sql_allows_select():
    from unittest.mock import patch, MagicMock
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchmany.return_value = [
        MagicMock(_mapping={"id": 1, "name": "test"})
    ]
    with patch("db.postgres.get_connection") as mock_ctx:
        mock_ctx.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
        result = execute_sql("SELECT * FROM datasets LIMIT 10")
        assert isinstance(result, list)

def test_slugify_edge_cases():
    assert slugify("___.csv") == "dataset"
    assert slugify("123data.csv").startswith("t_")
    result = slugify("a" * 70 + ".csv")
    assert len(result) <= 63
    assert not result.endswith("_")
