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
