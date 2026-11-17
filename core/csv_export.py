"""Spreadsheet-safe CSV serialization for typed dataset exports."""

import csv
from collections.abc import Iterable
from io import StringIO


def _csv_cell(value):
    if isinstance(value, str) and (
        value.startswith(("\t", "\r", "\n")) or value.lstrip().startswith(("=", "+", "-", "@"))
    ):
        return "'" + value
    return value


def csv_bytes(columns: Iterable[str], rows: Iterable[Iterable[object]]) -> bytes:
    output = StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow([_csv_cell(name) for name in columns])
    writer.writerows([_csv_cell(value) for value in row] for row in rows)
    return output.getvalue().encode("utf-8-sig")
