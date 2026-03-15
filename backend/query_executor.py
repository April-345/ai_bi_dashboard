"""
backend/query_executor.py
"""

import os
import sqlite3
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Database path
# ---------------------------------------------------------------------------

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_DB_PATH = os.path.abspath(os.path.join(_BACKEND_DIR, "..", "data", "bmw.db"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execute_query(sql: str, db_path: str = _DB_PATH) -> pd.DataFrame:
    """
    Execute a SQL SELECT query and return results as a pandas DataFrame.

    Args:
        sql:     A valid SQLite SELECT query string.
        db_path: Path to the SQLite database file.

    Returns:
        pandas DataFrame with query results (empty DataFrame if no rows).
    """
    if not sql or not sql.strip():
        raise ValueError("sql must be a non-empty string.")

    if not sql.strip().upper().startswith("SELECT"):
        raise ValueError(f"Only SELECT queries are allowed. Got: {sql!r}")

    if not os.path.isfile(db_path):
        raise FileNotFoundError(
            f"Database not found at: {db_path}\n"
            "Run database.py first to initialise bmw.db."
        )

    try:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(sql, conn)
        return df

    except sqlite3.OperationalError as exc:
        print(f"[execute_query] SQL error: {exc}")
        print(f"[execute_query] Query was: {sql}")
        raise
    except sqlite3.Error as exc:
        print(f"[execute_query] Database error: {exc}")
        raise


# ---------------------------------------------------------------------------
# Test block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, _BACKEND_DIR)
    from sql_generator import generate_sql

    question = "What are the 10 cheapest diesel cars?"

    print(f"Question: {question}")

    sql = generate_sql(question)
    print(f"\nGenerated SQL: {sql}")

    df = execute_query(sql)
    print(f"\nResults:")
    print(df.to_string(index=False))