import os
import sys
import sqlite3

import chardet
import pandas as pd

# Fix Windows terminal encoding so all print() output uses UTF-8
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Resolve paths relative to this file's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "data", "BMW Vehicle Inventory_clean.csv")
DB_PATH  = os.path.join(BASE_DIR, "..", "data", "bmw.db")


def _detect_encoding(path: str, sample_bytes: int = 65536) -> str:
    """
    Sniff the file encoding using chardet.
    Falls back to 'utf-8' if detection is inconclusive.
    """
    with open(path, "rb") as f:
        raw = f.read(sample_bytes)
    result = chardet.detect(raw)
    encoding = result.get("encoding") or "utf-8"
    confidence = result.get("confidence", 0)
    print(f"Detected encoding: {encoding!r}  (confidence: {confidence:.0%})")
    return encoding


def _read_csv_safe(path: str) -> pd.DataFrame:
    """
    Try reading the CSV with auto-detected encoding first,
    then fall back through common encodings if that fails.
    """
    fallbacks = ["utf-8-sig", "latin-1", "cp1252", "utf-16"]

    detected = _detect_encoding(path)
    for enc in [detected] + fallbacks:
        try:
            df = pd.read_csv(path, encoding=enc)
            # Reject obviously corrupt reads (single column with binary gunk)
            if len(df.columns) == 1 and df.columns[0].startswith("bplist"):
                raise ValueError("File appears to be a binary plist/WebArchive, not a CSV.")
            print(f"Successfully read with encoding: {enc!r}")
            return df
        except (UnicodeDecodeError, ValueError) as e:
            print(f"  [{enc}] failed: {e}")

    raise RuntimeError(
        f"Could not read '{path}' as a CSV with any known encoding.\n"
        "The file may be a Safari WebArchive (.webarchive) saved with a .csv extension.\n"
        "Fix: open the file in Excel or a text editor and re-save it as plain CSV (UTF-8)."
    )


def load_csv_to_db(csv_path: str = CSV_PATH, db_path: str = DB_PATH) -> None:
    """
    Read the BMW Vehicle Inventory CSV and load it into a SQLite database
    as a table named 'cars'. Creates or replaces the database at db_path.
    """
    # -- 1. Read CSV ----------------------------------------------------------
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print(f"Reading CSV from: {csv_path}")
    df = _read_csv_safe(csv_path)
    df = df.loc[:, ~df.columns.str.contains("^unnamed", case=False)]
    # Normalise column names: lowercase, spaces -> underscores, strip extras
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )

    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")
    print(f"Columns: {list(df.columns)}")

    # -- 2. Write to SQLite ---------------------------------------------------
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    print(f"Saving database to: {db_path}")
    with sqlite3.connect(db_path) as conn:
        df.to_sql(
            name="cars",
            con=conn,
            if_exists="replace",  # overwrite on each reload
            index=False,
        )
        row_count = conn.execute("SELECT COUNT(*) FROM cars").fetchone()[0]
        print(f"[OK] Table 'cars' created with {row_count:,} rows.")


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Return a SQLite connection to bmw.db.
    Raises RuntimeError if the database has not been initialised yet.
    """
    if not os.path.exists(db_path):
        raise RuntimeError(
            f"Database not found at '{db_path}'. "
            "Run load_csv_to_db() first to initialise it."
        )
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # enables column-name access on rows
    return conn


def get_schema(db_path: str = DB_PATH) -> str:
    """
    Return a human-readable schema string for the 'cars' table.
    Useful for feeding context into an LLM SQL-generator prompt.
    """
    with get_connection(db_path) as conn:
        cursor = conn.execute("PRAGMA table_info(cars)")
        columns = cursor.fetchall()

    lines = ["Table: cars", "Columns:"]
    for col in columns:
        lines.append(f"  - {col['name']} ({col['type']})")
    return "\n".join(lines)


if __name__ == "__main__":
    load_csv_to_db()
    print()
    print(get_schema())
