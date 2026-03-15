"""
backend/sql_generator.py
------------------------
Converts natural-language questions into SQLite SELECT queries.
Supports dynamic schemas — pass a custom schema string to generate_sql()
to work with any uploaded dataset. Falls back to the BMW schema if none given.
"""

import logging
import os
import re
import time

from groq import Groq, APIStatusError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load .env manually
# ---------------------------------------------------------------------------

def _load_env_file() -> None:
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    for path in [os.path.join(project_root, ".env"), os.path.join(script_dir, ".env")]:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key   = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and value:
                        os.environ[key] = value
            return

_load_env_file()


# ---------------------------------------------------------------------------
# Default BMW schema (used when no dataset is uploaded)
# ---------------------------------------------------------------------------

BMW_SCHEMA = """
Table: cars
Columns:
  - model        TEXT    (BMW model name, e.g. '320i', 'X5', 'M3')
  - year         REAL    (manufacture year, e.g. 2019.0)
  - price        REAL    (listing price in GBP, e.g. 25995.0)
  - transmission TEXT    (e.g. 'Manual', 'Automatic', 'Semi-Auto')
  - mileage      REAL    (odometer in miles, e.g. 12500.0)
  - fueltype     TEXT    (e.g. 'Petrol', 'Diesel', 'Hybrid', 'Electric')
  - tax          REAL    (annual road tax in GBP, e.g. 145.0)
  - mpg          REAL    (miles per gallon, e.g. 55.4)
  - enginesize   REAL    (engine displacement in litres, e.g. 2.0)
""".strip()


# ---------------------------------------------------------------------------
# System prompt — strict, example-driven, alias-safe
# ---------------------------------------------------------------------------

def _build_system_prompt(schema: str, table_name: str) -> str:
    return f"""You are an expert SQLite query generator for a business intelligence dashboard.

DATABASE SCHEMA
---------------
{schema}

YOUR TASK
---------
Convert the user's natural language question into a single valid SQLite SELECT query.

STRICT RULES — follow every rule exactly:

1.  OUTPUT ONLY the raw SQL query. No explanations, no markdown, no code fences, no comments.
2.  The query MUST start with SELECT.
3.  Query ONLY the table named "{table_name}". Never reference any other table.
4.  Use ONLY column names that appear in the schema above. Never invent column names.
5.  Every aggregate function MUST include parentheses with its argument:
      CORRECT:   COUNT(*), COUNT(col), AVG(price), MAX(mileage), SUM(tax), MIN(year)
      INCORRECT: COUNT *, AVG price, MAX mileage   <- these are syntax errors
6.  Every aggregate expression MUST have a descriptive alias using AS.
    CRITICAL: NEVER use a SQL reserved word as an alias. Good alias examples:
      COUNT(*) AS total_count     <- CORRECT
      COUNT(*) AS num_records     <- CORRECT
      COUNT(*) AS count           <- WRONG  ("count" is a reserved word)
      AVG(price) AS avg_price     <- CORRECT
      AVG(price) AS avg           <- WRONG  ("avg" is a reserved word)
      MAX(mileage) AS max_mileage <- CORRECT
      MAX(mileage) AS max         <- WRONG  ("max" is a reserved word)
7.  Do NOT add a trailing semicolon.
8.  For non-aggregate queries add LIMIT 100 unless the user specifies a number.
9.  Use LOWER(col) LIKE '%value%' for case-insensitive text filtering.
10. NEVER use: INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, REPLACE.
11. If GROUP BY is used, every non-aggregate column in SELECT must appear in GROUP BY.

CORRECT OUTPUT EXAMPLES:
  SELECT fueltype, COUNT(*) AS total_count FROM {table_name} GROUP BY fueltype
  SELECT model, AVG(price) AS avg_price FROM {table_name} GROUP BY model ORDER BY avg_price DESC LIMIT 10
  SELECT model, price, mileage FROM {table_name} ORDER BY price DESC LIMIT 10
  SELECT transmission, AVG(mpg) AS avg_mpg, COUNT(*) AS num_records FROM {table_name} GROUP BY transmission
  SELECT year, COUNT(*) AS total_count, AVG(price) AS avg_price FROM {table_name} GROUP BY year ORDER BY year

Now generate the SQL query. Return ONLY the SQL, nothing else.""".strip()


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class SQLGenerationError(Exception):
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "\nGROQ_API_KEY not found.\n"
            "Your .env file must contain:\n"
            "    GROQ_API_KEY=gsk_...\n"
        )
    return key


# All SQL reserved words — used to detect bad aliases and repair corrupted SQL.
# Aggregate names are included because aliasing as "count"/"avg"/etc. triggers
# the COUNT(*) AS count(FROM) corruption that caused repeated failures.
_SQL_RESERVED = frozenset([
    "SELECT", "FROM", "WHERE", "GROUP", "ORDER", "HAVING", "LIMIT", "OFFSET",
    "JOIN", "INNER", "LEFT", "RIGHT", "OUTER", "CROSS", "ON", "USING",
    "AS", "AND", "OR", "NOT", "IN", "IS", "NULL", "BETWEEN", "LIKE", "GLOB",
    "DISTINCT", "BY", "CASE", "WHEN", "THEN", "ELSE", "END", "IF",
    "SET", "INTO", "VALUES", "TABLE", "INDEX", "VIEW", "TRIGGER",
    # Aggregate names — never valid as aliases
    "COUNT", "MAX", "MIN", "AVG", "SUM",
])

# DML/DDL keywords that must never appear in a generated query
_UNSAFE_KEYWORDS = [
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER",
    "TRUNCATE", "CREATE", "REPLACE", "ATTACH", "DETACH",
]


def _repair(sql: str) -> str:
    """
    Repair  identifier(RESERVED_KEYWORD)  corruption.

    The old _clean() regex produced output like:
        COUNT(*) AS count(FROM) data GROUP BY fueltype
    because it matched the alias name "count" as an aggregate keyword and
    wrapped the following "FROM" in parentheses.

    This function undoes that damage unconditionally, so the pipeline
    self-heals regardless of which version of this file is installed.

    Examples:
        count(FROM)   ->  count FROM
        alias(GROUP)  ->  alias GROUP
        total(WHERE)  ->  total WHERE
    """
    _alt = "|".join(sorted(_SQL_RESERVED, key=len, reverse=True))
    return re.sub(
        rf'\b([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*({_alt})\b\s*\)',
        lambda m: f"{m.group(1)} {m.group(2)}",
        sql,
        flags=re.IGNORECASE,
    )


def _clean(raw: str) -> str:
    """
    Clean raw LLM output into a bare SQL string:
      1. Strip markdown fences and trailing semicolons
      2. Repair any word(RESERVED) corruption (self-healing)
      3. Fix bare aggregate syntax: MAX price -> MAX(price)
         Safe guards: skip when preceded by AS or next token is a reserved word
    """
    # 1. Strip markdown fences and trailing semicolons
    raw = re.sub(r"^```(?:sql)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$",          "", raw,         flags=re.IGNORECASE)
    raw = raw.strip().rstrip(";").strip()

    # 2. Repair any word(RESERVED) corruption before doing anything else
    raw = _repair(raw)

    # 3. Fix bare aggregate syntax using a token-by-token scan.
    #    Regex look-behinds of variable length are not supported in Python's re,
    #    so we walk the list explicitly for precise prev/next context.
    _AGG = frozenset(["MAX", "MIN", "AVG", "SUM", "COUNT"])

    tokens     = raw.split()
    out        = []
    prev_upper = ""   # upper-cased core of the previous token

    i = 0
    while i < len(tokens):
        tok      = tokens[i]
        tok_core = tok.rstrip(",()")   # strip trailing punctuation for keyword lookup
        tok_up   = tok_core.upper()

        is_bare_agg = (
            tok_up in _AGG         # it's an aggregate keyword
            and "(" not in tok     # no parenthesis already attached
            and prev_upper != "AS" # not being used as an alias value
        )

        if is_bare_agg and i + 1 < len(tokens):
            nxt      = tokens[i + 1]
            nxt_core = nxt.lstrip("(").rstrip(",)")
            nxt_up   = nxt_core.upper()

            # Only wrap when the next token is a real column name, not a keyword
            if nxt_up not in _SQL_RESERVED and re.match(r'^[A-Za-z_]\w*$', nxt_core):
                trailing = tok[len(tok_core):]    # preserve any trailing comma
                out.append(f"{tok_core}({nxt_core}){trailing}")
                prev_upper = nxt_up
                i += 2
                continue

        out.append(tok)
        prev_upper = tok_up
        i += 1

    return " ".join(out)


def _validate(sql: str, table_name: str) -> str:
    """
    Validate and return the SQL string. Always repairs first.

    Checks (in order):
      1. Must be a SELECT statement
      2. Must contain a FROM clause
      3. Must reference the expected table
      4. Must not contain unsafe DML/DDL keywords

    Returns the repaired, validated SQL string.
    Raises SQLGenerationError on any failure.
    """
    # Always repair corruption before checking — self-heals any _clean() version
    sql = _repair(sql).strip()
    upper = sql.upper()

    # 1. Block unsafe DML/DDL keywords first (whole-word match only)
    #    This runs before the SELECT check so DROP/DELETE get a clear message.
    for kw in _UNSAFE_KEYWORDS:
        if re.search(rf'\b{kw}\b', upper):
            raise SQLGenerationError(
                f"Unsafe SQL detected — '{kw}' is not permitted. "
                "Only SELECT queries are allowed."
            )

    # 2. Only SELECT queries allowed
    if not upper.lstrip().startswith("SELECT"):
        raise SQLGenerationError(
            f"Only SELECT queries are allowed. Got: {sql[:80]!r}"
        )

    # 3. Must have a FROM clause
    if not re.search(r'\bFROM\b', upper):
        raise SQLGenerationError(
            f"Missing FROM clause: {sql[:80]!r}"
        )

    # 4. Must reference the correct table name
    if not re.search(rf'\bFROM\s+{re.escape(table_name)}\b', sql, re.IGNORECASE):
        raise SQLGenerationError(
            f"Query does not reference table '{table_name}': {sql[:120]!r}"
        )

    return sql


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_sql(
    question:   str,
    schema:     str   = None,
    table_name: str   = "cars",
    *,
    max_retries:  int   = 3,
    retry_delay:  float = 1.5,
) -> str:
    """
    Convert a natural-language question into a valid SQLite SELECT statement.

    Args:
        question:    Natural-language question.
        schema:      Schema string describing the table. Defaults to BMW schema.
        table_name:  Name of the SQLite table to query. Defaults to 'cars'.
        max_retries: Number of LLM call attempts before raising.
        retry_delay: Base delay in seconds between retries (doubles each attempt).

    Returns:
        A clean, valid SQLite SELECT query string (no trailing semicolon).

    Raises:
        SQLGenerationError: with a user-friendly message if all attempts fail.
        ValueError: if question is empty.
    """
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string.")

    _schema     = schema     or BMW_SCHEMA
    _table      = table_name or "cars"
    _sys_prompt = _build_system_prompt(_schema, _table)

    client     = Groq(api_key=_get_api_key())
    last_error: Exception = SQLGenerationError("No attempts made.")

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                temperature=0,      # deterministic — same question always same query
                max_tokens=512,
                messages=[
                    {"role": "system", "content": _sys_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Generate a SQLite SELECT query for the following question.\n"
                            f"Table: {_table}\n\n"
                            f"Question: {question}\n\n"
                            "Return ONLY the SQL query, nothing else."
                        ),
                    },
                ],
            )
            raw = response.choices[0].message.content
            sql = _clean(raw)
            sql = _validate(sql, _table)   # repair + validate; returns clean SQL
            logger.debug("Attempt %d — SQL accepted: %s", attempt, sql)
            return sql

        except SQLGenerationError as exc:
            last_error = exc
            logger.warning("Attempt %d — validation failed: %s", attempt, exc)
            if attempt < max_retries:
                time.sleep(retry_delay * (2 ** (attempt - 1)))

        except APIStatusError as exc:
            if exc.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
                time.sleep(retry_delay * (2 ** (attempt - 1)))
                last_error = exc
            else:
                raise

    # All attempts exhausted — raise with a clear, user-friendly message
    raise SQLGenerationError(
        "The AI generated an invalid SQL query. "
        "Please try rephrasing the question.\n"
        f"(Technical detail: {last_error})"
    ) from last_error