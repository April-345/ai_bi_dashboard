"""
patch_dashboard.py  -  run ONCE from your project root
=======================================================

    cd D:\\ai_bi_dashboard
    python patch_dashboard.py

Fixes:
  1. backend/sql_generator.py  - COUNT(*) AS count(FROM) bug
  2. frontend/app.py           - use_container_width deprecation warnings
"""

import ast, os, re, shutil, sys
from pathlib import Path

ROOT    = Path(__file__).parent.resolve()
SQL_GEN = ROOT / "backend"  / "sql_generator.py"
APP_PY  = ROOT / "frontend" / "app.py"

for p in (SQL_GEN, APP_PY):
    if not p.exists():
        sys.exit(f"\nERROR: {p} not found.\nRun from D:\\ai_bi_dashboard\n")

# Backup originals (once only)
for p in (SQL_GEN, APP_PY):
    bak = p.with_suffix(".py.bak")
    if not bak.exists():
        shutil.copy2(p, bak)
        print(f"  Backed up  {p.name}  ->  {bak.name}")

# =============================================================================
# PATCH 1 - backend/sql_generator.py
# =============================================================================
src = SQL_GEN.read_text(encoding="utf-8")

if "_repair" in src:
    print("  sql_generator.py already patched - skipping.")
else:
    OLD_CLEAN = (
        'def _clean(raw: str) -> str:\n'
        '    raw = re.sub(r"^```(?:sql)?\\s*", "", raw.strip(), flags=re.IGNORECASE)\n'
        '    raw = re.sub(r"\\s*```$", "", raw, flags=re.IGNORECASE)\n'
        '    raw = raw.strip().rstrip(";").strip()\n'
        '    # Fix bare aggregate syntax: MAX price -> MAX(price)\n'
        '    raw = re.sub(\n'
        "        r'\\b(MAX|MIN|AVG|SUM|COUNT)\\s+(?!\\()([a-zA-Z_][a-zA-Z0-9_]*)\\b',\n"
        '        lambda m: f"{m.group(1)}({m.group(2)})",\n'
        '        raw,\n'
        '        flags=re.IGNORECASE,\n'
        '    )\n'
        '    return raw'
    )

    NEW_CLEAN = (
        'def _repair(sql: str) -> str:\n'
        '    """Fix identifier(RESERVED_KEYWORD) corruption produced by old _clean.\n'
        '    e.g.  COUNT(*) AS count(FROM) data  ->  COUNT(*) AS count FROM data\n'
        '    """\n'
        '    _W = sorted([\n'
        '        "FROM","WHERE","GROUP","ORDER","HAVING","LIMIT","OFFSET","JOIN","INNER",\n'
        '        "LEFT","RIGHT","OUTER","CROSS","ON","AS","AND","OR","NOT","IN","IS",\n'
        '        "NULL","BETWEEN","SELECT","DISTINCT","BY","CASE","WHEN","THEN","ELSE",\n'
        '        "END","SET","INTO","VALUES","TABLE","INDEX","COUNT","MAX","MIN","AVG","SUM",\n'
        '    ], key=len, reverse=True)\n'
        "    pat = r'\\b([A-Za-z_][A-Za-z0-9_]*)\\s*\\(\\s*(' + '|'.join(_W) + r')\\b\\s*\\)'\n"
        "    return re.sub(pat, lambda m: m.group(1) + ' ' + m.group(2), sql, flags=re.IGNORECASE)\n"
        '\n'
        '\n'
        'def _clean(raw: str) -> str:\n'
        '    raw = re.sub(r"^```(?:sql)?\\s*", "", raw.strip(), flags=re.IGNORECASE)\n'
        '    raw = re.sub(r"\\s*```$",          "", raw,         flags=re.IGNORECASE)\n'
        '    raw = raw.strip().rstrip(";").strip()\n'
        '    raw = _repair(raw)  # undo any word(RESERVED) corruption\n'
        '    _SKIP = frozenset(["FROM","WHERE","GROUP","ORDER","HAVING","LIMIT","OFFSET",\n'
        '        "JOIN","ON","AS","AND","OR","NOT","IN","IS","NULL","SELECT","DISTINCT",\n'
        '        "BY","CASE","WHEN","THEN","ELSE","END","COUNT","MAX","MIN","AVG","SUM"])\n'
        '    tokens, out, prev = raw.split(), [], ""\n'
        '    i = 0\n'
        '    while i < len(tokens):\n'
        '        t = tokens[i]; tc = t.rstrip(",()"); tu = tc.upper()\n'
        '        if tu in {"MAX","MIN","AVG","SUM","COUNT"} and "(" not in t and prev != "AS":\n'
        '            if i + 1 < len(tokens):\n'
        '                nc = tokens[i+1].lstrip("(").rstrip(",")\n'
        '                if nc.upper() not in _SKIP and re.match(r"^[A-Za-z_]\\w*$", nc):\n'
        '                    out.append(tc + "(" + nc + ")" + t[len(tc):])\n'
        '                    prev = nc.upper(); i += 2; continue\n'
        '        out.append(t); prev = tu; i += 1\n'
        '    return " ".join(out)'
    )

    OLD_VALIDATE = (
        'def _validate(sql: str, table_name: str) -> None:\n'
        '    upper = sql.upper().lstrip()\n'
        '    if not upper.startswith("SELECT"):\n'
        '        raise SQLGenerationError(f"Not a SELECT statement: {sql!r}")\n'
        '    if "FROM" not in upper:\n'
        '        raise SQLGenerationError(f"Missing FROM clause: {sql!r}")\n'
        '    if f"FROM {table_name.upper()}" not in upper:\n'
        '        raise SQLGenerationError(f"Query does not reference table \'{table_name}\': {sql!r}")\n'
        '    forbidden = {"DROP", "DELETE", "INSERT", "UPDATE", "CREATE", "ALTER", "TRUNCATE"}\n'
        '    hits = forbidden & set(re.findall(r"\\b[A-Z]+\\b", upper))\n'
        '    if hits:\n'
        '        raise SQLGenerationError(f"Forbidden keyword(s): {hits}")'
    )

    NEW_VALIDATE = (
        'def _validate(sql: str, table_name: str) -> str:\n'
        '    """Repair, validate and return SQL. Raises SQLGenerationError on failure."""\n'
        '    sql = _repair(sql)\n'
        '    upper = sql.upper().lstrip()\n'
        '    for kw in ["DROP","DELETE","UPDATE","INSERT","ALTER","TRUNCATE","CREATE"]:\n'
        '        if re.search(rf"\\b{kw}\\b", upper):\n'
        '            raise SQLGenerationError(f"Unsafe SQL: \'{kw}\' not permitted.")\n'
        '    if not upper.startswith("SELECT"):\n'
        '        raise SQLGenerationError(f"Not a SELECT statement: {sql!r}")\n'
        '    if "FROM" not in upper:\n'
        '        raise SQLGenerationError(f"Missing FROM clause: {sql!r}")\n'
        '    if f"FROM {table_name.upper()}" not in upper:\n'
        '        raise SQLGenerationError(f"Query does not reference table \'{table_name}\': {sql!r}")\n'
        '    return sql'
    )

    OLD_CALL = "            _validate(sql, _table)\n            return sql"
    NEW_CALL = "            sql = _validate(sql, _table)\n            return sql"

    patched = src
    found = []
    if OLD_CLEAN    in patched: patched = patched.replace(OLD_CLEAN,    NEW_CLEAN);    found.append("_clean")
    if OLD_VALIDATE in patched: patched = patched.replace(OLD_VALIDATE, NEW_VALIDATE); found.append("_validate")
    if OLD_CALL     in patched: patched = patched.replace(OLD_CALL,     NEW_CALL);     found.append("call site")

    if not found:
        print("  WARNING: sql_generator.py has an unrecognised version - could not patch automatically.")
        print("  Please replace the file manually using the downloaded sql_generator.py from this chat.")
        sys.exit(1)

    try:
        ast.parse(patched)
    except SyntaxError as e:
        sys.exit(f"\nERROR: Patch produced invalid Python: {e}\n")

    SQL_GEN.write_text(patched, encoding="utf-8")

    # Self-test
    test_ns = {"re": re}
    class _SGE(Exception): pass
    test_ns["SQLGenerationError"] = _SGE
    start = patched.index("def _repair")
    # find end of _validate
    end_marker = "\n# ---------------------------------------------------------------------------\n# Public API"
    end = patched.index(end_marker) if end_marker in patched else len(patched)
    exec(compile(patched[start:end], "<test>", "exec"), test_ns)
    bad   = "SELECT fueltype, COUNT(*) AS count(FROM) data GROUP BY fueltype"
    good  = "SELECT fueltype, COUNT(*) AS count FROM data GROUP BY fueltype"
    r     = test_ns["_validate"](bad, "data")
    if r == good:
        print(f"  OK  sql_generator.py patched ({', '.join(found)}) + self-test passed")
    else:
        print(f"  WARN  self-test unexpected result: {r!r}")

# =============================================================================
# PATCH 2 - frontend/app.py
# =============================================================================
app_src = APP_PY.read_text(encoding="utf-8")
n = app_src.count("use_container_width")
if n == 0:
    print("  app.py already patched - skipping.")
else:
    out = app_src.replace("use_container_width=True",  "width='stretch'")
    out = out.replace(    "use_container_width=False", "width='content'")
    APP_PY.write_text(out, encoding="utf-8")
    print(f"  OK  app.py patched ({n} use_container_width replaced)")

print("\nDone. Restart Streamlit:")
print("  streamlit run frontend/app.py")