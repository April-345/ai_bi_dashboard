"""
apply_updates.py  —  run from D:\\ai_bi_dashboard
=================================================
    python apply_updates.py

Downloads the latest app.py and sql_generator.py from this chat
by simply copying the files you place next to this script.

INSTRUCTIONS:
  1. Download app.py and sql_generator.py from the chat
  2. Put them in D:\\ai_bi_dashboard\\ (same folder as this script)
  3. Run:  python apply_updates.py
"""
import shutil, sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()

COPIES = [
    (ROOT / "app.py",            ROOT / "frontend" / "app.py"),
    (ROOT / "sql_generator.py",  ROOT / "backend"  / "sql_generator.py"),
]

any_copied = False
for src, dst in COPIES:
    if src.exists():
        if not dst.parent.exists():
            sys.exit(f"ERROR: {dst.parent} not found. Run from D:\\ai_bi_dashboard")
        bak = dst.with_suffix(".py.bak")
        shutil.copy2(dst, bak)
        shutil.copy2(src, dst)
        print(f"  OK  {src.name}  ->  {dst}  (backup: {bak.name})")
        any_copied = True

if not any_copied:
    print("No files found to copy.")
    print("Put app.py and/or sql_generator.py next to this script, then re-run.")
    sys.exit(1)

print("\nDone. Restart Streamlit:")
print("  streamlit run frontend/app.py")