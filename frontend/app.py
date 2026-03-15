"""
frontend/app.py — AI BI Dashboard · BMW Inventory
Dual-theme (rose light / neon dark), hero query input, KPI cards, Plotly charts.
"""

import os, sys, json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
_FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR  = os.path.abspath(os.path.join(_FRONTEND_DIR, "..", "backend"))
_DB_PATH      = os.path.abspath(os.path.join(_FRONTEND_DIR, "..", "data", "bmw.db"))
sys.path.insert(0, _BACKEND_DIR)

from sql_generator  import generate_sql, BMW_SCHEMA
from query_executor import execute_query

# ── Dynamic query executor — uses uploaded DB if present ─────────────────────
def execute_query_on(sql: str) -> "pd.DataFrame":
    """Run sql against the uploaded DB if one exists, else the default bmw.db."""
    db = st.session_state.get("uploaded_db")
    if db:
        return execute_query(sql, db_path=db)
    return execute_query(sql)

def get_active_schema() -> str:
    return st.session_state.get("active_schema") or BMW_SCHEMA

def get_active_table() -> str:
    return st.session_state.get("active_table") or "cars"

# ── History persistence helpers ───────────────────────────────────────────────
_HISTORY_FILE = os.path.join(_FRONTEND_DIR, "..", "data", "query_history.json")
_DATA_DIR = os.path.join(_FRONTEND_DIR, "..", "data")
_DATASET_DB_PATH = os.path.join(_DATA_DIR, "uploaded_dataset.db")
_DATASET_META_PATH = os.path.join(_DATA_DIR, "uploaded_dataset_meta.json")

def _save_dataset_meta(label: str, schema: str, table: str) -> None:
    """Persist uploaded dataset metadata so it survives reload."""
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
        with open(_DATASET_META_PATH, "w", encoding="utf-8") as f:
            json.dump({"active_label": label, "active_schema": schema, "active_table": table}, f)
    except Exception:
        pass

def _load_dataset_meta() -> dict | None:
    """Load persisted dataset metadata if present."""
    try:
        if os.path.isfile(_DATASET_META_PATH) and os.path.isfile(_DATASET_DB_PATH):
            with open(_DATASET_META_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None

def _clear_persisted_dataset() -> None:
    """Remove persisted dataset files when user clears the dataset."""
    try:
        if os.path.isfile(_DATASET_META_PATH):
            os.remove(_DATASET_META_PATH)
        if os.path.isfile(_DATASET_DB_PATH):
            os.remove(_DATASET_DB_PATH)
    except Exception:
        pass

def _save_history(history: list) -> None:
    """Save history to JSON, stripping DataFrames (not JSON-serialisable)."""
    try:
        serialisable = [
            {
                "question": e.get("question", ""),
                "sql":      e.get("sql", ""),
                "insight":  e.get("insight") or "",
                "rows":     len(e["df"]) if e.get("df") is not None else 0,
                # Store a small CSV snapshot so we can reconstruct charts
                "df_csv":   e["df"].head(50).to_csv(index=False) if e.get("df") is not None else "",
            }
            for e in history
        ]
        with open(_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, ensure_ascii=False)
    except Exception:
        pass

def _load_history() -> list:
    """Load history from JSON and reconstruct lightweight DataFrames."""
    try:
        if not os.path.isfile(_HISTORY_FILE):
            return []
        with open(_HISTORY_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        import io
        result = []
        for e in raw:
            df = None
            if e.get("df_csv"):
                try:
                    df = pd.read_csv(io.StringIO(e["df_csv"]))
                except Exception:
                    df = None
            result.append({
                "question": e.get("question", ""),
                "sql":      e.get("sql", ""),
                "insight":  e.get("insight", ""),
                "df":       df,
            })
        return result
    except Exception:
        return []
from groq import Groq as _Groq

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI BI Dashboard — Ask Questions About Your Data",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session defaults ──────────────────────────────────────────────────────────
for k, v in {
    "dark_mode":       True,
    "result_df":       None,
    "result_sql":      None,
    "result_err":      None,
    "result_insight":  None,
    "query_history":   _load_history(),
    "show_history":    False,
    "uploaded_db":     None,   # path to uploaded SQLite db file
    "active_schema":   None,   # schema string for LLM
    "active_table":    "cars", # table name to query
    "active_label":    None,   # display name for uploaded dataset
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Restore persisted dataset after reload (so dataset stays across refresh)
_meta = _load_dataset_meta()
if _meta and not st.session_state.get("uploaded_db"):
    st.session_state.uploaded_db   = _DATASET_DB_PATH
    st.session_state.active_label  = _meta.get("active_label")
    st.session_state.active_schema = _meta.get("active_schema")
    st.session_state.active_table  = _meta.get("active_table", "data")

# Persist theme in URL so it survives refresh
if "theme" in st.query_params:
    st.session_state.dark_mode = st.query_params.get("theme", "dark") != "light"

dark = st.session_state.dark_mode

# ══════════════════════════════════════════════════════════════════════════════
# THEME PALETTES
# ══════════════════════════════════════════════════════════════════════════════
if dark:
    # ── DARK: charcoal + purple + neon green + electric blue ──────────────────
    BG_PAGE        = "#0d0d14"
    BG_SIDEBAR     = "#0a0a12"
    BG_CARD        = "#13131f"
    BG_CARD2       = "#16162a"
    BG_INPUT       = "#0a0a12"
    BORDER         = "#2a1f5c"
    BORDER_FOCUS   = "#7c3aed"
    TEXT_H         = "#f0eeff"
    TEXT_BODY      = "#b0a8d8"
    TEXT_MUTED     = "#4a3d7a"
    # accent gradients
    GRAD_ACCENT    = "90deg,#7c3aed,#00f5a0,#00d4ff"
    GRAD_BTN       = "135deg,#7c3aed 0%,#00c9a7 100%"
    GRAD_BTN_H     = "135deg,#6d28d9 0%,#00a37d 100%"
    GRAD_HERO      = "135deg,#0d0d14 0%,#1a0a3a 50%,#0d1a14 100%"
    GRAD_CARD      = "135deg,#13131f,#1a1230"
    GLOW_BTN       = "rgba(124,58,237,0.55)"
    GLOW_INPUT     = "rgba(124,58,237,0.25)"
    GLOW_CARD      = "rgba(124,58,237,0.12)"
    GLOW_GREEN     = "rgba(0,245,160,0.15)"
    SHADOW         = "0 8px 40px rgba(0,0,0,0.6)"
    # chart colors
    CHART_SEQ      = [[0,"#1a0a3a"],[0.5,"#7c3aed"],[1,"#00f5a0"]]
    CHART_DISC     = ["#7c3aed","#00f5a0","#00d4ff","#f472b6","#fb923c","#facc15"]
    PLOT_BG        = "#0d0d14"
    GRID_CLR       = "#1a1a2e"
    FONT_CLR       = "#4a3d7a"
    KPI_COLORS     = [
        ("135deg,#7c3aed,#a855f7","rgba(124,58,237,0.3)"),
        ("135deg,#00c9a7,#00f5a0","rgba(0,201,167,0.3)"),
        ("135deg,#0ea5e9,#00d4ff","rgba(14,165,233,0.3)"),
        ("135deg,#f472b6,#fb7185","rgba(244,114,182,0.3)"),
    ]
    SCROLLBAR_TH   = "#7c3aed66"
    SCROLLBAR_TH_H = "#7c3aed"
    ERR_BG         = "#1a0a0a"; ERR_BORDER = "#5c1a1a"; ERR_TEXT = "#f87171"
    WARN_BG        = "#1a1400"; WARN_BORDER= "#5c4200"; WARN_TEXT= "#fbbf24"
    EXPANDER_ICON  = "#7c3aed"
    CODE_COLOR     = "#00f5a0"
    TOGGLE_LABEL   = "☀️ Switch to Light Mode"

else:
    # ── LIGHT: white + rose + coral + pink ───────────────────────────────────
    BG_PAGE        = "#fff5f7"
    BG_SIDEBAR     = "#fce7f0"
    BG_CARD        = "#ffffff"
    BG_CARD2       = "#fdf2f5"
    BG_INPUT       = "#fff0f3"
    BORDER         = "#f9a8c9"
    BORDER_FOCUS   = "#e11d6a"
    TEXT_H         = "#1a0010"
    TEXT_BODY      = "#5a1a35"
    TEXT_MUTED     = "#c084a0"
    GRAD_ACCENT    = "90deg,#e11d6a,#fb7185,#f97316"
    GRAD_BTN       = "135deg,#e11d6a 0%,#f97316 100%"
    GRAD_BTN_H     = "135deg,#be185d 0%,#ea580c 100%"
    GRAD_HERO      = "135deg,#fff5f7 0%,#ffe4ef 50%,#fff5f7 100%"
    GRAD_CARD      = "135deg,#ffffff,#fff0f3"
    GLOW_BTN       = "rgba(225,29,106,0.45)"
    GLOW_INPUT     = "rgba(225,29,106,0.18)"
    GLOW_CARD      = "rgba(225,29,106,0.08)"
    GLOW_GREEN     = "rgba(249,115,22,0.10)"
    SHADOW         = "0 4px 32px rgba(225,29,106,0.10)"
    CHART_SEQ      = [[0,"#ffe4ef"],[0.5,"#e11d6a"],[1,"#f97316"]]
    CHART_DISC     = ["#e11d6a","#f97316","#f59e0b","#8b5cf6","#10b981","#0ea5e9"]
    PLOT_BG        = "#fff5f7"
    GRID_CLR       = "#fce7f0"
    FONT_CLR       = "#c084a0"
    KPI_COLORS     = [
        ("135deg,#e11d6a,#fb7185","rgba(225,29,106,0.2)"),
        ("135deg,#f97316,#fb923c","rgba(249,115,22,0.2)"),
        ("135deg,#8b5cf6,#a78bfa","rgba(139,92,246,0.2)"),
        ("135deg,#0ea5e9,#38bdf8","rgba(14,165,233,0.2)"),
    ]
    SCROLLBAR_TH   = "#f9a8c9"
    SCROLLBAR_TH_H = "#e11d6a"
    ERR_BG         = "#fff5f5"; ERR_BORDER = "#fecaca"; ERR_TEXT = "#dc2626"
    WARN_BG        = "#fffbeb"; WARN_BORDER= "#fde68a"; WARN_TEXT= "#d97706"
    EXPANDER_ICON  = "#e11d6a"
    CODE_COLOR     = "#e11d6a"
    TOGGLE_LABEL   = "🌙 Switch to Dark Mode"

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
/* Force sidebar always expanded (override Streamlit collapsed state) */
section[data-testid="stSidebar"] {{
    display: block !important;
    visibility: visible !important;
    min-width: 300px !important;
    width: 300px !important;
}}
section[data-testid="stSidebar"] > div:first-child {{
    width: 300px !important;
    min-width: 300px !important;
    margin-left: 0 !important;
    transform: none !important;
}}
/* When Streamlit marks sidebar collapsed, force it visible anyway */
section[data-testid="stSidebar"][aria-expanded="false"],
section[data-testid="stSidebar"][aria-expanded="false"] > div:first-child {{
    width: 300px !important;
    min-width: 300px !important;
    margin-left: 0 !important;
    transform: none !important;
    visibility: visible !important;
}}

@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

*, html, body, [class*="css"] {{
    font-family: 'Space Grotesk', sans-serif !important;
    box-sizing: border-box;
}}

/* ── Base ── */
.stApp {{ background: {BG_PAGE} !important; color: {TEXT_BODY} !important; }}
.block-container {{ padding: 0 2.5rem 3rem 2.5rem !important; max-width: 1500px !important; }}
footer {{ visibility: hidden !important; }}
/* Top header/toolbar — light in light mode, visible text */
header,
header[data-testid="stHeader"],
.stApp header,
[data-testid="stAppViewContainer"] > header,
[data-testid="stHeader"] {{
    background: {BG_SIDEBAR} !important;
    border-bottom: 1px solid {BORDER} !important;
}}
header *,
header[data-testid="stHeader"] *,
.stApp header *,
[data-testid="stHeader"] * {{
    color: {TEXT_BODY} !important;
}}
/* Deploy / toolbar buttons in header stay visible */
header a,
header button {{
    color: {TEXT_H} !important;
}}
/* Keep toolbar (Running/Stop, Deploy) permanently visible */
header,
header *,
#MainMenu,
#MainMenu *,
[data-testid="stHeader"],
[data-testid="stHeader"] * {{
    visibility: visible !important;
}}
/* Don't hide 'header' — it can contain the sidebar expand/collapse control */

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width:5px; height:5px; }}
::-webkit-scrollbar-track {{ background:{BG_PAGE}; }}
::-webkit-scrollbar-thumb {{ background:{SCROLLBAR_TH}; border-radius:10px; }}
::-webkit-scrollbar-thumb:hover {{ background:{SCROLLBAR_TH_H}; }}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: {BG_SIDEBAR} !important;
    border-right: 1px solid {BORDER} !important;
}}
section[data-testid="stSidebar"] * {{ color: {TEXT_BODY} !important; }}
/* Hide sidebar collapse button (arrow) at top — sidebar is forced open */
section[data-testid="stSidebar"] button[aria-label*="ollapse"],
section[data-testid="stSidebar"] button[aria-label*="idebar"],
section[data-testid="stSidebar"] > div > button:first-of-type {{
    display: none !important;
}}
/* Hide sidebar resize handle (no double-arrow cursor/tooltip on hover) */
[data-testid="stSidebarResizeHandle"],
section[data-testid="stSidebar"] [role="separator"],
section[data-testid="stSidebar"] > div > [role="separator"],
section[data-testid="stSidebar"] div[style*="cursor"][style*="resize"] {{
    display: none !important;
    pointer-events: none !important;
}}
/* Hide icon-name text that can appear next to the sidebar (e.g. double_arrow_right) */
[data-testid="stSidebarResizeHandle"],
[data-testid="stSidebarResizeHandle"] * {{
    font-size: 0 !important;
    line-height: 0 !important;
    visibility: hidden !important;
}}
/* Hide sidebar collapse/expand button in main area (and icon name text) */
button[aria-label*="ollapse"], button[aria-label*="idebar"], button[aria-label*="xpand"],
[data-testid="stSidebarCollapseButton"], [data-testid="stSidebarExpandButton"] {{
    display: none !important;
}}

/* ── Animated top bar ── */
@keyframes shimmer {{
    0%   {{ background-position: -200% center; }}
    100% {{ background-position:  200% center; }}
}}
.top-bar {{
    height: 3px;
    background: linear-gradient({GRAD_ACCENT});
    background-size: 200% auto;
    animation: shimmer 3s linear infinite;
    margin-bottom: 0;
}}

/* ── Hero section ── */
@keyframes heroIn {{
    from {{ opacity:0; transform:translateY(-20px); }}
    to   {{ opacity:1; transform:translateY(0); }}
}}
.hero {{
    background: linear-gradient({GRAD_HERO});
    border-bottom: 1px solid {BORDER};
    padding: 3rem 3rem 2.5rem 3rem;
    margin: 0 -2.5rem 2.5rem -2.5rem;
    position: relative;
    overflow: hidden;
    animation: heroIn 0.7s cubic-bezier(.4,0,.2,1) both;
}}
.hero::before {{
    content:'';
    position:absolute; top:-100px; right:-100px;
    width:350px; height:350px;
    background: radial-gradient(circle, {GLOW_BTN} 0%, transparent 65%);
    border-radius:50%; pointer-events:none;
}}
.hero::after {{
    content:'';
    position:absolute; bottom:-80px; left:20%;
    width:250px; height:250px;
    background: radial-gradient(circle, {GLOW_GREEN} 0%, transparent 65%);
    border-radius:50%; pointer-events:none;
}}
.hero-eyebrow {{
    font-size:0.65rem; font-weight:700; letter-spacing:0.22em;
    text-transform:uppercase; color:{TEXT_MUTED};
    margin-bottom:0.7rem;
}}
.hero h1 {{
    font-size:2.5rem; font-weight:700; letter-spacing:-0.04em;
    line-height:1.1; color:{TEXT_H}; margin:0 0 0.6rem 0;
}}
.hero h1 em {{
    font-style:normal;
    background: linear-gradient({GRAD_ACCENT});
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text; background-size:200% auto;
    animation: shimmer 4s linear infinite;
}}
.hero-sub {{
    font-size:0.95rem; color:{TEXT_MUTED}; font-weight:400;
    max-width:600px; line-height:1.6;
}}

/* ── Hero input wrapper ── */
.hero-input-wrap {{
    position: relative;
    margin-top: 2rem;
    max-width: 860px;
}}
.hero-input-wrap::before {{
    content:'';
    position:absolute; inset:-2px;
    background: linear-gradient({GRAD_ACCENT});
    background-size: 200% auto;
    animation: shimmer 3s linear infinite;
    border-radius:18px;
    z-index:0;
    opacity:0.85;
}}
.hero-input-inner {{
    position:relative; z-index:1;
    background:{BG_INPUT};
    border-radius:16px;
    padding:2px;
}}
.stTextArea textarea {{
    background: {BG_INPUT} !important;
    border: none !important;
    border-radius: 14px !important;
    color: {TEXT_H} !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 400 !important;
    line-height: 1.6 !important;
    padding: 1.1rem 1.4rem !important;
    transition: box-shadow 0.25s ease !important;
    resize: none !important;
    caret-color: {BORDER_FOCUS} !important;
}}
.stTextArea textarea:focus {{
    box-shadow: 0 0 0 0px transparent !important;
    outline: none !important;
}}
.stTextArea textarea::placeholder {{ color:{TEXT_MUTED} !important; opacity:1 !important; }}
.stTextArea label {{ display:none !important; }}
[data-testid="stTextAreaResizeHandle"] {{ display:none !important; }}

/* ── Buttons ── */
.stButton > button {{
    background: linear-gradient({GRAD_BTN}) !important;
    color:#ffffff !important;
    border:none !important;
    border-radius:12px !important;
    font-family:'Space Grotesk',sans-serif !important;
    font-size:0.9rem !important;
    font-weight:700 !important;
    letter-spacing:0.04em !important;
    padding:0.65rem 2rem !important;
    transition: all 0.22s cubic-bezier(.4,0,.2,1) !important;
    box-shadow: 0 4px 20px {GLOW_BTN} !important;
}}
.stButton > button:hover {{
    background: linear-gradient({GRAD_BTN_H}) !important;
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 8px 32px {GLOW_BTN} !important;
}}
.stButton > button:active {{ transform:translateY(0) scale(0.99) !important; }}





/* ── Dataset Intelligence section ── */
.di-section {{
    background: linear-gradient({GRAD_CARD});
    border: 1px solid {BORDER};
    border-radius: 20px;
    padding: 2rem 2.2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: {SHADOW};
    animation: fadeUp 0.5s ease both;
}}
.di-section::before {{
    content:'';
    position:absolute; bottom:-60px; right:-60px;
    width:200px; height:200px;
    background: radial-gradient(circle, {GLOW_GREEN} 0%, transparent 70%);
    border-radius:50%; pointer-events:none;
}}
.di-header {{
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 1.4rem;
}}
.di-header-icon {{
    width: 36px; height: 36px; border-radius: 10px;
    background: linear-gradient({GRAD_BTN});
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    box-shadow: 0 2px 10px {GLOW_BTN};
    flex-shrink: 0;
}}
.di-header-text {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: {TEXT_H};
    letter-spacing: -0.01em;
}}
.di-header-sub {{
    font-size: 0.72rem;
    color: {TEXT_MUTED};
    font-weight: 400;
    margin-top: 0.1rem;
}}
.di-summary {{
    background: {BG_INPUT};
    border: 1px solid {BORDER};
    border-left: 3px solid {BORDER_FOCUS};
    border-radius: 10px;
    padding: 1rem 1.3rem;
    font-size: 0.92rem;
    color: {TEXT_BODY};
    line-height: 1.7;
    margin-top: 1.4rem;
    font-style: italic;
}}
.di-summary-label {{
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: {BORDER_FOCUS};
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}}

/* ── KPI cards ── */
@keyframes cardIn {{
    from {{ opacity:0; transform:translateY(16px) scale(0.97); }}
    to   {{ opacity:1; transform:translateY(0) scale(1); }}
}}
.kpi-card {{
    background: linear-gradient({GRAD_CARD});
    border:1px solid {BORDER};
    border-radius:18px;
    padding:1.6rem 1.8rem;
    position:relative; overflow:hidden;
    box-shadow: {SHADOW};
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    animation: cardIn 0.5s cubic-bezier(.4,0,.2,1) both;
    cursor:default;
}}
.kpi-card:hover {{
    transform:translateY(-4px);
    box-shadow: {SHADOW}, 0 0 30px {GLOW_CARD};
    border-color:{BORDER_FOCUS};
}}
.kpi-orb {{
    position:absolute; top:-35px; right:-35px;
    width:110px; height:110px;
    border-radius:50%; pointer-events:none; opacity:0.5;
}}
.kpi-icon {{ font-size:1.6rem; margin-bottom:0.8rem; display:block; }}
.kpi-label {{
    font-size:0.65rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.16em; color:{TEXT_MUTED}; margin-bottom:0.4rem;
}}
.kpi-value {{
    font-size:2.2rem; font-weight:700; letter-spacing:-0.04em;
    color:{TEXT_H}; line-height:1;
}}
.kpi-sub {{
    font-size:0.72rem; font-weight:500;
    margin-top:0.45rem; color:{TEXT_MUTED};
}}
.kpi-bar {{
    position:absolute; bottom:0; left:0; right:0;
    height:3px; border-radius:0 0 18px 18px;
}}

/* ── Section titles ── */
@keyframes fadeUp {{
    from {{ opacity:0; transform:translateY(10px); }}
    to   {{ opacity:1; transform:translateY(0); }}
}}
.section-hd {{
    display:flex; align-items:center; gap:0.7rem;
    margin:2.2rem 0 1rem 0;
    animation: fadeUp 0.4s ease both;
}}
.section-hd-line {{
    flex:1; height:1px;
    background: linear-gradient(90deg, {BORDER}, transparent);
}}
.section-hd-text {{
    font-size:0.68rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.18em; color:{TEXT_MUTED}; white-space:nowrap;
}}

/* ── Query history ── */
.hist-wrap {{
    animation: fadeUp 0.4s ease both;
}}
.hist-card {{
    background: linear-gradient({GRAD_CARD});
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: {SHADOW};
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s ease;
}}
.hist-card:hover {{ border-color: {BORDER_FOCUS}; }}
.hist-card-header {{
    display: flex; align-items: flex-start;
    justify-content: space-between; gap: 1rem;
    margin-bottom: 0.8rem;
}}
.hist-question {{
    font-size: 0.95rem; font-weight: 600;
    color: {TEXT_H}; line-height: 1.4; flex: 1;
}}
.hist-badge {{
    font-size: 0.62rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.14em;
    background: {BORDER_FOCUS}22; color: {BORDER_FOCUS};
    border: 1px solid {BORDER_FOCUS}44;
    border-radius: 20px; padding: 0.18rem 0.65rem;
    white-space: nowrap; flex-shrink: 0;
}}
.hist-insight {{
    font-size: 0.84rem; color: {TEXT_BODY};
    line-height: 1.65; font-style: italic;
    border-left: 3px solid {BORDER_FOCUS}66;
    padding-left: 0.85rem; margin-top: 0.3rem;
}}
.hist-num {{
    font-size: 0.68rem; color: {TEXT_MUTED};
    margin-top: 0.7rem; font-weight: 500;
}}

/* ── AI Insight card ── */
.insight-card {{
    background: linear-gradient(135deg, {BG_CARD2}, {BG_CARD});
    border: 1px solid {BORDER};
    border-left: 4px solid {BORDER_FOCUS};
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 0.5rem;
    box-shadow: {SHADOW}, 0 0 30px {GLOW_CARD};
    position: relative;
    overflow: hidden;
    animation: fadeUp 0.5s cubic-bezier(.4,0,.2,1) both;
}}
.insight-card::before {{
    content: '';
    position: absolute; top: -40px; right: -40px;
    width: 130px; height: 130px;
    background: radial-gradient(circle, {GLOW_BTN} 0%, transparent 70%);
    border-radius: 50%; pointer-events: none;
}}
.insight-avatar {{
    display: flex; align-items: center; gap: 0.6rem;
    margin-bottom: 0.9rem;
}}
.insight-avatar-icon {{
    width: 32px; height: 32px; border-radius: 50%;
    background: linear-gradient({GRAD_BTN});
    display: flex; align-items: center; justify-content: center;
    font-size: 0.9rem; flex-shrink: 0;
    box-shadow: 0 2px 10px {GLOW_BTN};
}}
.insight-avatar-label {{
    font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.16em; color: {BORDER_FOCUS};
}}
.insight-text {{
    font-size: 0.97rem; line-height: 1.75;
    color: {TEXT_BODY}; font-weight: 400;
}}

/* ── Results fade-in ── */
.results-wrap {{
    animation: fadeUp 0.5s cubic-bezier(.4,0,.2,1) both;
}}

/* ── Expander ── */
/* Hide the raw "_arrow_" text Streamlit injects into expander labels */
.stExpander > details > summary > span:first-child {{
    display: none !important;
}}
.stExpander > details > summary p {{
    color: {TEXT_BODY} !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
}}
.stExpander {{
    background:{BG_CARD} !important;
    border:1px solid {BORDER} !important;
    border-radius:14px !important;
    box-shadow:{SHADOW} !important;
    overflow:hidden !important;
}}
.stExpander > details > summary {{
    color:{TEXT_MUTED} !important;
    font-size:0.82rem !important; font-weight:600 !important;
    padding:0.85rem 1.1rem !important;
}}
.stExpander > details > summary:hover {{ color:{EXPANDER_ICON} !important; }}
.stExpander > details > summary svg {{ fill:{EXPANDER_ICON} !important; }}
pre, code {{
    font-family:'Fira Code',monospace !important;
    font-size:0.78rem !important;
    background:{BG_INPUT} !important;
    color:{CODE_COLOR} !important;
    border-radius:10px !important;
    border:none !important;
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    border:1px solid {BORDER} !important;
    border-radius:14px !important;
    overflow:hidden !important;
    box-shadow:{SHADOW} !important;
    animation: fadeUp 0.45s ease both;
}}

/* ── Error / warn boxes ── */
.err-box {{
    background:{ERR_BG}; border:1px solid {ERR_BORDER};
    border-left:4px solid #ef4444; border-radius:12px;
    padding:1rem 1.3rem; color:{ERR_TEXT};
    font-size:0.87rem; font-weight:500;
    animation: fadeUp 0.3s ease;
}}
.warn-box {{
    background:{WARN_BG}; border:1px solid {WARN_BORDER};
    border-left:4px solid #f59e0b; border-radius:12px;
    padding:1rem 1.3rem; color:{WARN_TEXT};
    font-size:0.87rem; font-weight:500;
    animation: fadeUp 0.3s ease;
}}

/* ── Sidebar ── */
.sb-brand {{
    text-align:center; padding:1.5rem 0 1.8rem 0;
    border-bottom:1px solid {BORDER}; margin-bottom:1.2rem;
}}
.sb-brand-icon {{ font-size:2.2rem; margin-bottom:0.4rem; }}
.sb-brand-name {{
    font-size:1rem; font-weight:700; color:{TEXT_H};
    letter-spacing:-0.02em;
}}
.sb-brand-sub {{
    font-size:0.65rem; text-transform:uppercase;
    letter-spacing:0.14em; color:{TEXT_MUTED}; margin-top:0.2rem;
}}
.sb-sec-title {{
    font-size:0.62rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.18em; color:{TEXT_MUTED};
    margin:1.3rem 0 0.6rem 0;
}}
.sb-chip {{
    background:{BG_CARD}; border:1px solid {BORDER};
    border-radius:8px; padding:0.48rem 0.8rem;
    font-size:0.76rem; color:{TEXT_BODY};
    margin-bottom:0.35rem; display:block;
    transition:border-color 0.15s, color 0.15s;
    cursor:pointer;
}}
.sb-chip:hover {{ border-color:{BORDER_FOCUS}; color:{TEXT_H}; }}
.sb-stat {{
    display:flex; justify-content:space-between;
    padding:0.38rem 0; border-bottom:1px solid {BORDER};
    font-size:0.76rem;
}}
.sb-stat:last-child {{ border-bottom:none; }}
.sb-stat .k {{ color:{TEXT_MUTED}; }}
.sb-stat .v {{ color:{TEXT_H}; font-weight:600; }}

/* ── File uploader (light boxes + visible text in both themes) ── */
[data-testid="stFileUploader"] {{
    background: transparent !important;
}}
[data-testid="stFileUploader"] > div,
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploader"] section {{
    background: {BG_CARD} !important;
    border: 2px dashed {BORDER} !important;
    border-radius: 12px !important;
    color: {TEXT_BODY} !important;
}}
[data-testid="stFileUploader"] > div *,
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] *,
[data-testid="stFileUploader"] section * {{
    color: {TEXT_BODY} !important;
}}
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] small {{
    color: {TEXT_MUTED} !important;
}}
[data-testid="stFileUploader"] button {{
    background: linear-gradient({GRAD_BTN}) !important;
    color: #ffffff !important;
    border: none !important;
}}

/* ── Toggle ── */
label[data-testid="stToggle"] span {{ color:{TEXT_BODY} !important; font-size:0.85rem !important; }}
</style>
""", unsafe_allow_html=True)

# Force sidebar to be expanded and hide resize handle / tooltip
st.markdown("""
<script>
(function() {
    function expandSidebar() {
        var sidebar = document.querySelector('section[data-testid="stSidebar"]');
        if (sidebar && sidebar.getAttribute('aria-expanded') === 'false') {
            sidebar.setAttribute('aria-expanded', 'true');
        }
    }
    function inTopToolbar(el) {
        if (!el || !el.getBoundingClientRect) return false;
        var top = el.getBoundingClientRect().top;
        return top >= 0 && top < 80;
    }
    function hideResizeHandleAndTooltip() {
        var sidebar = document.querySelector('section[data-testid="stSidebar"]');
        var keywords = /arrow|resize|keyboard|double|collapse|expand/i;
        function neutralize(el) {
            if (inTopToolbar(el)) return;
            var t = (el.getAttribute('title') || '') + (el.getAttribute('aria-label') || '');
            if (keywords.test(t) || el.getAttribute('data-testid') === 'stSidebarResizeHandle') {
                el.removeAttribute('title');
                el.removeAttribute('aria-label');
                el.style.setProperty('display', 'none', 'important');
                el.style.setProperty('pointer-events', 'none', 'important');
            }
        }
        if (sidebar) {
            sidebar.querySelectorAll('[title], [aria-label], [data-testid="stSidebarResizeHandle"]').forEach(neutralize);
            var next = sidebar.nextElementSibling;
            if (next && (next.getAttribute('title') || next.getAttribute('aria-label') || next.getAttribute('data-testid'))) {
                neutralize(next);
            }
        }
        document.querySelectorAll('[data-testid="stSidebarResizeHandle"]').forEach(neutralize);
        var sidebarOrParent = sidebar ? sidebar.parentElement : null;
        if (sidebarOrParent) {
            [].slice.call(sidebarOrParent.querySelectorAll('[title]')).forEach(function(el) {
                if (el.closest('header') || inTopToolbar(el)) return;
                var t = (el.getAttribute('title') || '').toLowerCase();
                if (/arrow|resize|keyboard|double/.test(t)) {
                    el.removeAttribute('title');
                    el.style.setProperty('display', 'none', 'important');
                    el.style.setProperty('pointer-events', 'none', 'important');
                }
            });
        }
    }
    function hideIconNameText() {
        var iconName = /keyboard_double_double_arrow_right|double_arrow|arrow_right|ouble_arrow/i;
        document.querySelectorAll('span, div, p, button, label').forEach(function(el) {
            if (el.closest('header') || inTopToolbar(el)) return;
            var t = (el.textContent || '').trim();
            if (iconName.test(t) && t.length < 80) {
                var btn = el.closest('button');
                if (btn) { btn.style.setProperty('display', 'none', 'important'); }
                else {
                    el.style.setProperty('display', 'none', 'important');
                    var parent = el.parentElement;
                    if (parent && !parent.closest('button') && (parent.textContent || '').trim().length < 80 && iconName.test((parent.textContent || '').trim())) {
                        parent.style.setProperty('display', 'none', 'important');
                    }
                }
            }
        });
    }
    function run() {
        expandSidebar();
        hideResizeHandleAndTooltip();
        hideIconNameText();
    }
    function keepToolbarVisible() {
        var toolbars = document.querySelectorAll('header, [data-testid="stHeader"], #MainMenu');
        toolbars.forEach(function(container) {
            container.style.setProperty('visibility', 'visible', 'important');
            var children = container.querySelectorAll('*');
            for (var i = 0; i < children.length; i++) {
                var el = children[i];
                el.style.setProperty('visibility', 'visible', 'important');
                try {
                    if (el.getBoundingClientRect().top < 120 && getComputedStyle(el).display === 'none') {
                        el.style.setProperty('display', 'flex', 'important');
                    }
                } catch (e) {}
            }
        });
        document.querySelectorAll('[data-testid="stAppViewContainer"] > div').forEach(function(div) {
            try {
                if (div.getBoundingClientRect().top < 80) {
                    div.style.setProperty('visibility', 'visible', 'important');
                    if (getComputedStyle(div).display === 'none') {
                        div.style.setProperty('display', 'flex', 'important');
                    }
                }
            } catch (e) {}
        });
    }
    function watchToolbar() {
        var toolbars = document.querySelectorAll('header, [data-testid="stHeader"], #MainMenu');
        toolbars.forEach(function(container) {
            if (container._toolbarWatcher) return;
            container._toolbarWatcher = true;
            var obs = new MutationObserver(function() {
                keepToolbarVisible();
            });
            obs.observe(container, { childList: true, subtree: true, attributes: true, attributeFilter: ['style', 'class'] });
        });
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            run();
            keepToolbarVisible();
            watchToolbar();
        });
    } else {
        run();
        keepToolbarVisible();
        watchToolbar();
    }
})();
</script>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DYNAMIC DATASET STATS — works for any uploaded CSV or BMW fallback
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_dynamic_meta(db_path: str, table: str):
    """Compute generic statistics for any table in any SQLite database."""
    try:
        total   = int(execute_query(f"SELECT COUNT(*) FROM {table}", db_path=db_path).iloc[0,0])
        n_cols  = int(execute_query(f"PRAGMA table_info({table})", db_path=db_path).shape[0])

        # Get column info
        col_info = execute_query(f"PRAGMA table_info({table})", db_path=db_path)
        num_cols = [r["name"] for _, r in col_info.iterrows()
                    if r["type"].upper() in ("REAL","INTEGER","NUMERIC","FLOAT","INT","DOUBLE")]
        txt_cols = [r["name"] for _, r in col_info.iterrows()
                    if r["type"].upper() in ("TEXT","VARCHAR","CHAR","STRING","BLOB","")]

        # Up to 3 numeric averages
        num_stats = {}
        for c in num_cols[:3]:
            try:
                v = execute_query(f"SELECT AVG({c}) FROM {table}", db_path=db_path).iloc[0,0]
                if v is not None:
                    num_stats[c] = float(v)
            except Exception:
                pass

        # Up to 2 top categorical values
        cat_stats = {}
        for c in txt_cols[:2]:
            try:
                v = execute_query(
                    f"SELECT {c} FROM {table} GROUP BY {c} ORDER BY COUNT(*) DESC LIMIT 1",
                    db_path=db_path
                ).iloc[0,0]
                if v is not None:
                    cat_stats[c] = str(v)
            except Exception:
                pass

        return {"total": total, "n_cols": n_cols,
                "num_stats": num_stats, "cat_stats": cat_stats,
                "num_cols": num_cols, "txt_cols": txt_cols}
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_dynamic_summary(stats_key: str, stats: dict, table: str):
    """Generate an AI summary for any dataset given its computed stats."""
    if not stats:
        return None
    try:
        _lines = [f"- Total records: {stats['total']:,}",
                  f"- Total columns: {stats['n_cols']}"]
        for c, v in stats.get("num_stats", {}).items():
            _lines.append(f"- Average {c}: {v:,.2f}")
        for c, v in stats.get("cat_stats", {}).items():
            _lines.append(f"- Most common {c}: {v}")
        _stat_block = "\n".join(_lines)
        _client = _Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
        _prompt = (
            f"Here are summary statistics for a dataset stored in table '{table}':\n"
            f"{_stat_block}\n\n"
            "Write a concise 3-sentence business summary for a non-technical user. "
            "Highlight key patterns, dominant categories, and notable values. "
            "Plain text only — no bullet points, no markdown, no headers."
        )
        _resp = _client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=180,
            messages=[
                {"role": "system", "content": "You are a friendly business intelligence analyst."},
                {"role": "user",   "content": _prompt},
            ],
        )
        return _resp.choices[0].message.content.strip()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_sidebar_breakdown(db_path: str, table: str):
    """Load column breakdown stats for sidebar display."""
    try:
        col_info = execute_query(f"PRAGMA table_info({table})", db_path=db_path)
        txt_cols = [r["name"] for _, r in col_info.iterrows()
                    if r["type"].upper() in ("TEXT","VARCHAR","CHAR","STRING","BLOB","")]
        breakdowns = {}
        for c in txt_cols[:2]:
            try:
                df = execute_query(
                    f"SELECT {c} val, COUNT(*) n FROM {table} GROUP BY {c} ORDER BY n DESC LIMIT 6",
                    db_path=db_path
                )
                breakdowns[c] = df
            except Exception:
                pass
        return breakdowns
    except Exception:
        return {}

# ══════════════════════════════════════════════════════════════════════════════
# RESOLVE ACTIVE DB PATH & COMPUTE STATS (session-aware, runs every render)
# ══════════════════════════════════════════════════════════════════════════════

_active_db    = st.session_state.get("uploaded_db") or _DB_PATH
_active_table = get_active_table()
_active_label = st.session_state.get("active_label") or "BMW Vehicle Inventory (demo)"
_is_uploaded  = bool(st.session_state.get("uploaded_db"))

# Load stats for whichever DB is active
_meta        = load_dynamic_meta(_active_db, _active_table)
_breakdowns  = load_sidebar_breakdown(_active_db, _active_table)

# Cache key for AI summary (changes when dataset changes)
_summary_key = f"{_active_db}_{_active_table}"
_ds_summary  = load_dynamic_summary(_summary_key, _meta, _active_table) if _meta else None

# ── Suggested questions — BMW-specific when demo, generic when uploaded ───────
if _is_uploaded and _meta:
    _num_c = (_meta.get("num_cols") or [])[:2]
    _txt_c = (_meta.get("txt_cols") or [])[:2]
    _suggestions = []
    if _num_c:
        _suggestions.append(f"What is the average {_num_c[0]}?")
        _suggestions.append(f"Show top 10 records by {_num_c[0]} descending")
    if len(_num_c) > 1:
        _suggestions.append(f"What is the total {_num_c[1]}?")
    if _txt_c:
        _suggestions.append(f"Count records grouped by {_txt_c[0]}")
    if len(_txt_c) > 1:
        _suggestions.append(f"What is the most common {_txt_c[1]}?")
    _suggestions.append("How many rows are in this dataset?")
else:
    _suggestions = [
        "What are the 10 cheapest diesel cars?",
        "Average price by fuel type",
        "Top 5 cars with best MPG",
        "How many automatic cars were made after 2018?",
        "Average mileage by transmission",
        "Most expensive cars under 50k miles",
        "Price vs mileage for petrol cars",
        "Cheapest automatic petrol cars",
    ]

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div class="sb-brand">
        <div class="sb-brand-icon">🚗</div>
        <div class="sb-brand-name">AI Data Analytics</div>
        <div class="sb-brand-sub">Natural Language · Any Dataset</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Dataset Upload ────────────────────────────────────────────────────────
    st.markdown('<div class="sb-sec-title">📂 Upload Dataset</div>', unsafe_allow_html=True)
    _uploaded_file = st.file_uploader(
        "Upload CSV", type=["csv"],
        label_visibility="collapsed",
        help="Upload any CSV file to query it with AI"
    )
    if _uploaded_file is not None:
        try:
            import sqlite3 as _sqlite3
            _df_up = pd.read_csv(_uploaded_file)
            # Normalise column names
            _df_up.columns = (
                _df_up.columns.str.strip()
                .str.lower()
                .str.replace(r"[^\w]", "_", regex=True)
                .str.replace(r"_+", "_", regex=True)
                .str.strip("_")
            )
            # Save to persistent path so dataset survives reload
            os.makedirs(_DATA_DIR, exist_ok=True)
            with _sqlite3.connect(_DATASET_DB_PATH) as _conn:
                _df_up.to_sql("data", _conn, if_exists="replace", index=False)
            # Build schema string
            _type_map = {"object": "TEXT", "int64": "INTEGER", "float64": "REAL",
                         "bool": "INTEGER", "datetime64[ns]": "TEXT"}
            _schema_lines = ["Table: data", "Columns:"]
            for _col, _dtype in _df_up.dtypes.items():
                _sql_type = _type_map.get(str(_dtype), "TEXT")
                _sample = str(_df_up[_col].dropna().iloc[0]) if not _df_up[_col].dropna().empty else ""
                _schema_lines.append(f"  - {_col:<20} {_sql_type:<8} (e.g. {_sample!r})")
            _schema_str = "\n".join(_schema_lines)
            # Store in session state and persist metadata for reload
            st.session_state.uploaded_db    = _DATASET_DB_PATH
            st.session_state.active_schema   = _schema_str
            st.session_state.active_table    = "data"
            st.session_state.active_label    = _uploaded_file.name
            _save_dataset_meta(_uploaded_file.name, _schema_str, "data")
            st.session_state.result_df      = None
            st.session_state.result_sql     = None
            st.session_state.result_err     = None
            st.session_state.result_insight = None
            load_dynamic_meta.clear()
            load_dynamic_summary.clear()
            load_sidebar_breakdown.clear()
            st.success(f"Dataset loaded!  {len(_df_up):,} rows · {len(_df_up.columns)} columns")
        except Exception as _e:
            st.error(f"Upload failed: {_e}")

    if st.session_state.get("active_label"):
        st.markdown(
            f'<div class="sb-stat" style="margin-top:0.3rem">' 
            f'<span class="k">Active dataset</span>' 
            f'<span class="v" style="color:{BORDER_FOCUS}">{st.session_state.active_label}</span></div>',
            unsafe_allow_html=True
        )
        if st.button("✕  Remove dataset", key="remove_dataset"):
            st.session_state.uploaded_db   = None
            st.session_state.active_schema = None
            st.session_state.active_table  = "cars"
            st.session_state.active_label  = None
            _clear_persisted_dataset()
            load_dynamic_meta.clear()
            load_dynamic_summary.clear()
            load_sidebar_breakdown.clear()
            st.rerun()

    # Theme toggle (persisted in URL so refresh keeps it)
    st.markdown('<div class="sb-sec-title">🎨 Theme</div>', unsafe_allow_html=True)
    toggled = st.toggle(TOGGLE_LABEL, value=st.session_state.dark_mode)
    if toggled != st.session_state.dark_mode:
        st.session_state.dark_mode = toggled
        st.query_params["theme"] = "dark" if toggled else "light"
        st.rerun()

    # Clear conversation
    st.markdown('<div class="sb-sec-title">🗂️ Conversation</div>', unsafe_allow_html=True)
    history_count = len(st.session_state.get("query_history", []))
    st.caption(f"{history_count} quer{'y' if history_count == 1 else 'ies'} in history")
    if st.button("🗑️  Clear Conversation", use_container_width=True):
        st.session_state.query_history  = []
        st.session_state.result_df      = None
        st.session_state.result_sql     = None
        st.session_state.result_err     = None
        st.session_state.result_insight = None
        _save_history([])   # wipe the file too
        st.rerun()

    # Examples — dynamic based on active dataset
    _ex_title = "💡 Example Questions" if _is_uploaded else "💡 Example Questions (BMW Demo)"
    st.markdown(f'<div class="sb-sec-title">{_ex_title}</div>', unsafe_allow_html=True)
    for ex in _suggestions:
        st.markdown(f'<div class="sb-chip">→ {ex}</div>', unsafe_allow_html=True)

    # Dataset overview — dynamic
    if _meta:
        st.markdown('<div class="sb-sec-title">📊 Dataset Overview</div>', unsafe_allow_html=True)
        _ov_rows = [
            ("Total Rows",    f"{_meta['total']:,}"),
            ("Total Columns", f"{_meta['n_cols']}"),
            ("Active Table",  _active_table),
        ]
        st.markdown("".join(
            f'<div class="sb-stat"><span class="k">{k}</span><span class="v">{v}</span></div>'
            for k, v in _ov_rows
        ), unsafe_allow_html=True)

    # Column breakdowns — dynamic top-value counts per text column
    if _breakdowns:
        for _bd_col, _bd_df in _breakdowns.items():
            _bd_title = _bd_col.replace("_", " ").title()
            st.markdown(
                f'<div class="sb-sec-title">📂 By {_bd_title}</div>',
                unsafe_allow_html=True
            )
            st.markdown("".join(
                f'<div class="sb-stat"><span class="k">{row["val"]}</span><span class="v">{int(row["n"]):,}</span></div>'
                for _, row in _bd_df.iterrows()
            ), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN — TOP BAR + HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="top-bar"></div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="hero">
    <div class="hero-eyebrow">⚡ Powered by Groq · LLaMA 3.3 70B · SQLite &nbsp;·&nbsp; Works with any structured dataset</div>
    <h1>Ask questions about<br>your <em>data</em> in plain English</h1>
    <p class="hero-sub">
        The AI converts your question into SQL, runs it against the database,
        and visualises insights instantly — no coding required.
    </p>
    <div style="margin-top:1rem;display:inline-flex;align-items:center;gap:0.5rem;
                background:rgba(255,255,255,0.06);border:1px solid {BORDER};
                border-radius:8px;padding:0.35rem 0.85rem;">
        <span style="font-size:0.7rem;color:{TEXT_MUTED};font-weight:600;
                     text-transform:uppercase;letter-spacing:0.12em;">
            {"Active dataset:" if _is_uploaded else "Demo dataset:"}
        </span>
        <span style="font-size:0.78rem;color:{TEXT_BODY};font-weight:500;">
            {_active_label}
        </span>
    </div>
    <div class="hero-input-wrap">
        <div class="hero-input-inner">
""", unsafe_allow_html=True)

question = st.text_area(
    label="question",
    placeholder="e.g. What are the top 10 most expensive cars?",
    height=72,
    label_visibility="collapsed",
    key="question_input",
)

# Run button
b1, _ = st.columns([1.1, 7])
with b1:
    run_btn = st.button("▶  Run Analysis", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# QUERY EXECUTION
# ══════════════════════════════════════════════════════════════════════════════
if run_btn:
    if not question.strip():
        st.markdown('<div class="warn-box">⚠️ Please enter a question first.</div>',
                    unsafe_allow_html=True)
    else:
        st.session_state.result_df      = None
        st.session_state.result_sql     = None
        st.session_state.result_err     = None
        st.session_state.result_insight = None

        with st.spinner("🤖 Generating SQL…"):
            try:
                sql = generate_sql(
                    question,
                    schema=get_active_schema(),
                    table_name=get_active_table(),
                )
                st.session_state.result_sql = sql
            except Exception as exc:
                st.session_state.result_err = f"SQL generation failed: {exc}"

        if st.session_state.result_sql and not st.session_state.result_err:
            with st.spinner("⚡ Querying database…"):
                try:
                    st.session_state.result_df = execute_query_on(st.session_state.result_sql)
                except Exception as exc:
                    st.session_state.result_err = f"Query execution failed: {exc}"

        # Generate AI insight from the results
        if (st.session_state.result_df is not None
                and not st.session_state.result_df.empty
                and not st.session_state.result_err):
            with st.spinner("💡 Generating AI insight…"):
                try:
                    _df   = st.session_state.result_df
                    _rows = _df.head(20).to_string(index=False)
                    _cols = ", ".join(_df.columns.tolist())
                    _insight_prompt = (
                        f"A user asked: \"{question}\"\n\n"
                        f"The query returned {len(_df)} rows with columns: {_cols}.\n"
                        f"Here is a sample of the data:\n{_rows}\n\n"
                        "Write a concise 2-4 sentence insight for a non-technical business user. "
                        "Describe what the results mean, highlight key numbers, note any trends or "
                        "interesting observations. Do NOT mention SQL, tables, or technical terms. "
                        "Respond with plain text only — no bullet points, no markdown, no headers."
                    )
                    _groq_client = _Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
                    _resp = _groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        temperature=0.4,
                        max_tokens=200,
                        messages=[
                            {"role": "system", "content": "You are a friendly business intelligence analyst who explains data insights in plain English."},
                            {"role": "user",   "content": _insight_prompt},
                        ],
                    )
                    st.session_state.result_insight = _resp.choices[0].message.content.strip()
                except Exception:
                    st.session_state.result_insight = None  # silently skip if insight fails

        # ── Append to query history ───────────────────────────────────────────
        if st.session_state.result_sql and not st.session_state.result_err:
            st.session_state.query_history.insert(0, {
                "question": question,
                "sql":      st.session_state.result_sql,
                "insight":  st.session_state.result_insight,
                "df":       st.session_state.result_df.copy() if st.session_state.result_df is not None else None,
            })
            # Keep max 10 entries
            st.session_state.query_history = st.session_state.query_history[:10]
            _save_history(st.session_state.query_history)

# ══════════════════════════════════════════════════════════════════════════════
# DATASET INTELLIGENCE — dynamic, works for any dataset
# ══════════════════════════════════════════════════════════════════════════════

if _meta:
    st.markdown('''
    <div class="section-hd">
        <span class="section-hd-text">🧠 Dataset Intelligence</span>
        <div class="section-hd-line"></div>
    </div>''', unsafe_allow_html=True)

    # Build KPI cards dynamically from detected stats
    _di_icons   = ["📊","🔢","📈","📉","🏆","🔤"]
    _di_kpis    = []

    # Always show total rows + columns
    _di_kpis.append(("📋", "Total Rows",    f"{_meta['total']:,}",  "records in dataset"))
    _di_kpis.append(("📐", "Total Columns", f"{_meta['n_cols']}",    "fields detected"))

    # Numeric averages
    for _ni, (_nc, _nv) in enumerate(_meta.get("num_stats", {}).items()):
        _icon = ["💹","📊","🔢"][_ni % 3]
        _di_kpis.append((_icon, f"Avg {_nc.replace('_',' ').title()}", f"{_nv:,.2f}", "average value"))

    # Categorical top values
    for _ci, (_cc, _cv) in enumerate(_meta.get("cat_stats", {}).items()):
        _icon = ["🏆","🔖"][_ci % 2]
        _di_kpis.append((_icon, f"Top {_cc.replace('_',' ').title()}", str(_cv), "most frequent"))

    _n_kpi_cols = min(len(_di_kpis), 6)
    _di_cols    = st.columns(_n_kpi_cols)
    for _di_i, (_di_col, (_di_icon, _di_label, _di_val, _di_sub)) in enumerate(
            zip(_di_cols, _di_kpis[:_n_kpi_cols])):
        _di_grad, _di_glow = KPI_COLORS[_di_i % len(KPI_COLORS)]
        with _di_col:
            st.markdown(f"""
            <div class="kpi-card" style="animation-delay:{_di_i*0.07}s">
                <div class="kpi-orb" style="background:radial-gradient(circle,{_di_glow} 0%,transparent 70%);"></div>
                <span class="kpi-icon">{_di_icon}</span>
                <div class="kpi-label">{_di_label}</div>
                <div class="kpi-value">{_di_val}</div>
                <div class="kpi-sub">{_di_sub}</div>
                <div class="kpi-bar" style="background:linear-gradient({_di_grad});"></div>
            </div>
            """, unsafe_allow_html=True)

    # AI-generated summary
    if _ds_summary:
        st.markdown(f"""
        <div class="di-section">
            <div class="di-header">
                <div class="di-header-icon">🤖</div>
                <div>
                    <div class="di-header-text">AI Dataset Summary</div>
                    <div class="di-header-sub">Auto-generated analysis · {_active_label}</div>
                </div>
            </div>
            <div class="di-summary-label">🧠 &nbsp; Insight</div>
            <div class="di-summary">{_ds_summary}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# QUERY HISTORY — single collapsible expander
# ══════════════════════════════════════════════════════════════════════════════

_history = st.session_state.get("query_history", [])

if _history:
    _btn_label = f"Hide History" if st.session_state.show_history else f"Open History ({len(_history)} {'query' if len(_history)==1 else 'queries'})"
    if st.button(_btn_label, key="toggle_history_btn"):
        st.session_state.show_history = not st.session_state.show_history
        st.rerun()

    if st.session_state.show_history:
        st.markdown("<br>", unsafe_allow_html=True)
        for _i, _entry in enumerate(_history):
            _h_df      = _entry.get("df")
            _h_num     = _h_df.select_dtypes(include="number").columns.tolist() if _h_df is not None else []
            _h_txt     = _h_df.select_dtypes(exclude="number").columns.tolist() if _h_df is not None else []
            _row_count = len(_h_df) if _h_df is not None else 0
            _insight   = _entry.get("insight") or ""
            _badge_num = len(_history) - _i

            with st.container():
                st.markdown(f"""
                <div class="hist-card">
                    <div class="hist-card-header">
                        <div class="hist-question">💬 {_entry["question"]}</div>
                        <div class="hist-badge">#{_badge_num}</div>
                    </div>
                    {"<div class=\"hist-insight\">🤖 " + _insight + "</div>" if _insight else ""}
                    <div class="hist-num">📊 {_row_count:,} rows returned</div>
                </div>
                """, unsafe_allow_html=True)

                if _h_df is not None and not _h_df.empty and _h_num:
                    _hy = _h_num[0]
                    _hx = _h_txt[0] if _h_txt else _h_df.columns[0]
                    _hn = len(_h_df)
                    _TL = dict(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PLOT_BG,
                        font=dict(family="Space Grotesk", color=FONT_CLR, size=10),
                        margin=dict(l=5, r=5, t=10, b=5), height=200,
                        xaxis=dict(gridcolor=GRID_CLR, tickfont=dict(size=9)),
                        yaxis=dict(gridcolor=GRID_CLR, tickfont=dict(size=9)),
                        coloraxis_showscale=False,
                    )
                    if _hn <= 30:
                        _fig = px.bar(
                            _h_df.sort_values(_hy, ascending=False).head(10),
                            x=_hx, y=_hy, color=_hy,
                            color_continuous_scale=CHART_SEQ,
                            labels={_hx: _hx.replace("_"," ").title(), _hy: _hy.replace("_"," ").title()},
                        )
                    else:
                        _fig = px.scatter(
                            _h_df, x=_hx, y=_hy, color=_hy,
                            color_continuous_scale=CHART_SEQ, opacity=0.7,
                            labels={_hx: _hx.replace("_"," ").title(), _hy: _hy.replace("_"," ").title()},
                        )
                    _fig.update_layout(**_TL)
                    _fig.update_traces(marker_line_width=0)
                    st.plotly_chart(_fig, use_container_width=True, key=f"hist_chart_{_i}")

            if _i < len(_history) - 1:
                st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.result_err:
    st.markdown(f'<div class="err-box">❌ {st.session_state.result_err}</div>',
                unsafe_allow_html=True)

elif st.session_state.result_sql:
    df           = st.session_state.result_df
    numeric_cols = df.select_dtypes(include="number").columns.tolist() if df is not None else []
    text_cols    = df.select_dtypes(exclude="number").columns.tolist() if df is not None else []



    if df is None or df.empty:
        st.markdown('<div class="warn-box">⚠️ The query returned no results. Try rephrasing your question.</div>',
                    unsafe_allow_html=True)

    else:
        st.markdown('<div class="results-wrap">', unsafe_allow_html=True)
        n = len(df)

        # ── INSIGHTS — KPI cards ──────────────────────────────────────────────
        def _section(icon, label):
            st.markdown(f"""
            <div class="section-hd">
                <span class="section-hd-text">{icon} {label}</span>
                <div class="section-hd-line"></div>
            </div>""", unsafe_allow_html=True)

        _section("📌", "Insights")

        # Build KPIs from query result
        kpi_data = []
        if numeric_cols:
            col = numeric_cols[0]
            vals = df[col].dropna()
            kpi_data = [
                ("🔢", "Row Count",    f"{n:,}",          "rows returned"),
                ("📈", f"Max {col}",   f"{vals.max():,.1f}", f"highest {col}"),
                ("📊", f"Avg {col}",   f"{vals.mean():,.1f}","mean value"),
                ("📉", f"Min {col}",   f"{vals.min():,.1f}", f"lowest {col}"),
            ]
        else:
            _ds_total = _meta["total"] if _meta else "—"
            kpi_data = [
                ("🔢", "Row Count",   f"{n:,}",             "rows returned"),
                ("📋", "Columns",     f"{len(df.columns)}", "in result"),
                ("📊", "Dataset Rows",f"{_ds_total:,}" if isinstance(_ds_total, int) else str(_ds_total), "total in dataset"),
                ("🔍", "Query",       "SELECT", "executed successfully"),
            ]

        k_cols = st.columns(4)
        for idx, (k_col, (icon, label, value, sub)) in enumerate(zip(k_cols, kpi_data)):
            grad, glow = KPI_COLORS[idx % len(KPI_COLORS)]
            with k_col:
                st.markdown(f"""
                <div class="kpi-card" style="animation-delay:{idx*0.08}s">
                    <div class="kpi-orb" style="background:radial-gradient(circle,{glow} 0%,transparent 70%);"></div>
                    <span class="kpi-icon">{icon}</span>
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-sub">{sub}</div>
                    <div class="kpi-bar" style="background:linear-gradient({grad});"></div>
                </div>
                """, unsafe_allow_html=True)

        # ── AI INSIGHT ───────────────────────────────────────────────────────────
        if st.session_state.get("result_insight"):
            _section("💡", "AI Insight")
            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-avatar">
                    <div class="insight-avatar-icon">🤖</div>
                    <div class="insight-avatar-label">AI Analysis · Data Insight</div>
                </div>
                <div class="insight-text">{st.session_state.result_insight}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── VISUALIZATION ─────────────────────────────────────────────────────
        if numeric_cols:
            _section("📊", "Visualization")

            y   = numeric_cols[0]
            x   = text_cols[0] if text_cols else df.columns[0]
            TL  = dict(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PLOT_BG,
                font=dict(family="Space Grotesk", color=FONT_CLR, size=11),
                margin=dict(l=10,r=10,t=35,b=10),
                xaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR, tickfont=dict(size=10)),
                yaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR, tickfont=dict(size=10)),
                coloraxis_showscale=False,
            )

            if n <= 20:
                # Horizontal bar
                fig = px.bar(
                    df.sort_values(y, ascending=True).tail(15),
                    x=y, y=x, orientation="h",
                    color=y, color_continuous_scale=CHART_SEQ,
                    labels={x: x.replace("_"," ").title(), y: y.replace("_"," ").title()},
                )
                fig.update_traces(marker_line_width=0, marker_cornerradius=4)
            elif n <= 80:
                fig = px.bar(
                    df, x=x, y=y,
                    color=y, color_continuous_scale=CHART_SEQ,
                    labels={x: x.replace("_"," ").title(), y: y.replace("_"," ").title()},
                )
                fig.update_traces(marker_line_width=0, marker_cornerradius=4)
            else:
                c2 = numeric_cols[1] if len(numeric_cols) > 1 else y
                fig = px.scatter(
                    df, x=x, y=y, color=c2,
                    color_continuous_scale=CHART_SEQ, opacity=0.75,
                    labels={x: x.replace("_"," ").title(), y: y.replace("_"," ").title()},
                )
            fig.update_layout(**TL)

            c_left, c_right = st.columns([3, 2], gap="large")
            with c_left:
                st.plotly_chart(fig, use_container_width=True)

            # Second chart — pie or scatter
            with c_right:
                if text_cols and n <= 30:
                    # Pie / donut
                    fig2 = go.Figure(go.Pie(
                        labels=df[x], values=df[y],
                        hole=0.55,
                        marker=dict(colors=CHART_DISC, line=dict(color=BG_PAGE, width=2)),
                        textfont=dict(family="Space Grotesk", size=11, color=FONT_CLR),
                    ))
                    fig2.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Space Grotesk", color=FONT_CLR),
                        margin=dict(l=10,r=10,t=35,b=10),
                        showlegend=True,
                        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=FONT_CLR, size=10)),
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                elif len(numeric_cols) >= 2:
                    fig2 = px.scatter(
                        df, x=numeric_cols[0], y=numeric_cols[1],
                        color=text_cols[0] if text_cols else None,
                        color_discrete_sequence=CHART_DISC, opacity=0.8,
                        labels={c: c.replace("_"," ").title() for c in df.columns},
                    )
                    fig2.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PLOT_BG,
                        font=dict(family="Space Grotesk", color=FONT_CLR, size=11),
                        margin=dict(l=10,r=10,t=35,b=10),
                        xaxis=dict(gridcolor=GRID_CLR), yaxis=dict(gridcolor=GRID_CLR),
                        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=FONT_CLR)),
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(fig2, use_container_width=True)

        # ── RESULT TABLE ──────────────────────────────────────────────────────
        _section("📋", "Result Table")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(f"✓ {n:,} row(s) · {len(df.columns)} column(s) returned")

        st.markdown('</div>', unsafe_allow_html=True)