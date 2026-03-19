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

# ══════════════════════════════════════════════════════════════════════════════
# SMART CHART SELECTION — question + dataframe → chart type
# ══════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------

# Keyword lists are ordered by specificity: more-specific patterns first so
# that a question like "percentage breakdown by year" resolves to "share"
# rather than "trend".

_TREND_KW = frozenset([
    "trend", "over time", "time series", "over the year", "over the years",
    "by year", "by month", "by quarter", "by week", "by day", "daily",
    "monthly", "quarterly", "yearly", "annually", "timeline", "growth",
    "change over", "historical", "progression", "evolution", "time",
])

_SHARE_KW = frozenset([
    "share", "proportion", "percentage", "percent", "% of", "ratio",
    "composition", "make up", "makes up", "portion", "fraction",
    "pie", "split", "breakdown by",
])

_DIST_KW = frozenset([
    "distribution", "histogram", "spread", "range", "frequency",
    "how many", "count of", "how often", "bins", "bucket", "buckets",
])

_RELATION_KW = frozenset([
    "relationship", "correlation", "correlate", "relate", "impact",
    "affect", "influence", "scatter", "vs", "versus", "against",
    "price vs", "mileage vs", "mpg vs", "compared to",
])

_COMPARE_KW = frozenset([
    "compare", "comparison", "difference between", "which is",
    "rank", "ranking", "top", "bottom", "best", "worst",
    "highest", "lowest", "most", "least", "average", "avg",
    "by fuel", "by transmission", "by model", "by type", "by category",
    "between", "cheapest", "expensive",
])


def _question_intent(q: str, numeric_cols: list, text_cols: list,
                     time_col: "str | None", n: int) -> str:
    """
    Return one of: "trend" | "share" | "distribution" | "relation" | "compare"

    Priority order (highest → lowest):
      1. Trend keywords  (most unambiguous — user explicitly wants time axis)
      2. Share keywords  (explicitly proportional)
      3. Distribution    (explicitly about spread/frequency)
      4. Relation        (two-numeric correlation, or explicit scatter ask)
      5. Compare         (everything else with categories + numbers)
      6. Structural fall-through
    """
    ql = q.lower()

    def _hit(kws: frozenset) -> bool:
        return any(kw in ql for kw in kws)

    # 1. Trend — explicit time words OR a detected time column in the result
    if _hit(_TREND_KW) or (time_col is not None):
        return "trend"

    # 2. Share — proportion / percentage intent (check before compare so
    #    "percentage breakdown" doesn't collapse to compare)
    if _hit(_SHARE_KW) and text_cols and numeric_cols:
        return "share"

    # 3. Distribution — spread/frequency questions
    if _hit(_DIST_KW) and numeric_cols:
        return "distribution"

    # 4. Relationship — scatter / correlation questions or 2 numeric cols
    if (_hit(_RELATION_KW) and len(numeric_cols) >= 2) or (
        len(numeric_cols) >= 2 and not text_cols and n > 20
    ):
        return "relation"

    # 5. Comparison — explicit compare words or default when we have
    #    a text category column alongside a numeric one
    if _hit(_COMPARE_KW) or (text_cols and numeric_cols):
        return "compare"

    # 6. Structural fall-through
    if len(numeric_cols) >= 2:
        return "relation"
    return "compare"


# ---------------------------------------------------------------------------
# Pie-safety guard
# ---------------------------------------------------------------------------

def _pie_safe(df: "pd.DataFrame", x_col: str) -> bool:
    """
    Return True only when a pie chart would be meaningful:
      - x column exists
      - more than one unique category
      - not too many slices (≤ 12 is readable)
    """
    if x_col not in df.columns:
        return False
    n_unique = df[x_col].nunique()
    return 1 < n_unique <= 12


# ---------------------------------------------------------------------------
# Main config builder
# ---------------------------------------------------------------------------

def _pick_chart_config(question: str, df: "pd.DataFrame") -> dict:
    """
    Decide the best primary + secondary chart types based on:
      1. Keyword signals in the user's question
      2. The structure of the resulting DataFrame

    Returns a dict:
      intent, primary, secondary, x, y, y2, color_col, chart_label
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    text_cols    = df.select_dtypes(exclude="number").columns.tolist()
    n            = len(df)

    # Detect time-axis column by name heuristic
    _TIME_NAME_KWS = ("year", "date", "month", "week", "quarter", "period", "day", "time")
    time_col = next(
        (c for c in df.columns if any(kw in c.lower() for kw in _TIME_NAME_KWS)),
        None,
    )

    intent = _question_intent(question, numeric_cols, text_cols, time_col, n)

    # ── Assign x / y columns based on intent ─────────────────────────────────
    y  = numeric_cols[0] if numeric_cols else (df.columns[-1] if len(df.columns) else None)
    y2 = numeric_cols[1] if len(numeric_cols) > 1 else y

    if intent == "trend":
        # x must be the time axis
        if time_col:
            x = time_col
        elif text_cols:
            x = text_cols[0]
        else:
            # All-numeric: treat first col as x, second as y
            x = numeric_cols[0]
            y = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
    elif intent == "relation":
        # Both axes are numeric
        x = numeric_cols[0]
        y = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
    else:
        x = text_cols[0] if text_cols else df.columns[0]

    # ── Primary chart ─────────────────────────────────────────────────────────
    if intent == "trend":
        primary = "line"

    elif intent == "share":
        # Guard: only use pie if there are multiple distinct categories
        primary = "pie" if _pie_safe(df, x) else "bar"

    elif intent == "distribution":
        primary = "histogram"

    elif intent == "relation":
        primary = "scatter"

    else:  # compare (default)
        # Horizontal bar for ≤ 15 rows (easier to read labels), vertical for more
        primary = "bar_h" if n <= 15 else "bar"

    # ── Secondary chart (companion panel) ─────────────────────────────────────
    secondary: "str | None"
    if intent == "trend":
        # Show a secondary bar to surface top performers alongside the trend
        secondary = "bar_h" if (text_cols and n <= 20) else None

    elif intent == "share":
        # Companion bar for the same data — easier to compare exact values
        secondary = "bar_h" if n <= 15 else "bar"

    elif intent == "distribution":
        # No useful companion for a histogram
        secondary = None

    elif intent == "relation":
        # Companion bar/pie if there are categories to break down by
        if text_cols:
            secondary = "bar_h" if n <= 15 else "bar"
        elif len(numeric_cols) >= 3:
            secondary = "scatter"   # second scatter with a different y axis
        else:
            secondary = None

    else:  # compare
        # Companion pie only when safe; otherwise a second numeric scatter
        if _pie_safe(df, x):
            secondary = "pie"
        elif len(numeric_cols) >= 2:
            secondary = "scatter"
        else:
            secondary = None

    # ── Human-readable label ──────────────────────────────────────────────────
    _labels = {
        "trend":        "Trend Over Time",
        "share":        "Share & Proportion",
        "distribution": "Distribution",
        "relation":     "Relationship",
        "compare":      "Category Comparison",
    }

    return dict(
        intent      = intent,
        primary     = primary,
        secondary   = secondary,
        x           = x,
        y           = y,
        y2          = y2,
        color_col   = text_cols[0] if text_cols and intent == "relation" else None,
        chart_label = _labels[intent],
    )


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def _pie_layout(CHART_DISC, BG_PAGE, FONT_CLR) -> dict:
    """Shared layout kwargs for a donut/pie chart."""
    return dict(
        paper_bgcolor = "rgba(0,0,0,0)",
        font          = dict(family="Space Grotesk", color=FONT_CLR),
        margin        = dict(l=10, r=10, t=35, b=10),
        showlegend    = True,
        legend        = dict(bgcolor="rgba(0,0,0,0)", font=dict(color=FONT_CLR, size=10)),
    )


def _build_primary_fig(cfg: dict, df: "pd.DataFrame", TL: dict) -> "go.Figure":
    """
    Build the primary Plotly figure from a chart config dict.
    All bar charts are sorted descending and capped at 15 rows.
    Pie charts are only rendered when _pie_safe() is satisfied (guaranteed
    by _pick_chart_config, but double-checked here as a safety net).
    """
    p            = cfg["primary"]
    x            = cfg["x"]
    y            = cfg["y"]
    n            = len(df)
    text_cols    = df.select_dtypes(exclude="number").columns.tolist()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    lbl          = {c: c.replace("_", " ").title() for c in df.columns}

    # ── Line chart — sorted by x axis ─────────────────────────────────────────
    if p == "line":
        _color = text_cols[0] if (text_cols and text_cols[0] != x) else None
        fig = px.line(
            df.sort_values(x) if x in df.columns else df,
            x=x, y=y,
            color=_color,
            color_discrete_sequence=CHART_DISC,
            labels=lbl,
            markers=True,
        )
        fig.update_traces(line_width=2.5, marker_size=6)
        fig.update_layout(**TL)

    # ── Horizontal bar — sorted desc, top-15 ──────────────────────────────────
    elif p == "bar_h":
        _df_plot = df.sort_values(y, ascending=False).head(15)
        # Re-sort ascending so the tallest bar is at the top of a horizontal chart
        _df_plot = _df_plot.sort_values(y, ascending=True)
        fig = px.bar(
            _df_plot,
            x=y, y=x, orientation="h",
            color=y, color_continuous_scale=CHART_SEQ,
            labels=lbl,
        )
        fig.update_traces(marker_line_width=0, marker_cornerradius=4)
        fig.update_layout(**TL)

    # ── Vertical bar — sorted desc, top-15 ────────────────────────────────────
    elif p == "bar":
        _df_plot = df.sort_values(y, ascending=False).head(15)
        fig = px.bar(
            _df_plot,
            x=x, y=y,
            color=y, color_continuous_scale=CHART_SEQ,
            labels=lbl,
        )
        fig.update_traces(marker_line_width=0, marker_cornerradius=4)
        fig.update_layout(**TL)

    # ── Histogram ─────────────────────────────────────────────────────────────
    elif p == "histogram":
        fig = px.histogram(
            df, x=y,
            color=text_cols[0] if text_cols else None,
            color_discrete_sequence=CHART_DISC,
            nbins=min(30, max(10, n // 5)),
            labels=lbl,
            opacity=0.85,
        )
        fig.update_traces(marker_line_width=0)
        fig.update_layout(**TL)

    # ── Pie / donut — with fallback to bar if not pie-safe ────────────────────
    elif p == "pie":
        if _pie_safe(df, x):
            fig = go.Figure(go.Pie(
                labels=df[x],
                values=df[y],
                hole=0.52,
                marker=dict(colors=CHART_DISC, line=dict(color=BG_PAGE, width=2)),
                textfont=dict(family="Space Grotesk", size=11, color=FONT_CLR),
            ))
            fig.update_layout(**_pie_layout(CHART_DISC, BG_PAGE, FONT_CLR))
            return fig   # pie manages its own layout — return early
        else:
            # Fallback: render as a sorted horizontal bar
            _df_plot = df.sort_values(y, ascending=False).head(15)
            _df_plot = _df_plot.sort_values(y, ascending=True)
            fig = px.bar(
                _df_plot,
                x=y, y=x, orientation="h",
                color=y, color_continuous_scale=CHART_SEQ,
                labels=lbl,
            )
            fig.update_traces(marker_line_width=0, marker_cornerradius=4)
            fig.update_layout(**TL)

    # ── Scatter — with OLS trendline when enough points ───────────────────────
    elif p == "scatter":
        x_col = x if x in numeric_cols else (numeric_cols[0] if numeric_cols else x)
        y_col = y if y in numeric_cols else (numeric_cols[1] if len(numeric_cols) > 1 else y)
        _color_arg = cfg.get("color_col") or (text_cols[0] if text_cols else None)
        fig = px.scatter(
            df, x=x_col, y=y_col,
            color=_color_arg,
            color_discrete_sequence=CHART_DISC,
            opacity=0.80,
            labels=lbl,
            trendline=None,  # statsmodels not required
        )
        fig.update_traces(marker_size=8, marker_line_width=0)
        fig.update_layout(**TL)

    # ── Safe fallback ─────────────────────────────────────────────────────────
    else:
        _df_plot = df.sort_values(y, ascending=False).head(15)
        fig = px.bar(_df_plot, x=x, y=y, color=y,
                     color_continuous_scale=CHART_SEQ, labels=lbl)
        fig.update_traces(marker_line_width=0, marker_cornerradius=4)
        fig.update_layout(**TL)

    return fig


def _build_secondary_fig(cfg: dict, df: "pd.DataFrame", TL: dict) -> "go.Figure | None":
    """
    Build the secondary/companion Plotly figure.
    Returns None when no meaningful companion chart can be built.
    All bar charts are sorted descending and capped at 15 rows.
    Pie charts are guarded by _pie_safe().
    """
    s            = cfg["secondary"]
    if s is None:
        return None

    x            = cfg["x"]
    y            = cfg["y"]
    y2           = cfg["y2"]
    text_cols    = df.select_dtypes(exclude="number").columns.tolist()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    lbl          = {c: c.replace("_", " ").title() for c in df.columns}
    n            = len(df)

    # ── Pie companion ─────────────────────────────────────────────────────────
    if s == "pie":
        if not (text_cols and numeric_cols and _pie_safe(df, x)):
            return None
        fig2 = go.Figure(go.Pie(
            labels=df[x],
            values=df[y],
            hole=0.52,
            marker=dict(colors=CHART_DISC, line=dict(color=BG_PAGE, width=2)),
            textfont=dict(family="Space Grotesk", size=11, color=FONT_CLR),
        ))
        fig2.update_layout(**_pie_layout(CHART_DISC, BG_PAGE, FONT_CLR))
        return fig2

    # ── Scatter companion ─────────────────────────────────────────────────────
    elif s == "scatter":
        if len(numeric_cols) < 2:
            return None
        fig2 = px.scatter(
            df,
            x=numeric_cols[0], y=numeric_cols[1],
            color=text_cols[0] if text_cols else None,
            color_discrete_sequence=CHART_DISC,
            opacity=0.80,
            labels=lbl,
        )
        fig2.update_layout(**TL)
        return fig2

    # ── Horizontal bar companion ──────────────────────────────────────────────
    elif s == "bar_h":
        if not (text_cols and numeric_cols):
            return None
        _df_plot = df.sort_values(y, ascending=False).head(15)
        _df_plot = _df_plot.sort_values(y, ascending=True)
        fig2 = px.bar(
            _df_plot,
            x=y, y=x, orientation="h",
            color=y, color_continuous_scale=CHART_SEQ,
            labels=lbl,
        )
        fig2.update_traces(marker_line_width=0, marker_cornerradius=4)
        fig2.update_layout(**TL)
        return fig2

    # ── Vertical bar companion ────────────────────────────────────────────────
    elif s == "bar":
        if not (text_cols and numeric_cols):
            return None
        _df_plot = df.sort_values(y, ascending=False).head(15)
        fig2 = px.bar(
            _df_plot,
            x=x, y=y,
            color=y, color_continuous_scale=CHART_SEQ,
            labels=lbl,
        )
        fig2.update_traces(marker_line_width=0, marker_cornerradius=4)
        fig2.update_layout(**TL)
        return fig2

    # ── Line companion ────────────────────────────────────────────────────────
    elif s == "line":
        if not numeric_cols:
            return None
        fig2 = px.line(
            df.sort_values(x) if x in df.columns else df,
            x=x, y=y2,
            labels=lbl,
            markers=True,
            color_discrete_sequence=CHART_DISC,
        )
        fig2.update_traces(line_width=2.5, marker_size=6)
        fig2.update_layout(**TL)
        return fig2

    return None


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPER FUNCTIONS — error display, guidance panel, suggestion runner
# ══════════════════════════════════════════════════════════════════════════════

def _get_dataset_columns() -> list[str]:
    """Return column names for the active dataset (used in error/guidance panels)."""
    try:
        db  = st.session_state.get("uploaded_db") or _DB_PATH
        tbl = st.session_state.get("active_table") or "cars"
        col_info = execute_query(f"PRAGMA table_info({tbl})", db_path=db)
        return col_info["name"].tolist()
    except Exception:
        return []


def _render_friendly_error(message: str, show_columns: bool = True) -> None:
    """Render a styled, friendly error card with optional column list."""
    cols_html = ""
    if show_columns:
        cols = _get_dataset_columns()
        if cols:
            chips = "".join(f'<span class="friendly-err-col">{c}</span>' for c in cols)
            cols_html = f"""
            <div style="margin-top:0.7rem;">
                <div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;
                            letter-spacing:0.14em;color:#f87171;margin-bottom:0.45rem;">
                    Available columns
                </div>
                <div class="friendly-err-cols">{chips}</div>
            </div>"""
    st.markdown(f"""
    <div class="friendly-err">
        <div class="friendly-err-title">⚠️ &nbsp;{message}</div>
        <div class="friendly-err-body">
            Try asking about the columns listed below, or rephrase your question
            using simpler terms like <em>average</em>, <em>top 10</em>, or <em>count by category</em>.
        </div>
        {cols_html}
    </div>""", unsafe_allow_html=True)


def _render_guidance_panel() -> None:
    """Render the 'What you can ask' guidance panel below the input."""
    cols = _get_dataset_columns()
    if not cols:
        return
    chips_html = "".join(f'<span class="guidance-col-chip">{c}</span>' for c in cols)

    # Pick example questions from the current suggestions list
    _ex = _suggestions[:5]
    examples_html = "".join(f"<span>{q}</span>" for q in _ex)

    st.markdown(f"""
    <div class="guidance-panel">
        <div class="guidance-title">💡 What you can ask</div>
        <div class="guidance-cols">{chips_html}</div>
        <div class="guidance-examples">{examples_html}</div>
    </div>""", unsafe_allow_html=True)


def _run_question(q: str) -> None:
    """Store a question as pending so it auto-runs on next render cycle."""
    st.session_state.pending_question = q
    st.rerun()


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI BI Dashboard — Ask Questions About Your Data",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session defaults ──────────────────────────────────────────────────────────
for k, v in {
    "dark_mode":           True,
    "result_df":           None,
    "result_sql":          None,
    "result_err":          None,
    "result_insight":      None,
    "result_empty_reason": None,
    "query_history":       _load_history(),
    "show_history":        False,
    "uploaded_db":         None,
    "active_schema":       None,
    "active_table":        "cars",
    "active_label":        None,
    "pending_question":    None,   # set by suggestion buttons → triggers auto-run
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
    # ── DARK: premium multi-surface AI analytics design system ────────────────
    # Five distinct dark tones create real depth between layers
    BG_PAGE        = "#0a0a0f"   # deepest — page background
    BG_SIDEBAR     = "#0c0c14"   # sidebar surface
    BG_CARD        = "#1a1a26"   # card surface — noticeably lighter than page
    BG_CARD2       = "#141420"   # slightly deeper card variant
    BG_INPUT       = "#0f0f18"   # input fields — between page and card
    BG_PANEL       = "#12121a"   # section panels
    BORDER         = "#26263a"   # subtle structural border
    BORDER_FOCUS   = "#7c3aed"   # purple focus/active state
    BORDER_SUBTLE  = "#1e1e2e"   # barely-visible inner borders
    TEXT_H         = "#f0eeff"   # headings — near white with cool tint
    TEXT_BODY      = "#a8a0cc"   # body — softer lavender gray
    TEXT_MUTED     = "#4a4468"   # muted — dimmed labels
    # Accent gradients — purple → green → blue
    GRAD_ACCENT    = "90deg,#7c3aed,#00f5a0,#00d4ff"
    GRAD_BTN       = "135deg,#7c3aed 0%,#5b21b6 50%,#4c1d95 100%"
    GRAD_BTN_H     = "135deg,#8b5cf6 0%,#7c3aed 100%"
    GRAD_HERO      = "135deg,#0a0a0f 0%,#110820 40%,#080f18 100%"
    GRAD_CARD      = "135deg,#1a1a26 0%,#141420 100%"
    GRAD_KPI       = "135deg,#1e1e2e 0%,#1a1a26 100%"
    # Glow — used only on interactive elements and hover states
    GLOW_BTN       = "rgba(124,58,237,0.45)"
    GLOW_INPUT     = "rgba(124,58,237,0.20)"
    GLOW_CARD      = "rgba(124,58,237,0.15)"
    GLOW_GREEN     = "rgba(0,245,160,0.10)"
    GLOW_BLUE      = "rgba(0,212,255,0.10)"
    SHADOW         = "0 4px 24px rgba(0,0,0,0.5), 0 1px 0 rgba(255,255,255,0.03)"
    SHADOW_DEEP    = "0 8px 40px rgba(0,0,0,0.7), 0 1px 0 rgba(255,255,255,0.04)"
    # Chart
    CHART_SEQ      = [[0,"#1a0a3a"],[0.4,"#7c3aed"],[0.7,"#00d4ff"],[1,"#00f5a0"]]
    CHART_DISC     = ["#7c3aed","#00f5a0","#00d4ff","#f472b6","#fb923c","#facc15"]
    PLOT_BG        = "#0a0a0f"
    GRID_CLR       = "#1a1a2e"
    FONT_CLR       = "#4a4468"
    KPI_COLORS     = [
        ("135deg,#7c3aed,#a855f7","rgba(124,58,237,0.25)"),
        ("135deg,#00b894,#00f5a0","rgba(0,184,148,0.25)"),
        ("135deg,#0284c7,#00d4ff","rgba(2,132,199,0.25)"),
        ("135deg,#db2777,#f472b6","rgba(219,39,119,0.25)"),
    ]
    SCROLLBAR_TH   = "#26263a"
    SCROLLBAR_TH_H = "#7c3aed"
    ERR_BG         = "#130a0a"; ERR_BORDER = "#3d1515"; ERR_TEXT = "#f87171"
    WARN_BG        = "#110e00"; WARN_BORDER= "#3d3000"; WARN_TEXT= "#fbbf24"
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
    BG_PANEL       = "#fce7f0"
    BORDER         = "#f9a8c9"
    BORDER_FOCUS   = "#e11d6a"
    BORDER_SUBTLE  = "#fdd5e6"
    TEXT_H         = "#1a0010"
    TEXT_BODY      = "#5a1a35"
    TEXT_MUTED     = "#c084a0"
    GRAD_ACCENT    = "90deg,#e11d6a,#fb7185,#f97316"
    GRAD_BTN       = "135deg,#e11d6a 0%,#f97316 100%"
    GRAD_BTN_H     = "135deg,#be185d 0%,#ea580c 100%"
    GRAD_HERO      = "135deg,#fff5f7 0%,#ffe4ef 50%,#fff5f7 100%"
    GRAD_CARD      = "135deg,#ffffff,#fff0f3"
    GRAD_KPI       = "135deg,#fdf2f5 0%,#ffffff 100%"
    GLOW_BTN       = "rgba(225,29,106,0.45)"
    GLOW_INPUT     = "rgba(225,29,106,0.18)"
    GLOW_CARD      = "rgba(225,29,106,0.08)"
    GLOW_GREEN     = "rgba(249,115,22,0.10)"
    GLOW_BLUE      = "rgba(225,29,106,0.08)"
    SHADOW         = "0 4px 32px rgba(225,29,106,0.10)"
    SHADOW_DEEP    = "0 8px 40px rgba(225,29,106,0.15), 0 1px 0 rgba(255,255,255,0.8)"
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
/* ═══════════════════════════════════════════════════════════════════════════
   SIDEBAR — forced always visible
   ═══════════════════════════════════════════════════════════════════════════ */
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
section[data-testid="stSidebar"][aria-expanded="false"],
section[data-testid="stSidebar"][aria-expanded="false"] > div:first-child {{
    width: 300px !important; min-width: 300px !important;
    margin-left: 0 !important; transform: none !important;
    visibility: visible !important;
}}

/* ═══════════════════════════════════════════════════════════════════════════
   FONTS & BASE RESET
   ═══════════════════════════════════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

*, html, body, [class*="css"] {{
    font-family: 'Space Grotesk', sans-serif !important;
    box-sizing: border-box;
}}

/* ═══════════════════════════════════════════════════════════════════════════
   BASE — layered surfaces
   ═══════════════════════════════════════════════════════════════════════════ */
.stApp {{ background: {BG_PAGE} !important; color: {TEXT_BODY} !important; }}
.block-container {{ padding: 0 2.5rem 3rem 2.5rem !important; max-width: 1500px !important; }}
footer {{ visibility: hidden !important; }}

header,
header[data-testid="stHeader"],
.stApp header,
[data-testid="stAppViewContainer"] > header,
[data-testid="stHeader"] {{
    background: {BG_SIDEBAR} !important;
    border-bottom: 1px solid {BORDER} !important;
}}
header *, header[data-testid="stHeader"] *, .stApp header *, [data-testid="stHeader"] * {{
    color: {TEXT_BODY} !important;
}}
header a, header button {{ color: {TEXT_H} !important; }}
header, header *, #MainMenu, #MainMenu *,
[data-testid="stHeader"], [data-testid="stHeader"] * {{ visibility: visible !important; }}

/* ═══════════════════════════════════════════════════════════════════════════
   SCROLLBAR
   ═══════════════════════════════════════════════════════════════════════════ */
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: {BG_PAGE}; }}
::-webkit-scrollbar-thumb {{
    background: {SCROLLBAR_TH};
    border-radius: 10px;
    transition: background 0.2s;
}}
::-webkit-scrollbar-thumb:hover {{ background: {SCROLLBAR_TH_H}; }}

/* ═══════════════════════════════════════════════════════════════════════════
   SIDEBAR SURFACE
   ═══════════════════════════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {{
    background: {BG_SIDEBAR} !important;
    border-right: 1px solid {BORDER} !important;
}}
section[data-testid="stSidebar"] * {{ color: {TEXT_BODY} !important; }}
section[data-testid="stSidebar"] button[aria-label*="ollapse"],
section[data-testid="stSidebar"] button[aria-label*="idebar"],
section[data-testid="stSidebar"] > div > button:first-of-type {{ display: none !important; }}
[data-testid="stSidebarResizeHandle"],
section[data-testid="stSidebar"] [role="separator"],
section[data-testid="stSidebar"] > div > [role="separator"],
section[data-testid="stSidebar"] div[style*="cursor"][style*="resize"] {{
    display: none !important;
    pointer-events: none !important;
}}
[data-testid="stSidebarResizeHandle"], [data-testid="stSidebarResizeHandle"] * {{
    font-size: 0 !important; line-height: 0 !important; visibility: hidden !important;
}}
button[aria-label*="ollapse"], button[aria-label*="idebar"], button[aria-label*="xpand"],
[data-testid="stSidebarCollapseButton"], [data-testid="stSidebarExpandButton"] {{
    display: none !important;
}}

/* ═══════════════════════════════════════════════════════════════════════════
   ANIMATIONS
   ═══════════════════════════════════════════════════════════════════════════ */
@keyframes shimmer {{
    0%   {{ background-position: -200% center; }}
    100% {{ background-position:  200% center; }}
}}
@keyframes heroIn {{
    from {{ opacity: 0; transform: translateY(-18px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes cardIn {{
    from {{ opacity: 0; transform: translateY(14px) scale(0.97); }}
    to   {{ opacity: 1; transform: translateY(0) scale(1); }}
}}
@keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes pulseGlow {{
    0%, 100% {{ box-shadow: 0 0 0 0 {GLOW_BTN}; }}
    50%       {{ box-shadow: 0 0 24px 4px {GLOW_BTN}; }}
}}

/* ═══════════════════════════════════════════════════════════════════════════
   ANIMATED TOP BAR
   ═══════════════════════════════════════════════════════════════════════════ */
.top-bar {{
    height: 2px;
    background: linear-gradient({GRAD_ACCENT});
    background-size: 200% auto;
    animation: shimmer 4s linear infinite;
    margin-bottom: 0;
}}

/* ═══════════════════════════════════════════════════════════════════════════
   HERO SECTION
   ═══════════════════════════════════════════════════════════════════════════ */
.hero {{
    background: linear-gradient({GRAD_HERO});
    border-bottom: 1px solid {BORDER};
    padding: 3.2rem 3rem 2.8rem 3rem;
    margin: 0 -2.5rem 2.5rem -2.5rem;
    position: relative;
    overflow: hidden;
    animation: heroIn 0.7s cubic-bezier(.4,0,.2,1) both;
}}
.hero::before {{
    content: '';
    position: absolute; top: -80px; right: -80px;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(124,58,237,0.18) 0%, transparent 60%);
    border-radius: 50%; pointer-events: none;
}}
.hero::after {{
    content: '';
    position: absolute; bottom: -60px; left: 15%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(0,245,160,0.08) 0%, transparent 60%);
    border-radius: 50%; pointer-events: none;
}}
.hero-eyebrow {{
    font-size: 0.62rem; font-weight: 700; letter-spacing: 0.24em;
    text-transform: uppercase; color: {TEXT_MUTED};
    margin-bottom: 0.8rem;
    display: flex; align-items: center; gap: 0.5rem;
}}
.hero-eyebrow::before {{
    content: '';
    width: 18px; height: 2px;
    background: linear-gradient(90deg, #7c3aed, #00f5a0);
    border-radius: 2px;
    display: inline-block;
}}
.hero h1 {{
    font-size: 2.8rem; font-weight: 700; letter-spacing: -0.04em;
    line-height: 1.08; color: {TEXT_H}; margin: 0 0 0.7rem 0;
}}
.hero h1 em {{
    font-style: normal;
    background: linear-gradient({GRAD_ACCENT});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; background-size: 200% auto;
    animation: shimmer 5s linear infinite;
}}
.hero-sub {{
    font-size: 0.95rem; color: {TEXT_MUTED}; font-weight: 400;
    max-width: 560px; line-height: 1.65;
}}

/* ═══════════════════════════════════════════════════════════════════════════
   HERO INPUT
   ═══════════════════════════════════════════════════════════════════════════ */
.hero-input-wrap {{
    position: relative; margin-top: 2.2rem; max-width: 860px;
}}
.hero-input-wrap::before {{
    content: '';
    position: absolute; inset: -1.5px;
    background: linear-gradient({GRAD_ACCENT});
    background-size: 200% auto;
    animation: shimmer 4s linear infinite;
    border-radius: 18px; z-index: 0; opacity: 0.7;
}}
.hero-input-inner {{
    position: relative; z-index: 1;
    background: {BG_INPUT}; border-radius: 16px; padding: 2px;
}}
.stTextArea textarea {{
    background: {BG_INPUT} !important;
    border: none !important; border-radius: 14px !important;
    color: {TEXT_H} !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.05rem !important; font-weight: 400 !important;
    line-height: 1.6 !important; padding: 1.1rem 1.4rem !important;
    transition: box-shadow 0.3s ease !important;
    resize: none !important; caret-color: {BORDER_FOCUS} !important;
}}
.stTextArea textarea:focus {{
    box-shadow: inset 0 0 0 1px rgba(124,58,237,0.4) !important;
    outline: none !important;
}}
.stTextArea textarea::placeholder {{ color: {TEXT_MUTED} !important; opacity: 1 !important; }}
.stTextArea label {{ display: none !important; }}
[data-testid="stTextAreaResizeHandle"] {{ display: none !important; }}

/* ═══════════════════════════════════════════════════════════════════════════
   BUTTONS
   ═══════════════════════════════════════════════════════════════════════════ */
.stButton > button {{
    background: linear-gradient({GRAD_BTN}) !important;
    color: #ffffff !important; border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.88rem !important; font-weight: 700 !important;
    letter-spacing: 0.05em !important; padding: 0.65rem 2rem !important;
    transition: all 0.25s cubic-bezier(.4,0,.2,1) !important;
    box-shadow: 0 2px 12px {GLOW_BTN}, inset 0 1px 0 rgba(255,255,255,0.1) !important;
}}
.stButton > button:hover {{
    background: linear-gradient({GRAD_BTN_H}) !important;
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 6px 28px {GLOW_BTN}, inset 0 1px 0 rgba(255,255,255,0.15) !important;
}}
.stButton > button:active {{ transform: translateY(0) scale(0.99) !important; }}

/* ═══════════════════════════════════════════════════════════════════════════
   DATASET INTELLIGENCE PANEL
   ═══════════════════════════════════════════════════════════════════════════ */
.di-section {{
    background: {BG_PANEL};
    border: 1px solid {BORDER};
    border-top: 2px solid #7c3aed;
    border-radius: 16px; padding: 1.8rem 2rem; margin-bottom: 2rem;
    position: relative; overflow: hidden;
    box-shadow: {SHADOW};
    animation: fadeUp 0.5s ease both;
    backdrop-filter: blur(8px);
}}
.di-section::after {{
    content: '';
    position: absolute; bottom: -50px; right: -50px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, {GLOW_GREEN} 0%, transparent 70%);
    border-radius: 50%; pointer-events: none;
}}
.di-header {{
    display: flex; align-items: center; gap: 0.8rem; margin-bottom: 1.4rem;
}}
.di-header-icon {{
    width: 34px; height: 34px; border-radius: 8px;
    background: linear-gradient({GRAD_BTN});
    display: flex; align-items: center; justify-content: center;
    font-size: 0.95rem; box-shadow: 0 2px 8px {GLOW_BTN}; flex-shrink: 0;
}}
.di-header-text {{ font-size: 0.95rem; font-weight: 700; color: {TEXT_H}; letter-spacing: -0.01em; }}
.di-header-sub {{ font-size: 0.7rem; color: {TEXT_MUTED}; font-weight: 400; margin-top: 0.1rem; }}
.di-summary {{
    background: {BG_INPUT};
    border: 1px solid {BORDER_SUBTLE};
    border-left: 3px solid {BORDER_FOCUS};
    border-radius: 10px; padding: 1rem 1.3rem;
    font-size: 0.91rem; color: {TEXT_BODY}; line-height: 1.7; margin-top: 1.4rem; font-style: italic;
}}
.di-summary-label {{
    font-size: 0.6rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.18em; color: {BORDER_FOCUS}; margin-bottom: 0.4rem;
    display: flex; align-items: center; gap: 0.4rem;
}}

/* ═══════════════════════════════════════════════════════════════════════════
   KPI CARDS
   ═══════════════════════════════════════════════════════════════════════════ */
.kpi-card {{
    background: linear-gradient({GRAD_KPI});
    border: 1px solid {BORDER}; border-radius: 16px;
    padding: 1.5rem 1.6rem; position: relative; overflow: hidden;
    box-shadow: {SHADOW};
    transition: transform 0.25s cubic-bezier(.4,0,.2,1),
                box-shadow 0.25s cubic-bezier(.4,0,.2,1),
                border-color 0.25s ease;
    animation: cardIn 0.5s cubic-bezier(.4,0,.2,1) both;
    cursor: default; backdrop-filter: blur(4px);
}}
.kpi-card:hover {{
    transform: translateY(-5px);
    box-shadow: {SHADOW_DEEP}, 0 0 32px {GLOW_CARD};
    border-color: {BORDER_FOCUS};
}}
.kpi-orb {{
    position: absolute; top: -40px; right: -40px;
    width: 120px; height: 120px; border-radius: 50%; pointer-events: none; opacity: 0.4;
}}
.kpi-icon {{ font-size: 1.5rem; margin-bottom: 0.75rem; display: block; }}
.kpi-label {{
    font-size: 0.62rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.18em; color: {TEXT_MUTED}; margin-bottom: 0.35rem;
}}
.kpi-value {{ font-size: 2.1rem; font-weight: 700; letter-spacing: -0.04em; color: {TEXT_H}; line-height: 1; }}
.kpi-sub {{ font-size: 0.7rem; font-weight: 500; margin-top: 0.4rem; color: {TEXT_MUTED}; }}
.kpi-bar {{ position: absolute; bottom: 0; left: 0; right: 0; height: 2px; border-radius: 0 0 16px 16px; }}

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION HEADERS
   ═══════════════════════════════════════════════════════════════════════════ */
.section-hd {{
    display: flex; align-items: center; gap: 0.7rem;
    margin: 2.4rem 0 1rem 0; animation: fadeUp 0.4s ease both;
}}
.section-hd-line {{
    flex: 1; height: 1px;
    background: linear-gradient(90deg, {BORDER}, transparent);
}}
.section-hd-text {{
    font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.2em; color: {TEXT_MUTED}; white-space: nowrap;
}}

/* ═══════════════════════════════════════════════════════════════════════════
   QUERY HISTORY CARDS
   ═══════════════════════════════════════════════════════════════════════════ */
.hist-wrap {{ animation: fadeUp 0.4s ease both; }}
.hist-card {{
    background: {BG_PANEL};
    border: 1px solid {BORDER}; border-radius: 14px;
    padding: 1.3rem 1.5rem; margin-bottom: 0.9rem;
    box-shadow: {SHADOW}; position: relative; overflow: hidden;
    transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
}}
.hist-card:hover {{
    border-color: {BORDER_FOCUS}; transform: translateY(-2px);
    box-shadow: {SHADOW}, 0 0 20px rgba(124,58,237,0.10);
}}
.hist-card-header {{
    display: flex; align-items: flex-start;
    justify-content: space-between; gap: 1rem; margin-bottom: 0.8rem;
}}
.hist-question {{ font-size: 0.92rem; font-weight: 600; color: {TEXT_H}; line-height: 1.4; flex: 1; }}
.hist-badge {{
    font-size: 0.6rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.14em;
    background: rgba(124,58,237,0.12); color: #a78bfa;
    border: 1px solid rgba(124,58,237,0.25);
    border-radius: 20px; padding: 0.16rem 0.6rem; white-space: nowrap; flex-shrink: 0;
}}
.hist-insight {{
    font-size: 0.82rem; color: {TEXT_BODY}; line-height: 1.65; font-style: italic;
    border-left: 2px solid rgba(124,58,237,0.4); padding-left: 0.85rem; margin-top: 0.3rem;
}}
.hist-num {{ font-size: 0.66rem; color: {TEXT_MUTED}; margin-top: 0.65rem; font-weight: 500; }}

/* ═══════════════════════════════════════════════════════════════════════════
   AI INSIGHT CARD
   ═══════════════════════════════════════════════════════════════════════════ */
.insight-card {{
    background: linear-gradient(135deg, {BG_CARD} 0%, {BG_CARD2} 100%);
    border: 1px solid {BORDER};
    border-left: 3px solid #7c3aed;
    border-radius: 14px; padding: 1.5rem 1.8rem; margin-bottom: 0.5rem;
    box-shadow: {SHADOW}, 0 0 40px rgba(124,58,237,0.10);
    position: relative; overflow: hidden;
    animation: fadeUp 0.5s cubic-bezier(.4,0,.2,1) both;
}}
.insight-card::before {{
    content: '';
    position: absolute; top: -30px; right: -30px;
    width: 120px; height: 120px;
    background: radial-gradient(circle, rgba(124,58,237,0.12) 0%, transparent 70%);
    border-radius: 50%; pointer-events: none;
}}
.insight-avatar {{ display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.85rem; }}
.insight-avatar-icon {{
    width: 30px; height: 30px; border-radius: 50%;
    background: linear-gradient({GRAD_BTN});
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85rem; flex-shrink: 0; box-shadow: 0 2px 8px {GLOW_BTN};
}}
.insight-avatar-label {{
    font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.18em; color: #a78bfa;
}}
.insight-text {{ font-size: 0.95rem; line-height: 1.78; color: {TEXT_BODY}; font-weight: 400; }}

/* ═══════════════════════════════════════════════════════════════════════════
   RESULTS AREA
   ═══════════════════════════════════════════════════════════════════════════ */
.results-wrap {{ animation: fadeUp 0.5s cubic-bezier(.4,0,.2,1) both; }}

/* ═══════════════════════════════════════════════════════════════════════════
   EXPANDER
   ═══════════════════════════════════════════════════════════════════════════ */
.stExpander > details > summary > span:first-child {{ display: none !important; }}
.stExpander > details > summary p {{
    color: {TEXT_BODY} !important; font-size: 0.86rem !important; font-weight: 600 !important;
}}
.stExpander {{
    background: {BG_PANEL} !important; border: 1px solid {BORDER} !important;
    border-radius: 12px !important; box-shadow: {SHADOW} !important; overflow: hidden !important;
}}
.stExpander > details > summary {{
    color: {TEXT_MUTED} !important; font-size: 0.82rem !important;
    font-weight: 600 !important; padding: 0.8rem 1rem !important;
    transition: color 0.2s ease !important;
}}
.stExpander > details > summary:hover {{ color: {EXPANDER_ICON} !important; }}
.stExpander > details > summary svg {{ fill: {EXPANDER_ICON} !important; }}
pre, code {{
    font-family: 'Fira Code', monospace !important; font-size: 0.78rem !important;
    background: {BG_INPUT} !important; color: {CODE_COLOR} !important;
    border-radius: 8px !important; border: none !important;
    border-left: 2px solid rgba(0,245,160,0.3) !important;
}}

/* ═══════════════════════════════════════════════════════════════════════════
   DATAFRAME
   ═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER} !important; border-radius: 12px !important;
    overflow: hidden !important; box-shadow: {SHADOW} !important;
    animation: fadeUp 0.45s ease both;
}}

/* ═══════════════════════════════════════════════════════════════════════════
   ERROR & WARNING BOXES
   ═══════════════════════════════════════════════════════════════════════════ */
.err-box {{
    background: {ERR_BG}; border: 1px solid {ERR_BORDER};
    border-left: 3px solid #ef4444; border-radius: 10px;
    padding: 1rem 1.3rem; color: {ERR_TEXT}; font-size: 0.87rem; font-weight: 500;
    animation: fadeUp 0.3s ease;
}}
.warn-box {{
    background: {WARN_BG}; border: 1px solid {WARN_BORDER};
    border-left: 3px solid #f59e0b; border-radius: 10px;
    padding: 1rem 1.3rem; color: {WARN_TEXT}; font-size: 0.87rem; font-weight: 500;
    animation: fadeUp 0.3s ease;
}}

/* ═══════════════════════════════════════════════════════════════════════════
   SIDEBAR CONTENT
   ═══════════════════════════════════════════════════════════════════════════ */
.sb-brand {{
    text-align: center; padding: 1.5rem 0 1.8rem 0;
    border-bottom: 1px solid {BORDER}; margin-bottom: 1.2rem;
}}
.sb-brand-icon {{ font-size: 2rem; margin-bottom: 0.4rem; }}
.sb-brand-name {{ font-size: 0.95rem; font-weight: 700; color: {TEXT_H}; letter-spacing: -0.02em; }}
.sb-brand-sub {{
    font-size: 0.62rem; text-transform: uppercase;
    letter-spacing: 0.16em; color: {TEXT_MUTED}; margin-top: 0.2rem;
}}
.sb-sec-title {{
    font-size: 0.6rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.2em; color: {TEXT_MUTED}; margin: 1.3rem 0 0.55rem 0;
}}
.sb-chip {{
    background: {BG_CARD}; border: 1px solid {BORDER};
    border-radius: 7px; padding: 0.45rem 0.75rem;
    font-size: 0.74rem; color: {TEXT_BODY}; margin-bottom: 0.3rem; display: block;
    transition: border-color 0.15s, color 0.15s, background 0.15s; cursor: pointer;
}}
.sb-chip:hover {{ border-color: {BORDER_FOCUS}; color: {TEXT_H}; background: {BG_PANEL}; }}
.sb-stat {{
    display: flex; justify-content: space-between;
    padding: 0.36rem 0; border-bottom: 1px solid {BORDER_SUBTLE}; font-size: 0.74rem;
}}
.sb-stat:last-child {{ border-bottom: none; }}
.sb-stat .k {{ color: {TEXT_MUTED}; }}
.sb-stat .v {{ color: {TEXT_H}; font-weight: 600; }}

/* ═══════════════════════════════════════════════════════════════════════════
   FILE UPLOADER
   ═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stFileUploader"] {{ background: transparent !important; }}
[data-testid="stFileUploader"] > div,
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploader"] section {{
    background: {BG_CARD} !important; border: 1px dashed {BORDER} !important;
    border-radius: 10px !important; color: {TEXT_BODY} !important;
    transition: border-color 0.2s ease !important;
}}
[data-testid="stFileUploader"] > div:hover,
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"]:hover {{
    border-color: {BORDER_FOCUS} !important;
}}
[data-testid="stFileUploader"] > div *,
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] *,
[data-testid="stFileUploader"] section * {{ color: {TEXT_BODY} !important; }}
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] small {{ color: {TEXT_MUTED} !important; }}
[data-testid="stFileUploader"] button {{
    background: linear-gradient({GRAD_BTN}) !important; color: #ffffff !important; border: none !important;
}}

/* ═══════════════════════════════════════════════════════════════════════════
   TOGGLE / DOWNLOAD / STATUS
   ═══════════════════════════════════════════════════════════════════════════ */
label[data-testid="stToggle"] span {{ color: {TEXT_BODY} !important; font-size: 0.84rem !important; }}

[data-testid="stDownloadButton"] > button {{
    background: linear-gradient({GRAD_BTN}) !important;
    color: #ffffff !important; border: none !important; border-radius: 9px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.82rem !important; font-weight: 700 !important;
    letter-spacing: 0.04em !important; padding: 0.5rem 1.4rem !important;
    margin-top: 0.6rem !important; transition: all 0.22s ease !important;
    box-shadow: 0 2px 12px {GLOW_BTN} !important;
}}
[data-testid="stDownloadButton"] > button:hover {{
    background: linear-gradient({GRAD_BTN_H}) !important;
    transform: translateY(-1px) !important; box-shadow: 0 5px 18px {GLOW_BTN} !important;
}}

[data-testid="stStatusWidget"],
[data-testid="stStatus"] {{
    background: {BG_PANEL} !important; border: 1px solid {BORDER} !important;
    border-radius: 12px !important; box-shadow: {SHADOW} !important; margin-bottom: 0.8rem !important;
}}
[data-testid="stStatusWidget"] *, [data-testid="stStatus"] * {{
    color: {TEXT_BODY} !important; font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.87rem !important;
}}
[data-testid="stStatusWidget"] p, [data-testid="stStatus"] p {{ color: {TEXT_BODY} !important; }}

/* ═══ SUGGESTION CHIPS ═══════════════════════════════════════════════════════ */
.sugg-label {{
    font-size: 0.62rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.18em; color: {TEXT_MUTED};
    margin-bottom: 0.6rem; margin-top: 1.2rem;
    display: flex; align-items: center; gap: 0.5rem;
}}
.sugg-label::before {{
    content: ''; width: 14px; height: 2px;
    background: linear-gradient(90deg, #7c3aed, #00f5a0);
    border-radius: 2px; display: inline-block;
}}
/* Style suggestion buttons to look like chips, not full gradient buttons */
div[data-testid="column"] > div[data-testid="stButton"] > button[kind="secondary"],
.sugg-btn-row div[data-testid="stButton"] > button {{
    background: {BG_PANEL} !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT_BODY} !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 0.38rem 0.7rem !important;
    border-radius: 8px !important;
    box-shadow: none !important;
    letter-spacing: 0 !important;
    transition: border-color 0.15s, color 0.15s, background 0.15s !important;
    white-space: normal !important;
    text-align: left !important;
    line-height: 1.3 !important;
    height: auto !important;
}}
div[data-testid="column"] > div[data-testid="stButton"] > button[kind="secondary"]:hover,
.sugg-btn-row div[data-testid="stButton"] > button:hover {{
    border-color: #7c3aed !important;
    color: {TEXT_H} !important;
    background: rgba(124,58,237,0.10) !important;
    transform: none !important;
    box-shadow: 0 0 0 1px rgba(124,58,237,0.3) !important;
}}
/* Sidebar suggestion buttons — same chip style */
section[data-testid="stSidebar"] div[data-testid="stButton"] > button {{
    background: {BG_CARD} !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT_BODY} !important;
    font-size: 0.74rem !important; font-weight: 500 !important;
    padding: 0.38rem 0.7rem !important; border-radius: 7px !important;
    box-shadow: none !important; letter-spacing: 0 !important;
    text-align: left !important; white-space: normal !important;
    line-height: 1.3 !important; height: auto !important;
    margin-bottom: 0.2rem !important;
}}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {{
    border-color: {BORDER_FOCUS} !important;
    color: {TEXT_H} !important;
    background: rgba(124,58,237,0.10) !important;
    transform: none !important; box-shadow: none !important;
}}

/* ═══ GUIDANCE PANEL ═════════════════════════════════════════════════════════ */
.guidance-panel {{
    background: {BG_PANEL}; border: 1px solid {BORDER};
    border-left: 3px solid #00d4ff; border-radius: 14px;
    padding: 1.2rem 1.5rem; margin-top: 1.2rem; animation: fadeUp 0.4s ease both;
}}
.guidance-title {{
    font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.18em; color: #00d4ff; margin-bottom: 0.8rem;
}}
.guidance-cols {{ display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 0.9rem; }}
.guidance-col-chip {{
    background: rgba(0,212,255,0.08); border: 1px solid rgba(0,212,255,0.2);
    border-radius: 6px; padding: 0.22rem 0.6rem; font-size: 0.75rem; font-weight: 600;
    color: #00d4ff; font-family: 'Fira Code', monospace;
}}
.guidance-examples {{ font-size: 0.8rem; color: {TEXT_MUTED}; line-height: 1.7; }}
.guidance-examples span {{ display: block; padding: 0.15rem 0; }}
.guidance-examples span::before {{ content: '→ '; color: #7c3aed; }}

/* ═══ FRIENDLY ERROR ══════════════════════════════════════════════════════════ */
.friendly-err {{
    background: {BG_PANEL}; border: 1px solid {BORDER};
    border-left: 3px solid #f87171; border-radius: 14px;
    padding: 1.3rem 1.5rem; animation: fadeUp 0.3s ease;
}}
.friendly-err-title {{ font-size: 0.88rem; font-weight: 700; color: #f87171; margin-bottom: 0.5rem; }}
.friendly-err-body {{ font-size: 0.85rem; color: {TEXT_BODY}; line-height: 1.65; margin-bottom: 0.8rem; }}
.friendly-err-cols {{ display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.6rem; }}
.friendly-err-col {{
    background: rgba(248,113,113,0.08); border: 1px solid rgba(248,113,113,0.2);
    border-radius: 5px; padding: 0.18rem 0.5rem; font-size: 0.72rem; font-weight: 600;
    color: #fca5a5; font-family: 'Fira Code', monospace;
}}

/* ═══ RESULTS SUMMARY TABLE ════════════════════════════════════════════════ */
.res-tbl-wrap {{
    background: {BG_PANEL}; border: 1px solid {BORDER};
    border-radius: 14px; padding: 1.2rem 1.5rem; margin-top: 1.4rem;
    box-shadow: {SHADOW};
}}
.res-tbl-title {{
    font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.16em; color: {BORDER_FOCUS}; margin-bottom: 0.9rem;
}}
.res-tbl {{
    width: 100%; border-collapse: collapse; font-family: 'Space Grotesk', sans-serif;
}}
.res-tbl-hdr {{
    font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.12em; color: {TEXT_MUTED};
    padding: 0.45rem 0.9rem; text-align: right;
    border-bottom: 1px solid {BORDER};
}}
.res-tbl-hdr-first {{ text-align: left; }}
.res-tbl-metric {{
    font-size: 0.82rem; font-weight: 600; color: {TEXT_H};
    padding: 0.55rem 0.9rem; text-align: left;
    border-bottom: 1px solid {BORDER_SUBTLE};
    white-space: nowrap;
}}
.res-tbl-cell {{
    font-size: 0.82rem; color: {TEXT_BODY}; font-variant-numeric: tabular-nums;
    padding: 0.55rem 0.9rem; text-align: right;
    border-bottom: 1px solid {BORDER_SUBTLE};
}}
.res-tbl tr:last-child .res-tbl-metric,
.res-tbl tr:last-child .res-tbl-cell {{ border-bottom: none; }}
.res-tbl tr:hover .res-tbl-metric,
.res-tbl tr:hover .res-tbl-cell {{
    background: {BG_INPUT} !important; color: {TEXT_H};
}}

/* ═══ MOBILE ═════════════════════════════════════════════════════════════════ */

/* Base: prevent ALL horizontal overflow site-wide */
html, body, .stApp, [data-testid="stAppViewContainer"], .block-container {{
    max-width: 100vw !important;
    overflow-x: hidden !important;
    box-sizing: border-box !important;
}}

@media (max-width: 900px) {{
    /* Tighten padding on tablets */
    .block-container {{ padding: 0 1.2rem 2rem 1.2rem !important; }}
    .hero {{ padding: 2rem 1.5rem 1.8rem 1.5rem; margin: 0 -1.2rem 1.5rem -1.2rem; }}
    /* Charts fill full width */
    .chart-panel {{ padding: 1rem !important; }}
    /* KPI grid: 2 cols on tablet */
    .kpi-value {{ font-size: 1.7rem !important; }}
    /* Tab labels compress */
    [data-testid="stTabs"] [role="tab"] {{
        padding: 0.45rem 0.9rem !important; font-size: 0.82rem !important;
    }}
}}

@media (max-width: 640px) {{
    /* Mobile phone layout */
    .block-container {{ padding: 0 0.8rem 2rem 0.8rem !important; }}
    .hero {{
        padding: 1.6rem 1rem 1.4rem 1rem !important;
        margin: 0 -0.8rem 1.2rem -0.8rem !important;
    }}
    .hero h1 {{ font-size: 1.7rem !important; line-height: 1.15 !important; }}
    .hero-sub {{ font-size: 0.85rem !important; max-width: 100% !important; }}
    /* Sidebar: full width overlay on mobile (already collapsed by default) */
    section[data-testid="stSidebar"] {{
        min-width: 0 !important;
    }}
    /* KPI cards: shrink for mobile */
    .kpi-card {{ padding: 1rem 1.1rem !important; border-radius: 12px !important; }}
    .kpi-value {{ font-size: 1.5rem !important; }}
    .kpi-icon {{ font-size: 1.2rem !important; margin-bottom: 0.5rem !important; }}
    /* Suggestion chips: stack 2×2 */
    div[data-testid="column"] {{ min-width: 0 !important; }}
    /* Charts: no min-width */
    .chart-panel {{ padding: 0.8rem !important; border-radius: 10px !important; }}
    /* Tab bar: scrollable on very small screens */
    [data-testid="stTabs"] [role="tablist"] {{
        overflow-x: auto !important;
        flex-wrap: nowrap !important;
        -webkit-overflow-scrolling: touch !important;
        padding: 3px !important;
    }}
    [data-testid="stTabs"] [role="tab"] {{
        padding: 0.4rem 0.65rem !important;
        font-size: 0.75rem !important;
        white-space: nowrap !important;
        flex-shrink: 0 !important;
    }}
    /* Input textarea full width */
    .stTextArea textarea {{ font-size: 0.95rem !important; }}
    /* Guidance panel: tighten on mobile */
    .guidance-panel {{ padding: 0.9rem 1rem !important; }}
    .guidance-col-chip {{ font-size: 0.68rem !important; }}
    /* ai-panel on mobile */
    .ai-panel {{ padding: 1.2rem 1.2rem !important; }}
    /* section headers */
    .section-hd {{ margin: 1.5rem 0 0.7rem 0 !important; }}
    /* workspace-panel */
    .workspace-panel {{ padding: 0.9rem 1rem !important; }}
}}

@media (max-width: 400px) {{
    .hero h1 {{ font-size: 1.4rem !important; }}
    .kpi-value {{ font-size: 1.3rem !important; }}
    .hero-eyebrow {{ font-size: 0.55rem !important; letter-spacing: 0.14em !important; }}
}}
</style>
""", unsafe_allow_html=True)

# Force sidebar to be expanded and hide resize handle / tooltip and hide resize handle / tooltip
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
            # Wipe all previous results — they belong to the old dataset
            st.session_state.result_df           = None
            st.session_state.result_sql          = None
            st.session_state.result_err          = None
            st.session_state.result_err_type     = "generic"
            st.session_state.result_insight      = None
            st.session_state.result_empty_reason = None
            # Clear history — old queries don't apply to the new schema
            st.session_state.query_history       = []
            _save_history([])
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
            # Clear dataset references
            st.session_state.uploaded_db   = None
            st.session_state.active_schema = None
            st.session_state.active_table  = "cars"
            st.session_state.active_label  = None
            # Wipe all results — they belong to the removed dataset
            st.session_state.result_df           = None
            st.session_state.result_sql          = None
            st.session_state.result_err          = None
            st.session_state.result_err_type     = "generic"
            st.session_state.result_insight      = None
            st.session_state.result_empty_reason = None
            # Clear history — queries from the old dataset are meaningless now
            st.session_state.query_history       = []
            _save_history([])
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
    if st.button("🗑️  Clear Conversation", width='stretch'):
        st.session_state.query_history  = []
        st.session_state.result_df      = None
        st.session_state.result_sql     = None
        st.session_state.result_err     = None
        st.session_state.result_insight = None
        _save_history([])   # wipe the file too
        st.rerun()

    # Examples — clickable buttons that populate and auto-run the query
    _ex_title = "💡 Quick Questions" if _is_uploaded else "💡 Quick Questions (BMW Demo)"
    st.markdown(f'<div class="sb-sec-title">{_ex_title}</div>', unsafe_allow_html=True)
    for _sb_i, _sb_q in enumerate(_suggestions[:6]):
        if st.button(f"→ {_sb_q}", key=f"sb_sugg_{_sb_i}", width='stretch'):
            _run_question(_sb_q)

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
# MAIN — TABBED DASHBOARD LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

# ── Ambient background upgrade (injected once after CSS loads) ────────────────
st.markdown(f"""
<style>
.stApp {{
    background:
        radial-gradient(circle at 15% 20%, rgba(124,58,237,0.18), transparent 32%),
        radial-gradient(circle at 82% 72%, rgba(0,245,160,0.10), transparent 35%),
        radial-gradient(circle at 55% 50%, rgba(0,212,255,0.06), transparent 40%),
        {BG_PAGE} !important;
}}
/* ── Tab bar ── */
[data-testid="stTabs"] [role="tablist"] {{
    background: {BG_PANEL};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 4px;
    gap: 4px;
    margin-bottom: 1.8rem;
}}
[data-testid="stTabs"] [role="tab"] {{
    background: transparent !important;
    border: none !important;
    border-radius: 10px !important;
    color: {TEXT_MUTED} !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 0.55rem 1.3rem !important;
    transition: all 0.2s ease !important;
}}
[data-testid="stTabs"] [role="tab"]:hover {{
    color: {TEXT_H} !important;
    background: rgba(124,58,237,0.12) !important;
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
    background: linear-gradient(135deg,#7c3aed,#5b21b6) !important;
    color: #ffffff !important;
    box-shadow: 0 2px 12px rgba(124,58,237,0.45) !important;
}}
/* Remove the default Streamlit tab underline */
[data-testid="stTabs"] [role="tab"][aria-selected="true"]::after,
[data-testid="stTabs"] [role="tabpanel"] {{ border: none !important; }}
/* ── Chart panel container ── */
.chart-panel {{
    background: linear-gradient(135deg,#161620,#111118);
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 1.4rem 1.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(0,0,0,0.5), 0 1px 0 rgba(255,255,255,0.03);
}}
.chart-panel::before {{
    content: '';
    position: absolute; top: -40px; right: -40px;
    width: 160px; height: 160px;
    background: radial-gradient(circle, rgba(124,58,237,0.10) 0%, transparent 70%);
    border-radius: 50%; pointer-events: none;
}}
.chart-panel-label {{
    font-size: 0.62rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.18em; color: {TEXT_MUTED};
    margin-bottom: 0.9rem; display: flex; align-items: center; gap: 0.5rem;
}}
.chart-panel-label::before {{
    content: '';
    width: 12px; height: 2px;
    background: linear-gradient(90deg,#7c3aed,#00f5a0);
    border-radius: 2px; display: inline-block;
}}
/* ── Analysis workspace ── */
.workspace-panel {{
    background: {BG_PANEL};
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    height: 100%;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}}
.workspace-label {{
    font-size: 0.6rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.18em; color: {TEXT_MUTED}; margin-bottom: 0.8rem;
    display: flex; align-items: center; gap: 0.4rem;
}}
/* ── AI Insights tab card ── */
.ai-panel {{
    background: linear-gradient(135deg, {BG_CARD} 0%, {BG_CARD2} 100%);
    border: 1px solid {BORDER};
    border-top: 2px solid #7c3aed;
    border-radius: 18px;
    padding: 2rem 2.2rem;
    box-shadow: 0 4px 32px rgba(0,0,0,0.5), 0 0 60px rgba(124,58,237,0.08);
    position: relative; overflow: hidden;
    animation: fadeUp 0.5s ease both;
}}
.ai-panel::after {{
    content: '';
    position: absolute; bottom: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(0,245,160,0.08) 0%, transparent 70%);
    border-radius: 50%; pointer-events: none;
}}
.ai-analyst-header {{
    display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;
}}
.ai-analyst-avatar {{
    width: 44px; height: 44px; border-radius: 50%;
    background: linear-gradient(135deg,#7c3aed,#5b21b6);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem; flex-shrink: 0;
    box-shadow: 0 4px 16px rgba(124,58,237,0.45);
}}
.ai-analyst-name {{
    font-size: 0.95rem; font-weight: 700; color: {TEXT_H};
    letter-spacing: -0.01em;
}}
.ai-analyst-role {{
    font-size: 0.68rem; color: {TEXT_MUTED};
    text-transform: uppercase; letter-spacing: 0.14em; margin-top: 0.1rem;
}}
.ai-analyst-text {{
    font-size: 1rem; line-height: 1.8; color: {TEXT_BODY}; font-weight: 400;
}}
.ai-empty-state {{
    text-align: center; padding: 3rem 2rem;
}}
.ai-empty-icon {{ font-size: 3rem; margin-bottom: 1rem; opacity: 0.4; }}
.ai-empty-text {{
    font-size: 0.9rem; color: {TEXT_MUTED};
    line-height: 1.6; max-width: 360px; margin: 0 auto;
}}
/* ── History tab ── */
.history-empty {{
    text-align: center; padding: 3rem;
    color: {TEXT_MUTED}; font-size: 0.9rem;
}}
</style>
""", unsafe_allow_html=True)

# ── Animated top bar ──────────────────────────────────────────────────────────
st.markdown('<div class="top-bar"></div>', unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
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

# ── Handle pending question from suggestion buttons ──────────────────────────
# Pattern: set session_state["question_input"] BEFORE the widget renders,
# then pop pending_question so it only fires once.
_pending = st.session_state.pop("pending_question", None)
_auto_run = False
if _pending:
    st.session_state["question_input"] = _pending
    _auto_run = True

question = st.text_area(
    label="question",
    placeholder="e.g. What are the top 10 most expensive cars?",
    height=72,
    label_visibility="collapsed",
    key="question_input",
)

b1, _ = st.columns([1.1, 7])
with b1:
    run_btn = st.button("▶  Run Analysis", width='stretch')

# ── Suggestion chips — ALWAYS visible so users can explore ───────────────────
st.markdown('<div class="sugg-label">Try one of these</div>', unsafe_allow_html=True)
_sugg_cols = st.columns(4)
for _si, (_sc, _sq) in enumerate(zip(_sugg_cols, _suggestions[:4])):
    with _sc:
        if st.button(_sq, key=f"sugg_{_si}", width='stretch'):
            _run_question(_sq)

# ── Guidance panel — always visible until a query runs ───────────────────────
if not st.session_state.get("result_sql"):
    _render_guidance_panel()

# ══════════════════════════════════════════════════════════════════════════════
# QUERY EXECUTION
# ══════════════════════════════════════════════════════════════════════════════
if run_btn or _auto_run:
    if not question.strip():
        st.markdown('<div class="warn-box">⚠️ Please enter a question first.</div>',
                    unsafe_allow_html=True)
    else:
        st.session_state.result_df           = None
        st.session_state.result_sql          = None
        st.session_state.result_err          = None
        st.session_state.result_insight      = None
        st.session_state.result_empty_reason = None

        with st.status("🤖  Running AI analysis…", expanded=True) as _status:
            st.write("⚙️  Generating SQL from your question…")
            try:
                sql = generate_sql(
                    question,
                    schema=get_active_schema(),
                    table_name=get_active_table(),
                )
                st.session_state.result_sql = sql
                st.write("✅  SQL generated successfully.")
            except Exception as exc:
                _exc_str = str(exc)
                # Classify the error for a friendly message
                if "no such column" in _exc_str.lower():
                    st.session_state.result_err = "__column_error__"
                elif "rephrase" in _exc_str.lower() or "invalid" in _exc_str.lower():
                    st.session_state.result_err = (
                        "I couldn't generate a valid query for that question. "
                        "Try asking something like: average price by fuel type, "
                        "top 10 cheapest cars, or count cars by transmission."
                    )
                else:
                    st.session_state.result_err = (
                        "I couldn't understand that question well enough to query the data. "
                        "Try asking about: price, fuel type, mileage, year, or transmission."
                    )
                st.write(f"❌  {_exc_str}")
                _status.update(label="❌  Analysis failed", state="error", expanded=True)

        if st.session_state.result_sql and not st.session_state.result_err:
            with st.status("⚡  Querying database…", expanded=True) as _status:
                st.write("🗄️  Running query against the database…")
                try:
                    st.session_state.result_df = execute_query_on(st.session_state.result_sql)
                    _rows = len(st.session_state.result_df)
                    st.write(f"✅  Query returned {_rows:,} row(s).")
                    _status.update(label=f"✅  Query complete — {_rows:,} row(s) returned", state="complete", expanded=False)
                except Exception as exc:
                    _exc_str = str(exc)
                    if "no such column" in _exc_str.lower() or "no such table" in _exc_str.lower():
                        st.session_state.result_err = "__column_error__"
                    elif "syntax error" in _exc_str.lower():
                        st.session_state.result_err = (
                            "The query had a syntax issue. "
                            "Try rephrasing — for example: 'average price by fuel type' "
                            "or 'show top 10 cars by mileage'."
                        )
                    else:
                        st.session_state.result_err = (
                            "Something went wrong running that query. "
                            "Try rephrasing or using a simpler question."
                        )
                    st.write(f"❌  {_exc_str}")
                    _status.update(label="❌  Query failed", state="error", expanded=True)

        if (st.session_state.result_df is not None
                and not st.session_state.result_df.empty
                and not st.session_state.result_err):
            with st.status("💡  Generating AI insight…", expanded=True) as _status:
                st.write("📊  Building visualization…")
                st.write("✅  Visualization ready.")
                st.write("🧠  Analysing results with AI…")
                try:
                    _df   = st.session_state.result_df
                    _rows_str = _df.head(20).to_string(index=False)
                    _cols = ", ".join(_df.columns.tolist())
                    _insight_prompt = (
                        f"A user asked: \"{question}\"\n\n"
                        f"The query returned {len(_df)} rows with columns: {_cols}.\n"
                        f"Here is a sample of the data:\n{_rows_str}\n\n"
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
                    st.write("✅  AI insight generated.")
                    _status.update(label="✅  Analysis complete", state="complete", expanded=False)
                except Exception:
                    st.session_state.result_insight = None
                    st.write("⚠️  AI insight unavailable — skipped.")
                    _status.update(label="⚠️  Analysis complete (no AI insight)", state="complete", expanded=False)

        elif (st.session_state.result_df is not None
                and st.session_state.result_df.empty
                and not st.session_state.result_err):
            with st.status("🔍  Explaining empty result…", expanded=True) as _status:
                st.write("🧠  Asking AI why no results were found…")
                try:
                    _empty_prompt = (
                        f"A user asked: \"{question}\"\n\n"
                        f"The query ran successfully but returned zero rows.\n\n"
                        "In 2-3 plain English sentences:\n"
                        "1. Explain WHY no results were found.\n"
                        "2. Suggest a practical alternative they could try instead.\n"
                        "Do NOT mention SQL, tables, or technical terms. Plain text only."
                    )
                    _groq_client = _Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
                    _resp = _groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        temperature=0.3,
                        max_tokens=160,
                        messages=[
                            {"role": "system", "content": "You are a helpful business intelligence assistant."},
                            {"role": "user",   "content": _empty_prompt},
                        ],
                    )
                    st.session_state.result_empty_reason = _resp.choices[0].message.content.strip()
                    st.write("✅  Explanation ready.")
                    _status.update(label="✅  Done", state="complete", expanded=False)
                except Exception:
                    st.session_state.result_empty_reason = None
                    st.write("⚠️  Explanation unavailable.")
                    _status.update(label="⚠️  Done", state="complete", expanded=False)

        if st.session_state.result_sql and not st.session_state.result_err:
            st.session_state.query_history.insert(0, {
                "question": question,
                "sql":      st.session_state.result_sql,
                "insight":  st.session_state.result_insight,
                "df":       st.session_state.result_df.copy() if st.session_state.result_df is not None else None,
            })
            st.session_state.query_history = st.session_state.query_history[:10]
            _save_history(st.session_state.query_history)

# ══════════════════════════════════════════════════════════════════════════════
# TABBED WORKSPACE
# ══════════════════════════════════════════════════════════════════════════════

tab_dash, tab_analysis, tab_ai, tab_history = st.tabs([
    "📊  Dashboard",
    "📈  Analysis",
    "🧠  AI Insights",
    "🕐  History",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — DASHBOARD  (Dataset Intelligence overview)
# ─────────────────────────────────────────────────────────────────────────────
with tab_dash:

    # ── Results KPI cards (shown after a query runs) ──────────────────────────
    df_res       = st.session_state.result_df
    numeric_cols = df_res.select_dtypes(include="number").columns.tolist() if df_res is not None else []

    if st.session_state.result_err:
        _err = st.session_state.result_err
        if _err == "__column_error__":
            _render_friendly_error(
                "This question refers to fields not present in the current dataset.",
                show_columns=True,
            )
        else:
            _render_friendly_error(_err, show_columns=True)

    elif df_res is not None and not df_res.empty:
        n = len(df_res)
        if numeric_cols:
            _col = numeric_cols[0]
            _vals = df_res[_col].dropna()
            kpi_data = [
                ("🔢", "Rows Returned", f"{n:,}",              "from this query"),
                ("📈", f"Max {_col}",   f"{_vals.max():,.1f}", f"highest value"),
                ("📊", f"Avg {_col}",   f"{_vals.mean():,.1f}","mean value"),
                ("📉", f"Min {_col}",   f"{_vals.min():,.1f}", f"lowest value"),
            ]
        else:
            _ds_total = _meta["total"] if _meta else "—"
            kpi_data = [
                ("🔢", "Rows Returned", f"{n:,}",                  "from this query"),
                ("📋", "Columns",       f"{len(df_res.columns)}",   "in result"),
                ("📊", "Dataset Rows",  f"{_ds_total:,}" if isinstance(_ds_total, int) else str(_ds_total), "total"),
                ("✅", "Status",        "Success",                  "query executed"),
            ]

        st.markdown("""<div class="section-hd">
            <span class="section-hd-text">📌 Query Results Overview</span>
            <div class="section-hd-line"></div></div>""", unsafe_allow_html=True)

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
                </div>""", unsafe_allow_html=True)

        # ── Summary stats table (numeric columns only) ────────────────────────
        if numeric_cols:
            _tbl_rows_html = ""
            for _i, _nc in enumerate(numeric_cols):
                _v = df_res[_nc].dropna()
                _row_bg = f"background:{BG_INPUT};" if _i % 2 == 0 else ""
                _tbl_rows_html += f"""
                <tr style="{_row_bg}">
                    <td class="res-tbl-metric">{_nc.replace('_',' ').title()}</td>
                    <td class="res-tbl-cell">{len(_v):,}</td>
                    <td class="res-tbl-cell">{_v.min():,.2f}</td>
                    <td class="res-tbl-cell">{_v.max():,.2f}</td>
                    <td class="res-tbl-cell">{_v.mean():,.2f}</td>
                    <td class="res-tbl-cell">{_v.median():,.2f}</td>
                    <td class="res-tbl-cell">{_v.std():,.2f}</td>
                </tr>"""
            st.markdown(f"""
            <div class="res-tbl-wrap">
                <div class="res-tbl-title">📋 &nbsp;Column Statistics</div>
                <div style="overflow-x:auto;">
                <table class="res-tbl">
                    <thead>
                        <tr>
                            <th class="res-tbl-hdr res-tbl-hdr-first">Column</th>
                            <th class="res-tbl-hdr">Count</th>
                            <th class="res-tbl-hdr">Min</th>
                            <th class="res-tbl-hdr">Max</th>
                            <th class="res-tbl-hdr">Average</th>
                            <th class="res-tbl-hdr">Median</th>
                            <th class="res-tbl-hdr">Std Dev</th>
                        </tr>
                    </thead>
                    <tbody>{_tbl_rows_html}</tbody>
                </table>
                </div>
            </div>""", unsafe_allow_html=True)

    elif df_res is not None and df_res.empty:
        _reason = st.session_state.get("result_empty_reason")
        if _reason:
            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-avatar">
                    <div class="insight-avatar-icon">🔍</div>
                    <div class="insight-avatar-label">No Results — AI Explanation</div>
                </div>
                <div class="insight-text">{_reason}</div>
            </div>""", unsafe_allow_html=True)
        else:
            _render_friendly_error(
                "No data matched this question. Try adjusting your filters or asking a different question.",
                show_columns=True,
            )
        if st.session_state.get("result_sql"):
            with st.expander("🔍  AI Generated SQL", expanded=False):
                st.code(st.session_state.result_sql, language="sql")

    # ── Dataset Intelligence KPI cards ────────────────────────────────────────
    if _meta:
        st.markdown("""<div class="section-hd">
            <span class="section-hd-text">🧠 Dataset Intelligence</span>
            <div class="section-hd-line"></div></div>""", unsafe_allow_html=True)

        _di_kpis = []
        _di_kpis.append(("📋", "Total Rows",    f"{_meta['total']:,}",  "records in dataset"))
        _di_kpis.append(("📐", "Total Columns", f"{_meta['n_cols']}",    "fields detected"))
        for _ni, (_nc, _nv) in enumerate(_meta.get("num_stats", {}).items()):
            _icon = ["💹","📊","🔢"][_ni % 3]
            _di_kpis.append((_icon, f"Avg {_nc.replace('_',' ').title()}", f"{_nv:,.2f}", "average value"))
        for _ci, (_cc, _cv) in enumerate(_meta.get("cat_stats", {}).items()):
            _icon = ["🏆","🔖"][_ci % 2]
            _di_kpis.append((_icon, f"Top {_cc.replace('_',' ').title()}", str(_cv), "most frequent"))

        _n_kpi_cols = min(len(_di_kpis), 4)
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
                </div>""", unsafe_allow_html=True)

        # Overflow KPI cards — row 2
        if len(_di_kpis) > 4:
            _di_cols2 = st.columns(min(len(_di_kpis) - 4, 4))
            for _di_i2, (_di_col2, (_di_icon2, _di_label2, _di_val2, _di_sub2)) in enumerate(
                    zip(_di_cols2, _di_kpis[4:8])):
                _di_grad2, _di_glow2 = KPI_COLORS[(_di_i2 + 4) % len(KPI_COLORS)]
                with _di_col2:
                    st.markdown(f"""
                    <div class="kpi-card" style="animation-delay:{(_di_i2+4)*0.07}s">
                        <div class="kpi-orb" style="background:radial-gradient(circle,{_di_glow2} 0%,transparent 70%);"></div>
                        <span class="kpi-icon">{_di_icon2}</span>
                        <div class="kpi-label">{_di_label2}</div>
                        <div class="kpi-value">{_di_val2}</div>
                        <div class="kpi-sub">{_di_sub2}</div>
                        <div class="kpi-bar" style="background:linear-gradient({_di_grad2});"></div>
                    </div>""", unsafe_allow_html=True)

        # AI dataset summary
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
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="ai-empty-state">
            <div class="ai-empty-icon">📂</div>
            <div class="ai-empty-text">
                Upload a CSV dataset using the sidebar to see dataset intelligence cards here,
                or use the demo BMW dataset — just run any query to get started.
            </div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — ANALYSIS  (chart + table side by side)
# ─────────────────────────────────────────────────────────────────────────────
with tab_analysis:

    df_a         = st.session_state.result_df
    numeric_cols = df_a.select_dtypes(include="number").columns.tolist() if df_a is not None else []
    text_cols    = df_a.select_dtypes(exclude="number").columns.tolist() if df_a is not None else []

    if st.session_state.result_err:
        _err_a = st.session_state.result_err
        if _err_a == "__column_error__":
            _render_friendly_error(
                "This question refers to fields not present in the current dataset.",
                show_columns=True,
            )
        else:
            _render_friendly_error(_err_a, show_columns=True)

    elif df_a is None:
        st.markdown(f"""
        <div class="ai-empty-state">
            <div class="ai-empty-icon">📈</div>
            <div class="ai-empty-text">
                Run a query to see your charts and data table here.
                Results will appear in a 70/30 split — chart on the left, data preview on the right.
            </div>
        </div>""", unsafe_allow_html=True)

    elif df_a.empty:
        _render_friendly_error(
            "No data matched this question. Try adjusting your filters or asking a different question.",
            show_columns=True,
        )

    else:
        n = len(df_a)

        # ── Shared Plotly layout ──────────────────────────────────────────────
        TL = dict(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PLOT_BG,
            font=dict(family="Space Grotesk", color=FONT_CLR, size=11),
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR, tickfont=dict(size=10)),
            yaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR, tickfont=dict(size=10)),
            coloraxis_showscale=False,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=FONT_CLR, size=10)),
        )

        if numeric_cols:
            _cfg       = _pick_chart_config(question, df_a)
            _icon_map  = {"line":"📈","bar":"📊","bar_h":"📊","histogram":"📉","pie":"🥧","scatter":"🔵"}
            _chart_icon = _icon_map.get(_cfg["primary"], "📊")

            fig_main = _build_primary_fig(_cfg, df_a, TL)

            # 70 / 30 split: chart panel left, data table right
            chart_col, table_col = st.columns([7, 3], gap="large")

            with chart_col:
                st.markdown(f"""
                <div class="chart-panel">
                    <div class="chart-panel-label">{_chart_icon} {_cfg['chart_label']}</div>
                """, unsafe_allow_html=True)
                st.plotly_chart(fig_main, width='stretch')
                st.markdown("</div>", unsafe_allow_html=True)

                # Secondary chart below primary if available
                fig2 = _build_secondary_fig(_cfg, df_a, TL)
                if fig2 is not None:
                    st.markdown("""
                    <div class="chart-panel" style="margin-top:1rem;">
                        <div class="chart-panel-label">📊 Companion View</div>
                    """, unsafe_allow_html=True)
                    st.plotly_chart(fig2, width='stretch')
                    st.markdown("</div>", unsafe_allow_html=True)

            with table_col:
                st.markdown('<div class="workspace-panel">', unsafe_allow_html=True)
                st.markdown('<div class="workspace-label">📋 Data Preview</div>',
                            unsafe_allow_html=True)
                st.dataframe(df_a, width='stretch', hide_index=True)
                st.caption(f"✓ {n:,} row(s) · {len(df_a.columns)} col(s)")
                st.markdown("</div>", unsafe_allow_html=True)

                # SQL expander in the table column
                if st.session_state.get("result_sql"):
                    with st.expander("🔍  Generated SQL", expanded=False):
                        st.code(st.session_state.result_sql, language="sql")

                # Download button
                st.download_button(
                    label="⬇️  Download CSV",
                    data=df_a.to_csv(index=False).encode("utf-8"),
                    file_name="query_results.csv",
                    mime="text/csv",
                )

        else:
            # No numeric cols — just show the table full width
            st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
            st.markdown('<div class="chart-panel-label">📋 Query Results</div>',
                        unsafe_allow_html=True)
            st.dataframe(df_a, width='stretch', hide_index=True)
            st.caption(f"✓ {n:,} row(s) · {len(df_a.columns)} col(s)")
            st.markdown("</div>", unsafe_allow_html=True)
            if st.session_state.get("result_sql"):
                with st.expander("🔍  Generated SQL", expanded=False):
                    st.code(st.session_state.result_sql, language="sql")
            st.download_button(
                label="⬇️  Download CSV",
                data=df_a.to_csv(index=False).encode("utf-8"),
                file_name="query_results.csv",
                mime="text/csv",
            )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — AI INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
with tab_ai:

    _insight = st.session_state.get("result_insight")
    _q_text  = question if question else (
        st.session_state.query_history[0]["question"]
        if st.session_state.query_history else ""
    )

    if _insight:
        st.markdown(f"""
        <div class="ai-panel">
            <div class="ai-analyst-header">
                <div class="ai-analyst-avatar">🤖</div>
                <div>
                    <div class="ai-analyst-name">AI Data Analyst</div>
                    <div class="ai-analyst-role">Powered by LLaMA 3.3 70B · Groq</div>
                </div>
            </div>
            <div style="font-size:0.72rem;color:{TEXT_MUTED};margin-bottom:0.6rem;
                        text-transform:uppercase;letter-spacing:0.14em;">
                Analysing: &ldquo;{_q_text}&rdquo;
            </div>
            <div class="ai-analyst-text">{_insight}</div>
        </div>""", unsafe_allow_html=True)

        # Dataset summary below if available
        if _ds_summary:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="di-section">
                <div class="di-header">
                    <div class="di-header-icon">🧠</div>
                    <div>
                        <div class="di-header-text">Dataset Context</div>
                        <div class="di-header-sub">{_active_label}</div>
                    </div>
                </div>
                <div class="di-summary-label">📊 &nbsp; Background</div>
                <div class="di-summary">{_ds_summary}</div>
            </div>""", unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="ai-panel">
            <div class="ai-empty-state">
                <div class="ai-empty-icon">🧠</div>
                <div class="ai-empty-text">
                    Run a query above and the AI analyst will appear here with a plain-English
                    explanation of your results — trends, key numbers, and what they mean
                    for your business.
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — HISTORY
# ─────────────────────────────────────────────────────────────────────────────
with tab_history:

    _history = st.session_state.get("query_history", [])

    if not _history:
        st.markdown("""
        <div class="history-empty">
            <div class="ai-empty-icon">🕐</div>
            <div class="ai-empty-text">No queries yet. Run your first question above.</div>
        </div>""", unsafe_allow_html=True)
    else:
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
                    {"<div class='hist-insight'>🤖 " + _insight + "</div>" if _insight else ""}
                    <div class="hist-num">📊 {_row_count:,} rows returned</div>
                </div>""", unsafe_allow_html=True)

                if _h_df is not None and not _h_df.empty and _h_num:
                    _hy = _h_num[0]
                    _hx = _h_txt[0] if _h_txt else _h_df.columns[0]
                    _hn = len(_h_df)
                    _TL_h = dict(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PLOT_BG,
                        font=dict(family="Space Grotesk", color=FONT_CLR, size=10),
                        margin=dict(l=5, r=5, t=10, b=5), height=200,
                        xaxis=dict(gridcolor=GRID_CLR, tickfont=dict(size=9)),
                        yaxis=dict(gridcolor=GRID_CLR, tickfont=dict(size=9)),
                        coloraxis_showscale=False,
                    )
                    if _hn <= 30:
                        _fig_h = px.bar(
                            _h_df.sort_values(_hy, ascending=False).head(10),
                            x=_hx, y=_hy, color=_hy,
                            color_continuous_scale=CHART_SEQ,
                            labels={_hx: _hx.replace("_"," ").title(), _hy: _hy.replace("_"," ").title()},
                        )
                    else:
                        _fig_h = px.scatter(
                            _h_df, x=_hx, y=_hy, color=_hy,
                            color_continuous_scale=CHART_SEQ, opacity=0.7,
                            labels={_hx: _hx.replace("_"," ").title(), _hy: _hy.replace("_"," ").title()},
                        )
                    _fig_h.update_layout(**_TL_h)
                    _fig_h.update_traces(marker_line_width=0)
                    st.plotly_chart(_fig_h, width='stretch', key=f"hist_chart_{_i}")

            if _i < len(_history) - 1:
                st.divider()