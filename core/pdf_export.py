"""Export utilities: PNG chart, CSV dataframe, and full-session PDF."""
import io
import re
import csv
import datetime
from typing import Any

import pandas as pd

try:
    from fpdf import FPDF, XPos, YPos
    _FPDF_AVAILABLE = True
except ImportError:
    FPDF = XPos = YPos = None
    _FPDF_AVAILABLE = False


# ── PNG / CSV exports ─────────────────────────────────────────────────────────

def chart_to_png(fig) -> bytes:
    return fig.to_image(format="png", width=1200, height=600, scale=2)


def dataframe_to_csv(rows: list[dict]) -> bytes:
    if not rows:
        return b""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


# ── Unicode sanitizer ─────────────────────────────────────────────────────────

_UNICODE_MAP = str.maketrans({
    "—": "--", "–": "-",
    "‘": "'",  "’": "'",
    "“": '"',  "”": '"',
    "…": "...",
    "•": "-",
    "·": ".",
    "−": "-",
    "×": "x",
})

_FILLER_RE = re.compile(
    r"^\s*(?:"
    r"here(?:'s| is) (?:the |a )?(?:visualization|chart|graph|plot|data|table|sql|query|result|output)"
    r"|the sql used(?:\s+for\s+this)?(?:\s+(?:data|query))?\s*(?:is|:)"
    r"|i(?:'ll| will) answer \d+ questions?"
    r")\s*[:\.]?\s*$",
    re.IGNORECASE,
)


def _s(text: str) -> str:
    text = str(text).translate(_UNICODE_MAP)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _truncate(text: str, n: int) -> str:
    text = _s(str(text))
    return text if len(text) <= n else text[: n - 1] + "."


_JSON_ARTIFACT_RE = re.compile(
    r'^\s*[\[{].*|"(answer|chart_config|sql_used|data_preview|sub_results)"\s*:',
    re.DOTALL,
)


def _clean_answer(text: str, max_chars: int = 3000) -> str:
    """Strip filler lines, JSON artifacts, and collapse excess blank lines."""
    if not text:
        return ""
    # If text looks like raw JSON output (starts with { or [), bail out early
    stripped_start = text.lstrip()
    if stripped_start.startswith("{") or stripped_start.startswith("["):
        return "[Structured response — see chart/table above]"
    text = _s(text[:max_chars] + ("..." if len(text) > max_chars else ""))
    lines = text.split("\n")
    kept, blank_run = [], 0
    for line in lines:
        stripped = line.strip()
        # Drop filler lines
        if _FILLER_RE.match(stripped):
            continue
        # Drop lines that look like JSON field declarations
        if re.match(r'^\s*"[a-z_]+"\s*:', stripped):
            continue
        # Drop lines that are pure JSON syntax artifacts
        if stripped in ("{", "}", "[", "]", "},", "],"):
            continue
        if stripped == "":
            blank_run += 1
            if blank_run <= 1:
                kept.append("")
        else:
            blank_run = 0
            kept.append(line)
    return "\n".join(kept).strip()


# ── Full session PDF ──────────────────────────────────────────────────────────

def session_to_pdf(chat_history: list, charts: dict[str, Any] | None = None) -> bytes:
    if not _FPDF_AVAILABLE:
        raise ImportError("fpdf2 required: pip install fpdf2")

    pdf = FPDF(unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()

    _draw_header(pdf)

    q_num = 0
    for actual_idx, msg in enumerate(chat_history):
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            q_num += 1
            _draw_question(pdf, q_num, content)

        elif role == "assistant":
            analysis = msg.get("analysis")
            if analysis:
                _draw_analysis(pdf, analysis, actual_idx, charts)
            else:
                _draw_answer_body(pdf, content)
            _draw_divider(pdf)

    return bytes(pdf.output())


# ── Section renderers ─────────────────────────────────────────────────────────

def _draw_header(pdf: "FPDF") -> None:
    pdf.set_fill_color(15, 23, 42)
    pdf.rect(0, 0, 210, 34, style="F")
    pdf.set_font("Helvetica", "B", 19)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(15, 8)
    pdf.cell(0, 10, _s("Data Analysis Session Report"))
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(148, 172, 209)
    pdf.set_xy(15, 21)
    pdf.cell(0, 6, _s(datetime.datetime.now().strftime("Generated %B %d, %Y at %H:%M")))
    pdf.set_text_color(30, 30, 30)
    pdf.ln(24)


def _draw_question(pdf: "FPDF", num: int, text: str) -> None:
    page_before = pdf.page_no()
    x0, y0 = pdf.get_x(), pdf.get_y()
    pdf.set_fill_color(241, 245, 249)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 41, 59)
    pdf.multi_cell(
        0, 7,
        _s(f"   Q{num}   {_truncate(text, 150)}"),
        fill=True,
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )
    y1 = pdf.get_y()
    if pdf.page_no() == page_before:
        pdf.set_fill_color(71, 85, 105)
        pdf.rect(x0, y0, 2.5, y1 - y0, style="F")
    pdf.set_text_color(30, 30, 30)
    pdf.ln(4)


def _draw_sub_result_card(pdf: "FPDF", sub, actual_idx: int, charts) -> None:
    """Render one sub-result as a structured card: accent header → answer → table → chart."""
    # Accent header bar
    pdf.set_fill_color(30, 64, 175)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(255, 255, 255)
    label = _s(f"  [{sub.index}]  {_truncate(sub.question or '', 120)}")
    pdf.multi_cell(
        180, 7, label,
        fill=True,
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )
    pdf.set_text_color(30, 30, 30)
    pdf.ln(3)

    # Answer
    answer = _clean_answer(sub.answer or "")
    if answer:
        _draw_answer_body(pdf, answer)

    # Data table
    if sub.data_preview:
        _section_label(pdf, "Records")
        _write_table(pdf, sub.data_preview[:10])

    # SQL
    if getattr(sub, "sql_used", None):
        _draw_sql_block(pdf, sub.sql_used)

    # Chart
    chart_key = f"sub_{actual_idx}_{sub.index}"
    if charts and chart_key in charts:
        _section_label(pdf, "Visualization")
        _embed_chart(pdf, charts[chart_key])

    # Card bottom rule
    pdf.ln(2)
    pdf.set_draw_color(203, 213, 225)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(6)


def _draw_analysis(pdf: "FPDF", analysis, actual_idx: int, charts) -> None:
    if getattr(analysis, "sub_results", None):
        for sub in analysis.sub_results:
            _draw_sub_result_card(pdf, sub, actual_idx, charts)
        return

    # Single-question path
    answer = _clean_answer(getattr(analysis, "answer", "") or "")
    if answer:
        _draw_answer_body(pdf, answer)

    preview = getattr(analysis, "data_preview", None) or []
    if preview:
        _section_label(pdf, "Data")
        _write_table(pdf, preview[:10])

    sql = getattr(analysis, "sql_used", "") or ""
    if sql:
        _draw_sql_block(pdf, sql)

    if charts and str(actual_idx) in charts:
        _section_label(pdf, "Visualization")
        _embed_chart(pdf, charts[str(actual_idx)])

    ds = getattr(analysis, "datasets_used", []) or []
    if ds:
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(51, 65, 85)
        pdf.cell(0, 5, _s(f"Source: {', '.join(ds)}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(30, 30, 30)
        pdf.ln(1)


def _section_label(pdf: "FPDF", label: str) -> None:
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 5, _s(label.upper()), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(30, 30, 30)
    pdf.ln(1)


def _draw_divider(pdf: "FPDF") -> None:
    pdf.ln(3)
    pdf.set_draw_color(226, 232, 240)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(6)


def _draw_sql_block(pdf: "FPDF", sql: str) -> None:
    _section_label(pdf, "SQL")
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(203, 213, 225)
    pdf.set_font("Courier", "", 8)
    pdf.set_text_color(15, 23, 42)
    sql_clean = _s(sql.strip()[:2000])
    pdf.multi_cell(
        0, 4.5, sql_clean,
        border=1, fill=True,
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )
    pdf.set_text_color(30, 30, 30)
    pdf.ln(4)


# ── Answer text with inline bold / bullet parsing ─────────────────────────────

def _draw_answer_body(pdf: "FPDF", text: str) -> None:
    """Render markdown-lite: **bold**, - bullets, numbered items."""
    cleaned = _clean_answer(text)
    for line in cleaned.split("\n"):
        stripped = line.strip()
        if not stripped:
            pdf.ln(2)
            continue
        _write_inline(pdf, stripped)
    pdf.ln(2)


def _write_inline(pdf: "FPDF", line: str) -> None:
    """Write one line with optional bold segments and bullet prefix."""
    # Bullet
    is_bullet = bool(re.match(r"^[-*]\s+", line))
    if is_bullet:
        line = re.sub(r"^[-*]\s+", "", line).strip()
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(30, 41, 59)
        pdf.write(5, "  - ")
        pdf.set_text_color(30, 30, 30)

    # Numbered list stays as-is (handled below)
    segments = re.split(r"\*\*(.+?)\*\*", line)

    if len(segments) == 1:
        # No bold — use multi_cell for proper word wrap
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(30, 30, 30)
        if is_bullet:
            # Already used write() for bullet prefix; continue inline
            pdf.write(5, _s(line))
            pdf.ln(5)
        else:
            pdf.multi_cell(0, 5, _s(line), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    else:
        # Inline bold via write()
        pdf.set_text_color(30, 30, 30)
        for i, seg in enumerate(segments):
            if not seg:
                continue
            pdf.set_font("Helvetica", "B" if i % 2 == 1 else "", 10)
            pdf.write(5, _s(seg))
        pdf.ln(5)
        pdf.set_font("Helvetica", "", 10)


# ── Table ─────────────────────────────────────────────────────────────────────

def _write_table(pdf: "FPDF", rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    col_w = min(45, max(18, 180 // max(len(cols), 1)))
    visible_cols = cols[: min(len(cols), 180 // col_w)]

    # Header row
    pdf.set_fill_color(51, 65, 85)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(255, 255, 255)
    for c in visible_cols:
        pdf.cell(col_w, 7, _truncate(str(c), col_w // 2 + 4), fill=True, align="C")
    pdf.ln()

    # Data rows — alternating fill
    pdf.set_font("Helvetica", "", 8)
    for r_i, row in enumerate(rows):
        if r_i % 2 == 0:
            pdf.set_fill_color(248, 250, 252)
        else:
            pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(30, 30, 30)
        for c in visible_cols:
            val = str(row.get(c, ""))
            pdf.cell(col_w, 6, _truncate(val, col_w // 2 + 4), fill=True)
        pdf.ln()

    # Bottom rule
    pdf.set_draw_color(203, 213, 225)
    x0 = pdf.get_x()
    pdf.line(x0, pdf.get_y(), x0 + col_w * len(visible_cols), pdf.get_y())
    pdf.ln(5)


# ── Chart embed ───────────────────────────────────────────────────────────────

def _embed_chart(pdf: "FPDF", fig) -> None:
    """Fit chart to remaining page height (min 55mm, max 80mm) to avoid blank pages."""
    try:
        png = chart_to_png(fig)
        buf = io.BytesIO(png)
        page_h = pdf.h - pdf.b_margin
        remaining = page_h - pdf.get_y() - 4
        min_h, max_h = 55, 80
        if remaining < min_h:
            pdf.add_page()
            remaining = page_h - pdf.get_y() - 4
        chart_h = min(max_h, max(min_h, remaining))
        # 16:8 ratio kept by width 180mm; clamp aspect by computing width if needed
        chart_w = min(180, chart_h * 2.0)
        x_offset = 15 + (180 - chart_w) / 2
        pdf.image(buf, x=x_offset, y=pdf.get_y(), w=chart_w, h=chart_h)
        pdf.ln(chart_h + 2)
    except Exception:
        pass
