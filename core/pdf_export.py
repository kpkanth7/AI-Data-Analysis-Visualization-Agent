"""Export utilities: PNG chart, CSV dataframe, and full-session PDF."""
import io
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


# ── PNG chart export ──────────────────────────────────────────────────────────

def chart_to_png(fig) -> bytes:
    """Convert a Plotly figure to PNG bytes using kaleido."""
    return fig.to_image(format="png", width=1000, height=500, scale=2)


# ── CSV dataframe export ──────────────────────────────────────────────────────

def dataframe_to_csv(rows: list[dict]) -> bytes:
    if not rows:
        return b""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


# ── Full session PDF export ───────────────────────────────────────────────────

def session_to_pdf(chat_history: list, charts: dict[str, Any] | None = None) -> bytes:
    """
    Build a structured PDF of the session.
    chat_history: list of {role, content, analysis?}
    charts: mapping of message index → plotly Figure (optional, for embedding PNGs)
    """
    if not _FPDF_AVAILABLE:
        raise ImportError("fpdf2 required: pip install fpdf2")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Header ────────────────────────────────────────────────────────────────
    pdf.set_fill_color(15, 23, 42)        # dark navy
    pdf.rect(0, 0, 210, 28, style="F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_xy(10, 6)
    pdf.cell(0, 10, _s("Data Analysis Agent -- Session Export"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_xy(10, 18)
    pdf.cell(0, 6, datetime.datetime.now().strftime("Generated %B %d, %Y at %H:%M"))
    pdf.ln(14)

    pdf.set_text_color(30, 30, 30)

    msg_idx = 0
    for msg in chat_history:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            # ── Question bubble ───────────────────────────────────────────────
            pdf.set_fill_color(224, 242, 254)   # light blue
            pdf.set_draw_color(59, 130, 246)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(30, 64, 175)
            pdf.cell(0, 7, f"Q{msg_idx + 1}  {_truncate(content, 120)}",
                     fill=True, border="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(2)

        elif role == "assistant":
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(30, 30, 30)

            analysis = msg.get("analysis")
            if analysis:
                # Answer text
                answer = getattr(analysis, "answer", "") or content
                _write_wrapped(pdf, answer, max_chars=2000)

                # Data preview table
                preview = getattr(analysis, "data_preview", []) or []
                if preview:
                    _write_table(pdf, preview[:10])

                # Sub-results
                for sub in (getattr(analysis, "sub_results", []) or []):
                    pdf.set_font("Helvetica", "B", 10)
                    pdf.set_text_color(80, 80, 80)
                    pdf.cell(0, 6, f"[{sub.index}] {_truncate(sub.question, 100)}",
                             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                    pdf.set_font("Helvetica", "", 10)
                    pdf.set_text_color(30, 30, 30)
                    _write_wrapped(pdf, sub.answer or "", max_chars=1000)
                    if sub.data_preview:
                        _write_table(pdf, sub.data_preview[:8])
                    # Embed chart image if available
                    if charts and f"sub_{msg_idx}_{sub.index}" in charts:
                        _embed_chart(pdf, charts[f"sub_{msg_idx}_{sub.index}"])

                # Top-level chart
                if charts and str(msg_idx) in charts:
                    _embed_chart(pdf, charts[str(msg_idx)])

                # Datasets used
                ds = getattr(analysis, "datasets_used", []) or []
                if ds:
                    pdf.set_font("Helvetica", "I", 8)
                    pdf.set_text_color(120, 120, 120)
                    pdf.cell(0, 5, _s(f"Datasets: {', '.join(ds)}"),
                             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            else:
                _write_wrapped(pdf, content, max_chars=2000)

            pdf.ln(4)
            pdf.set_draw_color(220, 220, 220)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(4)
            msg_idx += 1

    return bytes(pdf.output())


# ── Helpers ───────────────────────────────────────────────────────────────────

# Helvetica (built-in fpdf2 font) only covers latin-1 (ISO 8859-1).
# Replace every out-of-range character so we never crash on Unicode input.
_UNICODE_MAP = str.maketrans({
    "—": "--",   # em dash
    "–": "-",    # en dash
    "‘": "'",    # left single quote
    "’": "'",    # right single quote
    "“": '"',    # left double quote
    "”": '"',    # right double quote
    "…": "...",  # ellipsis
    "•": "*",    # bullet
    "’": "'",    # curly apostrophe
    "·": ".",    # middle dot
    "−": "-",    # minus sign
    "×": "x",    # multiplication sign
})


def _s(text: str) -> str:
    """Sanitise text to latin-1 safe characters for fpdf2 built-in fonts."""
    text = text.translate(_UNICODE_MAP)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _truncate(text: str, n: int) -> str:
    text = _s(text)
    return text if len(text) <= n else text[:n - 1] + "."


def _write_wrapped(pdf, text: str, max_chars: int = 1500) -> None:
    text = _s(text[:max_chars] + ("..." if len(text) > max_chars else ""))
    pdf.multi_cell(0, 5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)


def _write_table(pdf, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    col_w = min(35, max(15, 170 // max(len(cols), 1)))
    visible_cols = cols[:min(len(cols), 170 // col_w)]

    # Header row
    pdf.set_fill_color(243, 244, 246)
    pdf.set_font("Helvetica", "B", 7)
    pdf.set_text_color(50, 50, 50)
    for c in visible_cols:
        pdf.cell(col_w, 5, _truncate(str(c), col_w // 2 + 2), border=1, fill=True)
    pdf.ln()

    # Data rows
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(30, 30, 30)
    for row in rows:
        for c in visible_cols:
            val = str(row.get(c, ""))
            pdf.cell(col_w, 5, _truncate(val, col_w // 2 + 2), border=1)
        pdf.ln()
    pdf.ln(2)


def _embed_chart(pdf, fig) -> None:
    try:
        png = chart_to_png(fig)
        buf = io.BytesIO(png)
        x = pdf.get_x()
        y = pdf.get_y()
        if y > 240:
            pdf.add_page()
            y = pdf.get_y()
        pdf.image(buf, x=10, y=y, w=180)
        pdf.ln(95)
    except Exception:
        pass
