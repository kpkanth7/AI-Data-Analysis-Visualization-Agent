"""Owner session persistence: save/load/list JSON sessions."""
import json
import os
import datetime
from pathlib import Path

SESSIONS_DIR = Path("logs/sessions")


def _ensure_dir() -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def save_session(chat_history: list, label: str = "") -> str:
    """Save chat_history to a JSON file. Returns the file path."""
    _ensure_dir()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = "".join(c for c in label if c.isalnum() or c in "-_ ")[:40].strip()
    fname = f"{ts}{'_' + safe_label.replace(' ', '_') if safe_label else ''}.json"
    path = SESSIONS_DIR / fname

    # Serialise: store role + text + analysis metadata (no plotly fig objects)
    serialisable = []
    for msg in chat_history:
        entry = {"role": msg["role"], "content": msg["content"]}
        if msg.get("analysis"):
            a = msg["analysis"]
            entry["analysis"] = {
                "answer": a.answer,
                "chart_config": a.chart_config,
                "data_preview": a.data_preview,
                "sql_used": a.sql_used,
                "datasets_used": a.datasets_used,
                "sub_results": [
                    {
                        "index": s.index,
                        "question": s.question,
                        "answer": s.answer,
                        "chart_config": s.chart_config,
                        "sql_used": s.sql_used,
                        "data_preview": s.data_preview,
                    }
                    for s in (a.sub_results or [])
                ],
            }
        serialisable.append(entry)

    path.write_text(json.dumps({"saved_at": ts, "messages": serialisable}, indent=2))
    return str(path)


def load_session(path: str) -> dict:
    return json.loads(Path(path).read_text())


def list_sessions() -> list[dict]:
    """Return sessions sorted newest first, with metadata."""
    _ensure_dir()
    sessions = []
    for p in sorted(SESSIONS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(p.read_text())
            ts_raw = data.get("saved_at", p.stem[:15])
            try:
                dt = datetime.datetime.strptime(ts_raw, "%Y%m%d_%H%M%S")
                ts_display = dt.strftime("%b %d %Y, %H:%M")
            except Exception:
                ts_display = ts_raw
            msg_count = len(data.get("messages", []))
            # Derive label from filename (after timestamp)
            stem = p.stem
            label = stem[16:].replace("_", " ").strip() if len(stem) > 16 else ""
            sessions.append({
                "path": str(p),
                "filename": p.name,
                "ts_display": ts_display,
                "label": label,
                "message_count": msg_count,
                "data": data,
            })
        except Exception:
            continue
    return sessions[:50]


def delete_session(path: str) -> None:
    Path(path).unlink(missing_ok=True)


def auto_save_if_nonempty(chat_history: list) -> str | None:
    """Auto-save only if there's at least one exchange. Returns path or None."""
    user_msgs = [m for m in chat_history if m["role"] == "user"]
    if not user_msgs:
        return None
    return save_session(chat_history, label="autosave")
