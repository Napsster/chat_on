#!/usr/bin/env python3
"""
Standalone reporting script — invoked by cron, not the FastAPI app.

  --daily    Unanswered-questions digest for today (IST calendar day), sent
             once at the end of the day.
  --weekly   Repeated-questions digest for the trailing 7 days, clustered by
             semantic similarity (paraphrases of the same question grouped
             together), sent once a week.

Cron doesn't inherit the systemd service's environment, so this script loads
its own .env file if one exists (see _load_env_file below) before falling
back to whatever's already in the process environment.
"""
import os
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

APP_DIR = Path(__file__).parent


def _load_env_file(path: Path):
    """Minimal KEY=VALUE loader — no python-dotenv dependency needed for
    the one place (cron) that actually needs this."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_env_file(APP_DIR / ".env")

from file_upload_handler import FileUploadManager
from vector_store import cluster_similar_texts
from email_utils import send_email

IS_LOCAL = os.getenv('IS_LOCAL', 'true').lower() == 'true'
PROJECT_DIR = str(APP_DIR) if IS_LOCAL else "/home/chetan/apps/onboarding-agent"
REPORT_EMAIL_TO = os.environ.get("REPORT_EMAIL_TO", "anjali.gupta@recykal.com")
REPEAT_CLUSTER_THRESHOLD = float(os.environ.get("REPEAT_CLUSTER_THRESHOLD", "0.83"))

IST = timezone(timedelta(hours=5, minutes=30))

manager = FileUploadManager(
    upload_dir=f"{PROJECT_DIR}/uploads",
    db_path=f"{PROJECT_DIR}/chatbot.db",
)


def _start_of_today_ist_as_utc_naive() -> datetime:
    """Midnight IST today, expressed as a naive UTC datetime — matches how
    QuestionLog.asked_at is stored (datetime.utcnow(), no tzinfo)."""
    now_ist = datetime.now(timezone.utc).astimezone(IST)
    start_ist = now_ist.replace(hour=0, minute=0, second=0, microsecond=0)
    return start_ist.astimezone(timezone.utc).replace(tzinfo=None)


def _fmt_ist(iso_str: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_str).replace(tzinfo=timezone.utc).astimezone(IST)
        return dt.strftime("%b %d, %I:%M %p IST")
    except Exception:
        return iso_str


def send_daily_unanswered_report():
    since = _start_of_today_ist_as_utc_naive()
    questions = manager.get_questions_since(since, unanswered_only=True)

    if not questions:
        print(f"[daily] No unanswered questions since {since} UTC — nothing to send.")
        return

    lines = [
        f"Recykal Buddy — Unanswered Questions Today ({len(questions)})",
        "These are questions the bot couldn't answer from the current knowledge base.",
        "",
    ]
    for q in questions:
        lines.append(f"- [{_fmt_ist(q['asked_at'])}] ({q['channel']}, {q['session_key']}): {q['question']}")

    body = "\n".join(lines)
    ok, detail = send_email(REPORT_EMAIL_TO, f"Recykal Buddy: {len(questions)} unanswered question(s) today", body)
    print(f"[daily] {len(questions)} unanswered question(s) — send {'OK' if ok else 'FAILED'}: {detail}")


def send_weekly_repeated_report():
    since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=7)
    questions = manager.get_questions_since(since)

    if not questions:
        print(f"[weekly] No questions logged since {since} UTC — nothing to send.")
        return

    texts = [q["question"] for q in questions]
    clusters = cluster_similar_texts(texts, threshold=REPEAT_CLUSTER_THRESHOLD)
    repeated = [c for c in clusters if len(c) >= 2]
    repeated.sort(key=len, reverse=True)

    if not repeated:
        print(f"[weekly] {len(questions)} question(s) logged, no repeats found — nothing to send.")
        return

    lines = [
        f"Recykal Buddy — Repeated Questions This Week ({len(repeated)} topic(s))",
        "Questions grouped by similar meaning, not just exact wording — worth considering",
        "for the knowledge base or a pinned FAQ if a topic comes up often.",
        "",
    ]
    for i, cluster in enumerate(repeated, 1):
        lines.append(f"{i}. Asked {len(cluster)} times:")
        for idx in cluster:
            q = questions[idx]
            lines.append(f"   - [{_fmt_ist(q['asked_at'])}] ({q['channel']}, {q['session_key']}): {q['question']}")
        lines.append("")

    body = "\n".join(lines)
    ok, detail = send_email(REPORT_EMAIL_TO, f"Recykal Buddy: {len(repeated)} repeated question topic(s) this week", body)
    print(f"[weekly] {len(repeated)} repeated topic(s) across {len(questions)} question(s) — send {'OK' if ok else 'FAILED'}: {detail}")


if __name__ == "__main__":
    if "--daily" in sys.argv:
        send_daily_unanswered_report()
    elif "--weekly" in sys.argv:
        send_weekly_repeated_report()
    else:
        print("Usage: reports.py --daily | --weekly")
        sys.exit(1)
