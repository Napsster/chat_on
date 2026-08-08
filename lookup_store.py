"""
Candidate directory — matches an inbound WhatsApp number to a row in the
onboarding log (Google Sheet) so the bot can personalise by name + context.

Source priority:
  1. Google Sheets API (PRIVATE) — set ONBOARDING_SHEET_ID and provide a service-account
     JSON key at GOOGLE_SERVICE_ACCOUNT_FILE (default data/service_account.json). The sheet
     is shared read-only with the service account; data never leaves Google's auth boundary.
     Optional ONBOARDING_SHEET_RANGE (default "A1:Z10000", or "Tab!A1:Z10000").
  2. ONBOARDING_SHEET_CSV  : a "Publish to web → CSV" URL (public link — testing only).
  3. ONBOARDING_SHEET_FILE : a local CSV path (default: data/onboarding_log.csv).

Matching is on the last 10 digits of the phone number, so formatting
differences (+91, spaces, whatsapp: prefix) don't matter. Disabled quietly
if no source is present — the bot then behaves exactly as before.
"""

import os
import csv
import io
import re
import time
from datetime import datetime
from pathlib import Path

import requests

PHONE_HEADER_HINTS = ("phone", "mobile", "whatsapp", "contact", "number", "cell")
NAME_HEADER_HINTS = ("candidate name", "name", "candidate", "full name")
STATUS_HEADER_HINTS = ("status", "joining stage", "stage")
JOIN_DATE_HEADER_HINTS = ("joining date", "date of joining", "doj", "start date")
REFRESH_TTL = 300  # seconds (for the CSV URL)

# Status-column values → segment. Substring match against the lowercased cell.
POST_JOIN_STATUS_MARKERS = ("joined", "active", "onboarded", "working", "employee")
PRE_JOIN_STATUS_MARKERS = (
    "yet to join", "not joined", "pending", "offer", "upcoming", "candidate", "in progress",
    "accepted",
)

_DATE_FORMATS = ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%d %b %Y", "%d %B %Y")


def normalize_phone(value: str) -> str:
    digits = re.sub(r"\D", "", value or "")
    return digits[-10:] if len(digits) >= 10 else digits


class CandidateDirectory:
    def __init__(self, app_dir: Path):
        # Source 1: private Google Sheets API
        self.sheet_id = os.environ.get("ONBOARDING_SHEET_ID")
        self.sheet_range = os.environ.get("ONBOARDING_SHEET_RANGE", "A1:Z10000")
        self.creds_file = os.environ.get(
            "GOOGLE_SERVICE_ACCOUNT_FILE", str(app_dir / "data" / "service_account.json")
        )
        # Source 2/3: published CSV URL / local CSV file
        self.url = os.environ.get("ONBOARDING_SHEET_CSV")
        self.file = os.environ.get(
            "ONBOARDING_SHEET_FILE", str(app_dir / "data" / "onboarding_log.csv")
        )
        self.source = "none"
        self._index: dict[str, dict] = {}
        self._phone_col: str | None = None
        self._name_col: str | None = None
        self._status_col: str | None = None
        self._join_date_col: str | None = None
        self._loaded_at = 0.0
        self.ready = False
        self._load(force=True)

    # ---- loading ----------------------------------------------------------
    def _read_sheets_api(self) -> list[dict]:
        """Private read via Google Sheets API using a service account."""
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        creds = service_account.Credentials.from_service_account_file(
            self.creds_file,
            scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
        )
        svc = build("sheets", "v4", credentials=creds, cache_discovery=False)
        res = svc.spreadsheets().values().get(
            spreadsheetId=self.sheet_id, range=self.sheet_range
        ).execute()
        values = res.get("values", [])
        if len(values) < 2:
            return []
        headers = [h.strip() for h in values[0]]
        rows = []
        for r in values[1:]:
            r = list(r) + [""] * (len(headers) - len(r))  # pad short rows
            rows.append(dict(zip(headers, r)))
        return rows

    def _read_rows(self) -> list[dict]:
        if self.sheet_id and Path(self.creds_file).exists():
            self.source = "sheets_api"
            return self._read_sheets_api()
        if self.url:
            self.source = "csv_url"
            r = requests.get(self.url, timeout=15)
            r.raise_for_status()
            return list(csv.DictReader(io.StringIO(r.text)))
        if self.file and Path(self.file).exists():
            self.source = "csv_file"
            return list(csv.DictReader(io.StringIO(Path(self.file).read_text(encoding="utf-8"))))
        self.source = "none"
        return []

    def _pick_columns(self, rows: list[dict]):
        headers = list(rows[0].keys()) if rows else []
        low = {h: (h or "").strip().lower() for h in headers}
        # phone column by header hint, else by value pattern
        self._phone_col = next(
            (h for h in headers if any(k in low[h] for k in PHONE_HEADER_HINTS)), None
        )
        if not self._phone_col:
            for h in headers:
                vals = [row.get(h, "") for row in rows[:10]]
                if sum(len(normalize_phone(v)) == 10 for v in vals) >= max(1, len(vals) // 2):
                    self._phone_col = h
                    break
        self._name_col = next(
            (h for h in headers if any(low[h] == k for k in NAME_HEADER_HINTS)), None
        ) or next(
            (h for h in headers if "name" in low[h]), None
        )
        self._status_col = next(
            (h for h in headers if any(k in low[h] for k in STATUS_HEADER_HINTS)), None
        )
        self._join_date_col = next(
            (h for h in headers if any(k in low[h] for k in JOIN_DATE_HEADER_HINTS)), None
        )

    def _load(self, force: bool = False):
        if not force and (time.time() - self._loaded_at) < REFRESH_TTL:
            return
        try:
            rows = self._read_rows()
            if not rows:
                self.ready = False
                self._loaded_at = time.time()
                return
            self._pick_columns(rows)
            index = {}
            if self._phone_col:
                for row in rows:
                    key = normalize_phone(row.get(self._phone_col, ""))
                    if key:
                        index[key] = row
            self._index = index
            self.ready = bool(index)
            self._loaded_at = time.time()
            print(f"✓ Candidate directory loaded: {len(index)} rows "
                  f"(phone col: {self._phone_col!r}, name col: {self._name_col!r})")
        except Exception as e:
            print(f"⚠️  Candidate directory unavailable: {e}")
            self.ready = False
            self._loaded_at = time.time()

    # ---- lookup -----------------------------------------------------------
    def lookup(self, phone: str) -> dict | None:
        self._load()  # refresh if TTL elapsed
        if not self.ready:
            return None
        return self._index.get(normalize_phone(phone))

    def list_all(self) -> list[dict]:
        """Every indexed candidate row, each tagged with its normalized
        phone (10-digit, for matching) and raw phone (as entered in the
        sheet, for actually sending a message — may or may not include a
        country code depending on how the sheet was filled in)."""
        self._load()
        if not self.ready:
            return []
        return [
            {"phone_normalized": key, "phone_raw": row.get(self._phone_col, ""), "row": row}
            for key, row in self._index.items()
        ]

    def name_of(self, row: dict) -> str | None:
        if row and self._name_col:
            return (row.get(self._name_col) or "").strip() or None
        return None

    def segment_of(self, row: dict) -> str | None:
        """"pre_join" (offer accepted, not yet started) or "post_join" (already
        working), derived from a status-like column if present, else a
        joining-date column compared to today. None if neither is available
        or can't be parsed — caller should fall back to asking the user."""
        if not row:
            return None

        if self._status_col:
            status = (row.get(self._status_col) or "").strip().lower()
            if status:
                if any(marker in status for marker in POST_JOIN_STATUS_MARKERS):
                    return "post_join"
                if any(marker in status for marker in PRE_JOIN_STATUS_MARKERS):
                    return "pre_join"

        if self._join_date_col:
            raw = (row.get(self._join_date_col) or "").strip()
            for fmt in _DATE_FORMATS:
                try:
                    join_date = datetime.strptime(raw, fmt)
                    return "post_join" if join_date.date() <= datetime.now().date() else "pre_join"
                except ValueError:
                    continue

        return None

    def profile_block(self, row: dict) -> str:
        """Render the candidate's row as a labelled context block."""
        lines = []
        for k, v in row.items():
            if k == self._phone_col:
                continue
            v = (v or "").strip()
            if v:
                lines.append(f"- {k.strip()}: {v}")
        return "\n".join(lines)
