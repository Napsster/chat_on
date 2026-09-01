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
EMP_ID_HEADER_HINTS = ("emp id", "emp. id", "employee id", "emp code", "employee code")
FUNCTION_HEADER_HINTS = ("function", "department", "vertical", "business unit")
JOIN_DATE_HEADER_HINTS = ("joining date", "date of joining", "doj", "start date")
REFRESH_TTL = 300  # seconds (for the CSV URL)

def _read_with_retry(read_fn, label: str, attempts: int = 3, backoff: float = 2.0):
    """Call read_fn() (a directory's _read_rows) with a few retries — Google
    Sheets occasionally times out on a single request (seen in production:
    "The read operation timed out"), and that's usually transient, not a
    real outage. Without this, one slow response left a directory empty for
    the rest of its REFRESH_TTL window (up to 5 minutes) before the next
    call retried it. Raises the last exception if every attempt fails, same
    as calling read_fn() directly once — callers' existing except/log path
    is unchanged."""
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return read_fn()
        except Exception as e:
            last_exc = e
            if attempt < attempts:
                print(f"⚠️  {label} read failed (attempt {attempt}/{attempts}): {e} — retrying")
                time.sleep(backoff * attempt)
    raise last_exc

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
            rows = _read_with_retry(self._read_rows, "Candidate directory")
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


# Exact (lowercased) Status-column values that let a WhatsApp number through.
# Deliberately exact match, not substring — this is an access gate, not the
# looser pre/post-join heuristic above, so it shouldn't accidentally treat
# e.g. "Active - Notice Period" or a future new status as allowed.
ALLOWED_EMPLOYEE_STATUSES = {"active", "resigned"}


class EmployeeDirectory:
    """Gates the WhatsApp bot to current/recently-departed employees only,
    via the company's employee roster sheet — a phone number not in the
    sheet, or in it with a status other than Active/Resigned (e.g.
    Terminated, Absconding), gets a fallback reply instead of the normal
    grounded-answer flow. Separate sheet/purpose from CandidateDirectory
    above (that one is for onboarding personalization, not a hard gate)."""

    def __init__(self, app_dir: Path):
        self.sheet_id = os.environ.get("EMPLOYEE_ROSTER_SHEET_ID")
        self.sheet_range = os.environ.get("EMPLOYEE_ROSTER_SHEET_RANGE", "A1:Z10000")
        self.creds_file = os.environ.get(
            "GOOGLE_SERVICE_ACCOUNT_FILE", str(app_dir / "data" / "service_account.json")
        )
        self._index: dict[str, dict] = {}  # normalized phone -> full row
        self._status_col: str | None = None
        self._name_col: str | None = None
        self._emp_id_col: str | None = None
        self._function_col: str | None = None
        self._loaded_at = 0.0
        self.ready = False
        self._load(force=True)

    def _read_rows(self) -> list[dict]:
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
            r = list(r) + [""] * (len(headers) - len(r))
            rows.append(dict(zip(headers, r)))
        return rows

    def _load(self, force: bool = False):
        if not force and (time.time() - self._loaded_at) < REFRESH_TTL:
            return
        if not self.sheet_id:
            self.ready = False
            self._loaded_at = time.time()
            return
        try:
            rows = _read_with_retry(self._read_rows, "Employee directory")
            if not rows:
                self.ready = False
                self._loaded_at = time.time()
                return
            headers = list(rows[0].keys())
            low = {h: (h or "").strip().lower() for h in headers}
            phone_col = next((h for h in headers if any(k in low[h] for k in PHONE_HEADER_HINTS)), None)
            self._status_col = next((h for h in headers if any(k in low[h] for k in STATUS_HEADER_HINTS)), None)
            self._name_col = next(
                (h for h in headers if any(low[h] == k for k in NAME_HEADER_HINTS)), None
            ) or next((h for h in headers if "name" in low[h]), None)
            self._emp_id_col = next(
                (h for h in headers if any(k in low[h] for k in EMP_ID_HEADER_HINTS)), None
            )
            self._function_col = next(
                (h for h in headers if any(k in low[h] for k in FUNCTION_HEADER_HINTS)), None
            )
            index = {}
            if phone_col and self._status_col:
                for row in rows:
                    key = normalize_phone(row.get(phone_col, ""))
                    if key:
                        index[key] = row
            self._index = index
            self.ready = bool(index)
            self._loaded_at = time.time()
            print(f"✓ Employee directory loaded: {len(index)} rows "
                  f"(phone col: {phone_col!r}, status col: {self._status_col!r}, "
                  f"name col: {self._name_col!r})")
        except Exception as e:
            print(f"⚠️  Employee directory unavailable: {e}")
            self.ready = False
            self._loaded_at = time.time()

    def is_allowed(self, phone: str) -> bool:
        """True if the phone is in the roster with an allowed status. When
        the directory itself isn't ready (sheet ID unset, not yet shared
        with the service account, API error) this fails OPEN — returns True
        for everyone — same as the bot behaved before this gate existed.
        Only once the roster genuinely loads does a real, specific
        non-match start returning False. That order matters: it's what
        makes this safe to deploy before the sheet is actually shared."""
        self._load()
        if not self.ready:
            return True
        row = self._index.get(normalize_phone(phone))
        status = (row.get(self._status_col) or "").strip().lower() if row else ""
        return status in ALLOWED_EMPLOYEE_STATUSES

    def name_of(self, phone: str) -> str | None:
        """The roster's Employee Name for this phone, or None if it isn't
        in the roster / no name column was found — callers should fall back
        to showing the raw phone number in that case."""
        self._load()
        if not self.ready or not self._name_col:
            return None
        row = self._index.get(normalize_phone(phone))
        if not row:
            return None
        return (row.get(self._name_col) or "").strip() or None

    def emp_id_of(self, phone: str) -> str | None:
        """The roster's Employee ID for this phone, or None if it isn't in
        the roster / no ID column was found."""
        self._load()
        if not self.ready or not self._emp_id_col:
            return None
        row = self._index.get(normalize_phone(phone))
        if not row:
            return None
        return (row.get(self._emp_id_col) or "").strip() or None

    def function_of(self, phone: str) -> str | None:
        """The roster's Function/Department/Vertical for this phone, or None
        if it isn't in the roster / no such column was found — callers must
        treat None as genuinely unknown (ask or fall back to general policy
        info), never guess a function to fill the gap."""
        self._load()
        if not self.ready or not self._function_col:
            return None
        row = self._index.get(normalize_phone(phone))
        if not row:
            return None
        return (row.get(self._function_col) or "").strip() or None
