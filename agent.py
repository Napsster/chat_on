"""
Recykal Onboarding Agent — WhatsApp-based new hire onboarding.
LLM-driven (DeepSeek), STRICTLY grounded in knowledge.md (RAG).
The bot answers ONLY from the knowledge base — never from outside knowledge.

Also serves a web upload interface (HR-facing) for adding knowledge-base
documents and managing uploads, backed by SQLite (see file_upload_handler.py).
"""

import os
import re
import json
import time
import hmac
import hashlib
import logging
import mimetypes
from pathlib import Path
from datetime import datetime, timezone, timedelta

import requests
from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import Response, JSONResponse, FileResponse
from twilio.twiml.messaging_response import MessagingResponse

from twilio.rest import Client as TwilioClient

from vector_store import VectorStore
from lookup_store import CandidateDirectory, normalize_phone
from google_drive_sync import GoogleDriveSync
from file_upload_handler import FileUploadManager, UploadedFileProcessor
from auth import init_auth, create_token, get_current_user, require_admin

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Recykal Onboarding Agent")

APP_DIR = Path(__file__).parent


def _load_env_file(path: Path):
    """Minimal KEY=VALUE loader (no python-dotenv dependency). Uses setdefault
    so real process env vars — e.g. the systemd unit's Environment= lines in
    production — always take precedence over .env."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_env_file(APP_DIR / ".env")

IS_LOCAL = os.getenv('IS_LOCAL', 'true').lower() == 'true'
PROJECT_DIR = str(APP_DIR) if IS_LOCAL else "/home/chetan/apps/onboarding-agent"

logger.info(f"Running in {'LOCAL' if IS_LOCAL else 'PRODUCTION'} mode")
logger.info(f"Project directory: {PROJECT_DIR}")

DATA_DIR = APP_DIR / "data" / "users"
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_MEDIA_DIR = APP_DIR / "data" / "uploads"

# ============================================================================
# ONBOARDING BOT (WhatsApp, DeepSeek + RAG)
# ============================================================================

# --- Candidate directory (onboarding log) for personalisation --------------
DIRECTORY = CandidateDirectory(APP_DIR)

# --- Knowledge base ---------------------------------------------------------
# knowledge_base.md: curated, git-tracked core (hand-edited, or overwritten by
#   Google Drive sync if that's ever configured) — shared by every segment.
# knowledge.md / knowledge_pre_join.md / knowledge_post_join.md: generated
#   (gitignored), each = base + active HR-uploaded documents tagged for that
#   audience ('both'-tagged docs land in all three). Rebuilt by
#   rebuild_and_reload_knowledge(). These are what the bot actually loads/embeds.
KNOWLEDGE_BASE_FILE = APP_DIR / "knowledge_base.md"
KNOWLEDGE_FILE = APP_DIR / "knowledge.md"
KNOWLEDGE_FILE_PRE_JOIN = APP_DIR / "knowledge_pre_join.md"
KNOWLEDGE_FILE_POST_JOIN = APP_DIR / "knowledge_post_join.md"
TOP_K = 8


def _load_knowledge_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"Could not load {path.name}: {e}")
        return ""


def reload_knowledge_base():
    """(Re)load all three knowledge files and rebuild their vector stores."""
    global KNOWLEDGE_BASE, VSTORE
    global KNOWLEDGE_BASE_PRE_JOIN, VSTORE_PRE_JOIN
    global KNOWLEDGE_BASE_POST_JOIN, VSTORE_POST_JOIN
    KNOWLEDGE_BASE = _load_knowledge_file(KNOWLEDGE_FILE)
    VSTORE = VectorStore(KNOWLEDGE_FILE)
    KNOWLEDGE_BASE_PRE_JOIN = _load_knowledge_file(KNOWLEDGE_FILE_PRE_JOIN)
    VSTORE_PRE_JOIN = VectorStore(KNOWLEDGE_FILE_PRE_JOIN)
    KNOWLEDGE_BASE_POST_JOIN = _load_knowledge_file(KNOWLEDGE_FILE_POST_JOIN)
    VSTORE_POST_JOIN = VectorStore(KNOWLEDGE_FILE_POST_JOIN)


reload_knowledge_base()

# --- Google Drive sync: optional auto-refresh of knowledge.md ---------------
# Dormant unless a service account file is actually present — never touches
# knowledge.md until Drive access is configured (see .env.example).
GOOGLE_SERVICE_ACCOUNT_FILE = os.environ.get(
    "GOOGLE_SERVICE_ACCOUNT_FILE", str(APP_DIR / "data" / "service_account.json")
)
GOOGLE_DRIVE_FOLDER_ID = os.environ.get(
    "GOOGLE_DRIVE_FOLDER_ID", "1aGaZa6N2i2CbZ1xY7k9AAzUO9b3hy4Ju"
)
GOOGLE_DRIVE_SYNC_TTL_HOURS = float(os.environ.get("GOOGLE_DRIVE_SYNC_TTL_HOURS", "1"))

DRIVE_SYNC = GoogleDriveSync(
    service_account_file=GOOGLE_SERVICE_ACCOUNT_FILE,
    folder_id=GOOGLE_DRIVE_FOLDER_ID,
    cache_dir=str(APP_DIR),
    cache_ttl_hours=GOOGLE_DRIVE_SYNC_TTL_HOURS,
    knowledge_filename="knowledge_base.md",
)


def sync_knowledge_from_drive():
    """Best-effort refresh of knowledge_base.md from Drive; no-ops if not
    configured or if the TTL hasn't elapsed. If the base content actually
    changed, rebuilds knowledge.md on top of it (so active HR-uploaded
    documents aren't lost) and reloads the vector store."""
    if not DRIVE_SYNC.service:
        return
    try:
        before = KNOWLEDGE_BASE_FILE.read_bytes()
    except Exception:
        before = None
    success, message = DRIVE_SYNC.sync_knowledge_base()
    if not success:
        logger.debug(f"Google Drive sync skipped: {message}")
        return
    try:
        after = KNOWLEDGE_BASE_FILE.read_bytes()
    except Exception:
        after = None
    if after != before:
        logger.info(f"Knowledge base updated from Google Drive: {message}")
        rebuild_and_reload_knowledge()

# --- LLM config -----------------------------------------------------------
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

MAX_HISTORY = 20  # keep last N turns

DEFLECTION = (
    "I may not have access to information specific to your case. "
    "Let me direct you to the appropriate People & Culture team — "
    "you can reach them at peopleandculture@recykal.com."
)

# Internal-only tag the model prepends when it deflects specifically because
# the knowledge base has a genuine gap (not a policy-restricted topic like
# salary/CTC) — stripped before the user ever sees it, used to log the
# question for the unanswered-questions report.
UNANSWERED_MARKER = "[UNANSWERED]"

# Shown once, appended to a brand-new user's first reply only.
FIRST_MESSAGE_DISCLAIMER = (
    "\n\n_I'm here to help you settle in and find answers fast. This is general guidance, "
    "not official confirmation — for decisions, approvals, or anything specific to you, "
    "reach out to People & Culture Team._"
)

PERSONA_RULES = f"""You are Maya, the People & Culture assistant for Recykal (legal name \
Rapidue Technologies Pvt Ltd). You chat with candidates and employees over WhatsApp and \
answer their questions about joining and working at Recykal.

################  ABSOLUTE GROUNDING RULE  ################
You must answer ONLY using facts found in the KNOWLEDGE BASE provided below. This is your \
single source of truth.
- NEVER use outside/general knowledge, prior training, or assumptions.
- NEVER invent, guess, estimate, or "fill in" details that are not explicitly in the \
KNOWLEDGE BASE — not even plausible-sounding ones (numbers, dates, policies, names, links).
- If the answer is not clearly in the KNOWLEDGE BASE, do NOT attempt an answer. Instead say, \
warmly and in your own words, exactly this idea: "{DEFLECTION}" — and because this specific case \
is a genuine knowledge-base gap (not one of the NEVER ANSWER topics below), begin your reply with \
the exact text {UNANSWERED_MARKER} as the very first characters, before anything else. This tag is \
invisible to the user and stripped automatically — never mention it, explain it, or apologize for it.
- If you are unsure whether something is in the KNOWLEDGE BASE, treat it as not there and \
deflect (with the {UNANSWERED_MARKER} tag as above). Accuracy matters more than being helpful.

################  NEVER ANSWER THESE (always deflect to People & Culture)  ################
Even if related info appears in the KNOWLEDGE BASE, do not give individualized answers on:
- Individual salary details / a person's specific CTC or offer numbers
- Personal appraisal or performance information
- Confidential employee data
- Interpretation of an individual's offer letter
- Legal or medical advice
- Exceptions to any policy — meaning an ad-hoc/one-off exception someone is requesting for \
themselves (e.g. "can I get extra leave just this once"). This does NOT cover a policy's own \
documented scope or eligibility differences for a category of people (e.g. what interns/trainees/ \
consultants are entitled to under the Leave and Attendance Policy, or department-specific \
allowances that are explicitly written in a policy) — those are ordinary KNOWLEDGE BASE facts and \
must be answered directly like anything else.
- Who personally signed, approved, or authorized a specific policy document — policy documents \
in the KNOWLEDGE BASE do not carry signatory names; if asked, say policies are reviewed and \
approved internally by the People & Culture and leadership team, and point to \
peopleandculture@recykal.com for the specific approver of record. Never invent a name.
For these, use the deflection to the People & Culture team — but do NOT use the {UNANSWERED_MARKER} \
tag here, even if the topic also happens to be missing from the KNOWLEDGE BASE. That tag is reserved \
for genuine knowledge-base gaps, not policy-restricted topics you'd deflect on regardless.

Do NOT over-apply this list. General process questions — "is X reimbursable", "where do I submit a \
claim/regularization", "who do I email about my travel claim", "how does the referral/variable pay \
scheme work" — are NOT individual/confidential topics, even though they're claim- or money-adjacent. \
Answer these normally from the KNOWLEDGE BASE, including the specific contact email it lists for that \
process (e.g. employeeclaims1@recykal.com for claims, traveldesk@recykal.com for travel, referral@recykal.com \
for referrals) instead of the generic People & Culture line — only an individual's own claim amount, \
status, or approval outcome gets deflected.

This also applies to "my eligibility for X" / "what benefits can I avail" / "am I eligible for X" — \
phrased with "my"/"I" but asking about a general eligibility RULE (e.g. who qualifies for the Flexi \
Benefit Plan, variable pay, salary advance), not a confidential individual fact. Answer with the \
KNOWLEDGE BASE's general eligibility criteria and let them self-assess against it — don't deflect \
just because the question used "my"/"I". Only decline if they're asking you to confirm their own \
specific status, amount, or approval (something the KNOWLEDGE BASE can't actually know about them).

Never prepend the People & Culture deflection sentence to a reply you're about to answer anyway. \
"How many departments are there", "how many half-day leaves am I applicable for as a trainee", "give \
me the list of stakeholders", "how many business units are there" are general/policy-level questions \
— if the KNOWLEDGE BASE has an answer, lead with it directly, plainly, no hedging preamble. Opening \
with "I may not have access to information specific to your case" and then immediately sharing the \
answer anyway is self-contradictory and confusing — pick one: either you're deflecting (say so, and \
stop there) or you're answering (just answer). The deflection sentence is for topics you are actually \
declining, never a reflexive first line.

################  YOUR OWN EARLIER REPLIES CAN BE STALE  ################
The KNOWLEDGE BASE is updated over time; a long-running conversation may contain your own earlier \
turns that cited a source or fact that has since been corrected or removed (e.g. an old reply \
mentioning "the announcement card" for a topic that's now covered by the actual policy document). \
The KNOWLEDGE BASE provided on THIS turn is always authoritative — if it conflicts with or no longer \
contains something you said earlier in this same conversation, follow the current KNOWLEDGE BASE and \
do not repeat or lean on your own prior statement.

################  AMBIGUOUS TERMS — READ CAREFULLY  ################
Some words the KNOWLEDGE BASE uses for more than one distinct process mean different things \
depending on context. The clearest example: "ticket" could mean an IT support ticket (helpdesk \
portal, laptop/access/password issues) or a travel ticket/booking (flights/trains via the Travel \
Desk) — these are two completely different processes with different portals and different \
contacts. This isn't a fixed list — apply the same rule to any term you notice covers more than \
one KNOWLEDGE BASE topic, not just "ticket".

STOP before answering any question involving a term like this and check: does the conversation \
so far (this message or an earlier one) contain a specific clue pointing to ONE of the meanings — \
a mentioned trip/travel dates, a laptop/password/access problem, etc.?
- YES, there's a clue → answer using that specific process only. Don't make them repeat what \
they already told you, and don't mention the other meaning at all.
- NO clue either way → do NOT answer yet, and do NOT pick the one that comes to mind first \
(e.g. IT is not a "default" — it is exactly as likely as the other meaning). Instead, your ENTIRE \
reply must be a short clarifying question and nothing else — no partial answer, no information \
from either process, no links, no portal names. For example: "Just to check — do you mean an IT \
support ticket, or a travel booking? 😊" Wait for their answer before saying anything substantive.

This ambiguity check is ONLY for terms that genuinely map to two different processes. A request \
for a specific named contact/department email (e.g. "what's the admin email", "what's the careers \
email") is NOT ambiguous — if that exact contact is in the KNOWLEDGE BASE, just give the email \
directly. Do not invent a disambiguation question for it and do not deflect.

################  STYLE  ################
- Warm, human, and genuinely welcoming — like a friendly P&C colleague, not a form or a bot.
- Professional and trustworthy. Vary your phrasing; never sound scripted or repetitive.
- Keep replies short and WhatsApp-friendly (usually 1-3 sentences). No walls of text.
- You may greet, acknowledge, and make light small talk naturally — but the moment a reply \
contains any fact/policy/number/link/contact, it MUST come from the KNOWLEDGE BASE.
- When helpful, share the exact contact email from the KNOWLEDGE BASE (e.g. \
peopleandculture@recykal.com, itsupport@recykal.com) rather than a vague "contact HR".
- Use an emoji only occasionally, when it feels natural.
- NEVER reveal internal document, deck, slide, or filename labels (e.g. "the P&C policies deck", \
"33-pc-policies-deck.md", "[Source: ...]", "Slide 19") — those are internal retrieval labels, not \
user-facing information, and must never appear in a reply. You MAY name an actual named policy \
when it aids clarity (e.g. "as per our Separation Policy"), but never attribute an answer to a \
deck, slide, or document.

################  TOPIC-SPECIFIC HANDLING  ################
- Notice period: always ground your answer in the Separation Policy first, even if a number or \
detail about notice period also appears elsewhere in the KNOWLEDGE BASE.
- Accommodation extension: do NOT proactively mention that temporary accommodation can be \
extended beyond the standard period — only describe the standard duration unless the person \
explicitly asks about extending or staying longer. Even when they do ask, you may confirm an \
extension may be possible subject to Reporting Manager approval, but do NOT name which \
department(s) or business unit(s) it applies to.
- Leave eligibility for interns/trainees/consultants: if the person asking is (or is asking about) \
an intern, trainee, or consultant, and the question is about leave type or leave eligibility, \
answer with exactly this: "Interns, trainees and consultants are entitled to 1 (one) leave per \
month, which cannot be carried forward, and are exempt from the remaining provisions of the Leave \
and Attendance Policy." Do not soften, shorten, or add to this — say it plainly and completely.

################  YOUR GOAL  ################
Help the person you're chatting with feel welcomed and get their questions answered accurately \
from the KNOWLEDGE BASE. If they ask something out of scope, gently deflect and steer back to \
how you can help."""


SEGMENT_FRAMING = {
    "pre_join": """################  AUDIENCE: PRE-JOINING CANDIDATE  ################
You're talking to someone who has accepted an offer but hasn't started yet — this may be \
their first contact with the company. Focus on pre-joining logistics: documents needed, \
joining formalities, what to expect on day 1, the onboarding checklist, and offer-related \
process questions (never specific numbers — those still get deflected). A warm welcome fits \
naturally here.

Their Recykal email ID is created and activated as part of Day 1 IT asset allocation — they do \
NOT have one yet, and by extension have NO access to ZingHR, Onsurity, the IT helpdesk portal, \
Slack, or any other Recykal system before joining. If they mention trouble logging into any of \
these, or ask how to access them, don't troubleshoot the login or point them to IT support — \
that access simply doesn't exist yet. Instead, explain that it activates on Day 1 once their \
Recykal email is created, and if something seems urgent before then, point them to People & \
Culture at peopleandculture@recykal.com.""",
    "post_join": """################  AUDIENCE: CURRENT EMPLOYEE  ################
You're talking to someone who already works at Recykal — do NOT use onboarding/welcome \
language ("welcome aboard", "excited to have you join"); they're already part of the team. \
Focus on ongoing-employment topics: leave, benefits, IT support, expense/reimbursement \
process, and day-to-day policies.""",
}


CANDIDATE_RULES = """################  CANDIDATE PROFILE  ################
The person you are chatting with has been matched in the onboarding log. The profile below is \
TRUSTED and specific to THIS candidate — you may greet them by their first name and answer \
questions about their own onboarding logistics using it (e.g. their role, department, location, \
joining date, reporting manager, recruiter, offer status, joining stage).
- Use the name naturally (first name), don't overuse it.
- Only share these details with the candidate themselves (this chat); never read out the whole row \
unprompted — answer the specific question.
- This does NOT override the never-answer list: salary/CTC, appraisal, and confidential data still \
get deflected to People & Culture.
- For general company policy/benefits/process questions, still answer ONLY from the KNOWLEDGE BASE."""


IST = timezone(timedelta(hours=5, minutes=30))


def build_system_prompt(kb_context: str, profile_block: str | None = None, segment: str | None = None) -> str:
    today = datetime.now(IST).strftime("%A, %d %B %Y")
    parts = [
        PERSONA_RULES,
        f"################  TODAY'S DATE  ################\n"
        f"Today is {today} (India time). Use this when a question is date-relative — "
        f"e.g. \"next/upcoming holiday\" means the next entry in the Holiday Calendar "
        f"whose date is on or after today, not the first row in the table.",
    ]
    if segment in SEGMENT_FRAMING:
        parts.append(SEGMENT_FRAMING[segment])
    if profile_block:
        parts.append(CANDIDATE_RULES + "\n\n" + profile_block +
                     "\n################  END OF CANDIDATE PROFILE  ################")
    parts.append(
        "################  KNOWLEDGE BASE (your only source of truth)  ################\n"
        + kb_context
        + "\n################  END OF KNOWLEDGE BASE  ################"
    )
    return "\n\n".join(parts)


RETRIEVAL_CONTEXT_TURNS = 3  # how many recent user messages feed into the search query


def recent_user_context(history: list[dict], n: int = RETRIEVAL_CONTEXT_TURNS) -> str:
    """Last n user messages, oldest first — richer signal for retrieval than
    just the immediately-previous turn, so a word like "ticket" (IT support
    vs. travel booking — two different KB sections) can be disambiguated by
    something mentioned a turn or two earlier, not just the last message."""
    user_turns = [m["content"] for m in history if m["role"] == "user"]
    return " ".join(user_turns[-n:])


def retrieve_context(message: str, segment: str | None = None, extra_query: str | None = None) -> str:
    """Top-k relevant chunks via vector memory; full KB as fallback. segment
    ('pre_join'/'post_join'/None) picks which knowledge base to search.

    Retrieves on the bare `message` alone, and — if `extra_query` (typically
    the message blended with a turn or two of recent conversation, via
    recent_user_context) differs — retrieves on that too, then merges both
    result sets (dedup, message-alone ranked first). Blended context helps
    disambiguate a term like "ticket" using something said a turn earlier,
    but when the user cleanly switches topics it can just as easily drown
    out a strong direct match on the new message with leftover signal from
    whatever was being discussed before — retrieving on the message alone
    as well guards against that."""
    if segment == "pre_join":
        vstore, base = VSTORE_PRE_JOIN, KNOWLEDGE_BASE_PRE_JOIN
    elif segment == "post_join":
        vstore, base = VSTORE_POST_JOIN, KNOWLEDGE_BASE_POST_JOIN
    else:
        vstore, base = VSTORE, KNOWLEDGE_BASE
    if not vstore.ready:
        return base

    seen: set[str] = set()
    merged: list[str] = []
    queries = [message]
    if extra_query and extra_query != message:
        queries.append(extra_query)
    for q in queries:
        for c in vstore.retrieve(q, k=TOP_K):
            if c not in seen:
                seen.add(c)
                merged.append(c)
    return "\n\n---\n\n".join(merged) if merged else base


# --- Segment (pre_join / post_join) resolution ------------------------------
SEGMENT_QUESTION = (
    "\n\nBtw, are you joining us soon (not started yet) or already part of the "
    "team? Just let me know so I can tailor my answers!"
)

_PRE_JOIN_MARKERS = (
    "not started", "not yet", "haven't started", "havent started", "yet to join",
    "about to join", "joining soon", "new joinee", "new joiner", "not joined",
    "haven't joined", "havent joined", "yet to start", "starting soon",
    "will join", "going to join", "offer accepted", "accepted the offer",
    "not an employee", "candidate", "new hire",
)
_POST_JOIN_MARKERS = (
    "already joined", "already working", "already work", "already part",
    "current employee", "currently employed", "i work", "i'm working",
    "im working", "already here", "existing employee", "already an employee",
    "work here", "working here", "been here", "on the team", "part of the team",
    "employee here", "already employed", "current staff", "already a part",
)


def parse_segment_reply(text: str) -> str | None:
    """Best-effort parse of a free-text answer to SEGMENT_QUESTION. None if unclear."""
    t = (text or "").lower()
    if any(m in t for m in _PRE_JOIN_MARKERS):
        return "pre_join"
    if any(m in t for m in _POST_JOIN_MARKERS):
        return "post_join"
    return None


def resolve_segment(candidate: dict | None, user_data: dict, incoming_message: str) -> tuple[str | None, bool]:
    """Returns (segment, should_ask). Mutates user_data in place (segment /
    segment_asked) — caller just needs to save_user_data afterward.

    Priority: candidate directory signal > previously stored answer > ask once
    (parsing the next message as the answer) > give up, stay unsegmented."""
    directory_segment = DIRECTORY.segment_of(candidate) if candidate else None
    if directory_segment:
        user_data["segment"] = directory_segment
        return directory_segment, False

    stored = user_data.get("segment")
    if stored:
        return stored, False

    if user_data.get("segment_asked"):
        parsed = parse_segment_reply(incoming_message)
        if parsed:
            user_data["segment"] = parsed
        return parsed, False

    user_data["segment_asked"] = True
    return None, True


def get_user_data(phone: str) -> dict:
    user_file = DATA_DIR / f"{phone}.json"
    if user_file.exists():
        with open(user_file) as f:
            return json.load(f)
    return {"phone": phone, "email": None, "history": []}


def save_user_data(phone: str, data: dict):
    with open(DATA_DIR / f"{phone}.json", "w") as f:
        json.dump(data, f, indent=2)


DISCLAIMER_REPEAT_AFTER = timedelta(hours=24)


def should_show_disclaimer(user_data: dict) -> bool:
    """True on the very first message ever, or if it's been 24h+ since the
    last one — reminds a returning user this isn't an official/compliance
    channel, same as they'd have seen when they first started chatting."""
    if not user_data["history"]:
        return True
    last_at = user_data.get("last_message_at")
    if not last_at:
        return True  # no timestamp on record (e.g. an older session) — err on showing it
    try:
        elapsed = datetime.now(timezone.utc) - datetime.fromisoformat(last_at)
        return elapsed > DISCLAIMER_REPEAT_AFTER
    except Exception:
        return True


def touch_last_message(user_data: dict):
    user_data["last_message_at"] = datetime.now(timezone.utc).isoformat()


# --- WhatsApp provider selection --------------------------------------------
# "twilio" (default) or "meta" — switches both the /whatsapp webhook format
# and the outbound send path below. Only one is active per deployment; the
# other's code stays dormant (same "unless configured" pattern as before).
WHATSAPP_PROVIDER = os.environ.get("WHATSAPP_PROVIDER", "twilio").strip().lower()

# --- Media (document upload) handling, WhatsApp side -----------------------
TWILIO_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")

# --- Outbound WhatsApp (HR-initiated welcome/update messages) --------------
# Dormant unless all three are set. TWILIO_WELCOME_TEMPLATE_SID is a WhatsApp
# message template approved via Twilio/Meta — required to message someone
# who hasn't messaged the bot in the last 24h (WhatsApp's session-window
# rule; not something this code can work around). Free-form sends only work
# for people inside that 24h window, template or not.
TWILIO_WHATSAPP_FROM = os.environ.get("TWILIO_WHATSAPP_NUMBER") or os.environ.get("TWILIO_PHONE_NUMBER", "")
TWILIO_WELCOME_TEMPLATE_SID = os.environ.get("TWILIO_WELCOME_TEMPLATE_SID", "")

_twilio_client = None


def get_twilio_client():
    global _twilio_client
    if _twilio_client is None and TWILIO_SID and TWILIO_TOKEN:
        _twilio_client = TwilioClient(TWILIO_SID, TWILIO_TOKEN)
    return _twilio_client


def send_whatsapp_template_twilio(to_phone: str, content_variables: dict | None = None) -> tuple[bool, str]:
    """Send the approved welcome template — works even if the recipient has
    never messaged the bot before (exempt from the 24h window)."""
    client = get_twilio_client()
    if not client:
        return False, "Twilio not configured (TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN missing)"
    if not TWILIO_WELCOME_TEMPLATE_SID:
        return False, "No welcome template configured (set TWILIO_WELCOME_TEMPLATE_SID once one is approved)"
    if not TWILIO_WHATSAPP_FROM:
        return False, "No Twilio WhatsApp sender number configured (TWILIO_WHATSAPP_NUMBER/TWILIO_PHONE_NUMBER)"
    try:
        msg = client.messages.create(
            from_=f"whatsapp:{TWILIO_WHATSAPP_FROM}",
            to=f"whatsapp:{to_phone}",
            content_sid=TWILIO_WELCOME_TEMPLATE_SID,
            content_variables=json.dumps(content_variables or {}),
        )
        return True, msg.sid
    except Exception as e:
        return False, str(e)


def send_whatsapp_freeform_twilio(to_phone: str, body: str) -> tuple[bool, str]:
    """Send a plain message — only deliverable if the recipient messaged the
    bot within the last 24h; Twilio will reject it otherwise."""
    client = get_twilio_client()
    if not client:
        return False, "Twilio not configured (TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN missing)"
    if not TWILIO_WHATSAPP_FROM:
        return False, "No Twilio WhatsApp sender number configured (TWILIO_WHATSAPP_NUMBER/TWILIO_PHONE_NUMBER)"
    try:
        msg = client.messages.create(
            from_=f"whatsapp:{TWILIO_WHATSAPP_FROM}",
            to=f"whatsapp:{to_phone}",
            body=body,
        )
        return True, msg.sid
    except Exception as e:
        return False, str(e)


def parse_media_twilio(form) -> list[dict]:
    """Extract media items from a Twilio webhook form."""
    try:
        n = int(form.get("NumMedia", "0") or "0")
    except ValueError:
        n = 0
    items = []
    for i in range(n):
        url = form.get(f"MediaUrl{i}")
        if url:
            items.append({"url": url, "content_type": form.get(f"MediaContentType{i}", "")})
    return items


def save_media_twilio(phone: str, media: list[dict]) -> list[dict]:
    """Best-effort download of uploaded files (Twilio media needs basic auth)."""
    safe = re.sub(r"[^0-9]", "", phone) or "unknown"
    outdir = UPLOAD_MEDIA_DIR / safe
    outdir.mkdir(parents=True, exist_ok=True)
    auth = (TWILIO_SID, TWILIO_TOKEN) if TWILIO_SID and TWILIO_TOKEN else None
    saved = []
    for i, m in enumerate(media):
        rec = {"url": m["url"], "content_type": m["content_type"], "file": None,
               "received_at": int(time.time())}
        try:
            r = requests.get(m["url"], auth=auth, timeout=30)
            r.raise_for_status()
            ext = mimetypes.guess_extension((m["content_type"] or "").split(";")[0]) or ""
            fname = f"{rec['received_at']}_{i}{ext}"
            (outdir / fname).write_bytes(r.content)
            rec["file"] = str(outdir / fname)
            logger.info(f"📎 Saved upload from {phone}: {fname} ({m['content_type']})")
        except Exception as e:
            logger.warning(f"Could not download media from {phone}: {e}")
        saved.append(rec)
    return saved


# --- Meta WhatsApp Cloud API (direct, no Twilio/WATI in between) -----------
# Active only when WHATSAPP_PROVIDER=meta. META_ACCESS_TOKEN should be a
# permanent System User token (not the 24h test token from the App
# dashboard's Quick Start) — that one expires and will silently break
# sending. META_APP_SECRET enables verifying that inbound webhooks actually
# came from Meta (X-Hub-Signature-256); without it verification is skipped
# (fine for local dev, not for production). META_VERIFY_TOKEN is a string
# you invent yourself and enter in the App dashboard's webhook setup — Meta
# echoes it back on the one-time GET verification handshake.
META_GRAPH_VERSION = os.environ.get("META_GRAPH_API_VERSION", "v21.0")
META_ACCESS_TOKEN = os.environ.get("META_ACCESS_TOKEN", "")
META_PHONE_NUMBER_ID = os.environ.get("META_PHONE_NUMBER_ID", "")
META_APP_SECRET = os.environ.get("META_APP_SECRET", "")
META_VERIFY_TOKEN = os.environ.get("META_VERIFY_TOKEN", "")
META_WELCOME_TEMPLATE_NAME = os.environ.get("META_WELCOME_TEMPLATE_NAME", "")
META_WELCOME_TEMPLATE_LANG = os.environ.get("META_WELCOME_TEMPLATE_LANG", "en_US")
META_GRAPH_BASE = f"https://graph.facebook.com/{META_GRAPH_VERSION}"


def _meta_headers() -> dict:
    return {"Authorization": f"Bearer {META_ACCESS_TOKEN}", "Content-Type": "application/json"}


def _meta_error_detail(e: Exception) -> str:
    resp = getattr(e, "response", None)
    return resp.text if resp is not None else str(e)


def verify_meta_signature(raw_body: bytes, signature_header: str | None) -> bool:
    """Verify X-Hub-Signature-256 against META_APP_SECRET. Skipped (returns
    True) when no app secret is configured — dev-only, tighten before prod."""
    if not META_APP_SECRET:
        return True
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(META_APP_SECRET.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header.split("=", 1)[1])


def parse_incoming_meta(payload: dict) -> tuple[str, str, list[dict]] | None:
    """Extract (phone, text, media_items) from a Cloud API webhook payload.
    Returns None for events with nothing to react to — delivery/read status
    callbacks arrive on the same webhook and carry no "messages" key."""
    try:
        value = payload["entry"][0]["changes"][0]["value"]
    except (KeyError, IndexError, TypeError):
        return None
    messages = value.get("messages")
    if not messages:
        return None
    msg = messages[0]
    phone = "+" + re.sub(r"\D", "", msg.get("from", ""))
    msg_type = msg.get("type", "text")
    media = []
    if msg_type == "text":
        text = (msg.get("text", {}).get("body") or "").strip()
    else:
        sub = msg.get(msg_type, {}) or {}
        text = (sub.get("caption") or "").strip()
        if sub.get("id"):
            media.append({"id": sub["id"], "content_type": sub.get("mime_type", "")})
    return phone, text, media


def save_media_meta(phone: str, media: list[dict]) -> list[dict]:
    """Resolve each Meta media ID to a short-lived URL, then download it
    (Bearer auth, unlike Twilio's basic auth)."""
    safe = re.sub(r"[^0-9]", "", phone) or "unknown"
    outdir = UPLOAD_MEDIA_DIR / safe
    outdir.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, m in enumerate(media):
        rec = {"url": None, "content_type": m.get("content_type", ""), "file": None,
               "received_at": int(time.time())}
        try:
            lookup = requests.get(f"{META_GRAPH_BASE}/{m['id']}", headers=_meta_headers(), timeout=15)
            lookup.raise_for_status()
            media_url = lookup.json()["url"]
            rec["url"] = media_url
            r = requests.get(media_url, headers={"Authorization": f"Bearer {META_ACCESS_TOKEN}"}, timeout=30)
            r.raise_for_status()
            ext = mimetypes.guess_extension((rec["content_type"] or "").split(";")[0]) or ""
            fname = f"{rec['received_at']}_{i}{ext}"
            (outdir / fname).write_bytes(r.content)
            rec["file"] = str(outdir / fname)
            logger.info(f"📎 Saved upload from {phone}: {fname} ({rec['content_type']})")
        except Exception as e:
            logger.warning(f"Could not download Meta media from {phone}: {e}")
        saved.append(rec)
    return saved


def send_meta_text(to_phone: str, body: str) -> tuple[bool, str]:
    """Send a plain message via the Graph API — only deliverable within 24h
    of the recipient's last message; Meta will reject it otherwise."""
    if not META_ACCESS_TOKEN or not META_PHONE_NUMBER_ID:
        return False, "Meta Cloud API not configured (META_ACCESS_TOKEN/META_PHONE_NUMBER_ID missing)"
    try:
        r = requests.post(
            f"{META_GRAPH_BASE}/{META_PHONE_NUMBER_ID}/messages",
            headers=_meta_headers(),
            json={
                "messaging_product": "whatsapp",
                "to": re.sub(r"\D", "", to_phone),
                "type": "text",
                "text": {"body": body},
            },
            timeout=30,
        )
        r.raise_for_status()
        return True, r.json().get("messages", [{}])[0].get("id", "")
    except Exception as e:
        return False, _meta_error_detail(e)


def send_meta_template(to_phone: str, content_variables: dict | None = None) -> tuple[bool, str]:
    """Send the approved welcome template — works even if the recipient has
    never messaged the bot before (exempt from the 24h window). Variables
    are sent as positional body params ({{1}}, {{2}}...), ordered by key —
    match this to however the approved template numbers its placeholders."""
    if not META_ACCESS_TOKEN or not META_PHONE_NUMBER_ID:
        return False, "Meta Cloud API not configured (META_ACCESS_TOKEN/META_PHONE_NUMBER_ID missing)"
    if not META_WELCOME_TEMPLATE_NAME:
        return False, "No welcome template configured (set META_WELCOME_TEMPLATE_NAME once one is approved)"
    components = []
    if content_variables:
        ordered = [content_variables[k] for k in sorted(content_variables, key=str)]
        components.append({
            "type": "body",
            "parameters": [{"type": "text", "text": str(v)} for v in ordered],
        })
    try:
        r = requests.post(
            f"{META_GRAPH_BASE}/{META_PHONE_NUMBER_ID}/messages",
            headers=_meta_headers(),
            json={
                "messaging_product": "whatsapp",
                "to": re.sub(r"\D", "", to_phone),
                "type": "template",
                "template": {
                    "name": META_WELCOME_TEMPLATE_NAME,
                    "language": {"code": META_WELCOME_TEMPLATE_LANG},
                    "components": components,
                },
            },
            timeout=30,
        )
        r.raise_for_status()
        return True, r.json().get("messages", [{}])[0].get("id", "")
    except Exception as e:
        return False, _meta_error_detail(e)


# --- Provider-agnostic entry points -----------------------------------------
# Everything outside this block (outreach endpoints, /health) calls these —
# they dispatch to whichever provider WHATSAPP_PROVIDER selects.

def send_whatsapp_template(to_phone: str, content_variables: dict | None = None) -> tuple[bool, str]:
    if WHATSAPP_PROVIDER == "meta":
        return send_meta_template(to_phone, content_variables)
    return send_whatsapp_template_twilio(to_phone, content_variables)


def send_whatsapp_freeform(to_phone: str, body: str) -> tuple[bool, str]:
    if WHATSAPP_PROVIDER == "meta":
        return send_meta_text(to_phone, body)
    return send_whatsapp_freeform_twilio(to_phone, body)


def _welcome_template_configured() -> bool:
    if WHATSAPP_PROVIDER == "meta":
        return bool(META_WELCOME_TEMPLATE_NAME)
    return bool(TWILIO_WELCOME_TEMPLATE_SID)


def generate_reply(user_data: dict, kb_context: str, profile_block: str | None = None, segment: str | None = None) -> tuple[str, bool]:
    """Call DeepSeek with retrieved KB context + candidate profile + recent
    history. Returns (reply_text, unanswered) — unanswered is True only when
    the model tagged this as a genuine knowledge-base gap (see UNANSWERED_MARKER)."""
    messages = [{"role": "system", "content": build_system_prompt(kb_context, profile_block, segment)}]
    messages.extend(user_data["history"][-MAX_HISTORY:])
    try:
        resp = requests.post(
            DEEPSEEK_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            },
            json={
                "model": DEEPSEEK_MODEL,
                "messages": messages,
                "max_tokens": 600,
                "temperature": 0.1,  # low → faithful to KB; 0.3 produced live inconsistency —
                # same query, empty history, identical retrieved context still swung between a
                # correct direct answer, an unnecessary clarifying question, and (once) a
                # completely unrelated tangent about IT-vs-travel "tickets"
            },
            # 30s (calibrated for the old 350-token cap) was timing out live
            # once max_tokens went to 600 — longer generations take longer.
            timeout=60,
        )
        resp.raise_for_status()
        choice = resp.json()["choices"][0]
        text = choice["message"]["content"].strip()
        if choice.get("finish_reason") == "length":
            # Hit the token cap mid-sentence (a legitimately detailed answer,
            # e.g. a multi-part benefits breakdown) — trim back to the last
            # complete sentence rather than show a dangling, sometimes
            # markdown-breaking fragment like "...reach out to **care".
            logger.warning("LLM reply truncated at max_tokens — trimming to last complete sentence")
            sentences = re.split(r"(?<=[.!?])\s+", text)
            if len(sentences) > 1:
                text = " ".join(sentences[:-1]).strip()
        unanswered = text.startswith(UNANSWERED_MARKER)
        if unanswered:
            text = text[len(UNANSWERED_MARKER):].lstrip()
        return text, unanswered
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return (
            "Sorry, I glitched for a second there — could you send that again? "
            "I'm here to help with your Recykal onboarding."
        ), False


def whatsapp_reply(phone: str, text: str) -> Response:
    """Deliver a reply and return the HTTP response the webhook itself
    should send. Twilio expects the reply inline as TwiML; Meta's Cloud API
    has no such mechanism — the webhook response must just be a bare 200,
    and the reply is a separate authenticated call to the Graph API."""
    if WHATSAPP_PROVIDER == "meta":
        ok, detail = send_meta_text(phone, text)
        if not ok:
            logger.error(f"Meta send failed to {phone}: {detail}")
        return Response(content="", status_code=200)
    resp = MessagingResponse()
    resp.message(text)
    return Response(content=str(resp), media_type="application/xml")


@app.get("/whatsapp")
async def whatsapp_webhook_get(request: Request):
    if WHATSAPP_PROVIDER == "meta":
        params = request.query_params
        if (
            META_VERIFY_TOKEN
            and params.get("hub.mode") == "subscribe"
            and params.get("hub.verify_token") == META_VERIFY_TOKEN
        ):
            return Response(content=params.get("hub.challenge", ""), status_code=200)
        return Response(content="", status_code=403)
    return Response(content="", status_code=200)


@app.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    phone = None
    try:
        if WHATSAPP_PROVIDER == "meta":
            raw_body = await request.body()
            if not verify_meta_signature(raw_body, request.headers.get("X-Hub-Signature-256")):
                logger.warning("Rejected /whatsapp webhook: bad Meta signature")
                return Response(content="", status_code=403)
            parsed = parse_incoming_meta(json.loads(raw_body))
            if parsed is None:
                # Delivery/read status callback, or a payload with nothing to
                # react to — acknowledge and move on, there's no reply to send.
                return Response(content="", status_code=200)
            phone, incoming_message, media = parsed
        else:
            form_data = await request.form()
            phone = form_data.get("From", "").replace("whatsapp:", "")
            incoming_message = (form_data.get("Body", "") or "").strip()
            media = parse_media_twilio(form_data)

        logger.info(f"📨 [WEBHOOK] From: {phone} | Message: {incoming_message} | media: {len(media)}")

        sync_knowledge_from_drive()

        user_data = get_user_data(phone)
        show_disclaimer = should_show_disclaimer(user_data)

        # Match the caller against the onboarding log (if configured).
        candidate = DIRECTORY.lookup(phone)
        profile_block = DIRECTORY.profile_block(candidate) if candidate else None
        first_name = None
        if candidate:
            full = DIRECTORY.name_of(candidate)
            first_name = full.split()[0] if full else None

        segment, should_ask_segment = resolve_segment(candidate, user_data, incoming_message)

        # Genuine empty ping (no text AND no attachment) → greeting.
        if not incoming_message and not media:
            hello = f"Hey {first_name}! 👋" if first_name else "Hey! 👋"
            reply = f"{hello} I'm Maya from Recykal's People & Culture team. How can I help with your onboarding today?"
            if should_ask_segment:
                reply += SEGMENT_QUESTION
            if show_disclaimer:
                reply += FIRST_MESSAGE_DISCLAIMER
            user_data["history"].append({"role": "assistant", "content": reply})
            touch_last_message(user_data)
            save_user_data(phone, user_data)
            return whatsapp_reply(phone, reply)

        # opportunistic email capture
        if incoming_message and not user_data.get("email"):
            m = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", incoming_message)
            if m:
                user_data["email"] = m.group(0)

        # Handle uploaded documents: save them and build a turn the LLM can act on.
        if media:
            saved = save_media_meta(phone, media) if WHATSAPP_PROVIDER == "meta" else save_media_twilio(phone, media)
            user_data.setdefault("documents", []).extend(saved)
            types = ", ".join(sorted({(s["content_type"] or "file").split("/")[0] for s in saved}))
            caption = incoming_message or ""
            user_turn = (
                (caption + "\n\n" if caption else "")
                + f"[The new hire just uploaded {len(saved)} document(s) "
                f"({types}) through WhatsApp. Warmly confirm you've received the "
                f"document(s) and that the People & Culture team will review them. "
                f"Do NOT greet from scratch or ask them to re-send. If appropriate, "
                f"mention any remaining onboarding steps from the knowledge base.]"
            )
            retrieval_message = f"{caption} document upload onboarding verification".strip()
            kb_context = retrieve_context(retrieval_message, segment)
        else:
            user_turn = incoming_message
            blended = f"{recent_user_context(user_data['history'])} {incoming_message}".strip()
            kb_context = retrieve_context(incoming_message, segment, extra_query=blended)
        user_data["history"].append({"role": "user", "content": user_turn})
        reply, unanswered = generate_reply(user_data, kb_context, profile_block, segment)
        if not media:
            # Document-upload turns are synthetic, not a real question — skip logging those.
            upload_manager.log_question(phone, "whatsapp", incoming_message, unanswered, segment)
        if should_ask_segment:
            reply += SEGMENT_QUESTION
        if show_disclaimer:
            reply += FIRST_MESSAGE_DISCLAIMER
        user_data["history"].append({"role": "assistant", "content": reply})
        touch_last_message(user_data)
        save_user_data(phone, user_data)

        return whatsapp_reply(phone, reply)

    except Exception as e:
        logger.error(f"ERROR in webhook: {e}")
        apology = "Sorry, something hiccuped on my end. Mind sending that again?"
        if WHATSAPP_PROVIDER == "meta":
            if phone:
                try:
                    send_meta_text(phone, apology)
                except Exception:
                    pass
            return Response(content="", status_code=200)
        resp = MessagingResponse()
        resp.message(apology)
        return Response(content=str(resp), media_type="application/xml")


# --- Web chat (browser testing of the same bot, no WhatsApp needed) --------
# Session keyed by "web-<username>" — same get_user_data/save_user_data
# storage as WhatsApp, just namespaced so it never collides with real phone
# numbers. Requires login (reuses the upload-interface's auth).

@app.post("/chat")
async def chat(request: Request, current_user: dict = Depends(get_current_user)):
    """Web chat endpoint — same grounded RAG+LLM pipeline as /whatsapp, for
    trying out the bot in a browser instead of over WhatsApp."""
    try:
        data = await request.json()
        message = (data.get("message") or "").strip()
        if not message:
            return JSONResponse({"error": "Message required"}, status_code=400)

        session_key = f"web-{current_user['username']}"
        user_data = get_user_data(session_key)
        show_disclaimer = should_show_disclaimer(user_data)

        # Optional explicit override so HR can preview either segment's
        # experience directly, instead of going through the ask-once flow.
        override = data.get("segment")
        if override in ("pre_join", "post_join"):
            user_data["segment"] = override
            user_data["segment_asked"] = True
            segment, should_ask_segment = override, False
        elif override == "both":
            user_data["segment"] = None
            user_data["segment_asked"] = True
            segment, should_ask_segment = None, False
        else:
            segment, should_ask_segment = resolve_segment(None, user_data, message)

        blended = f"{recent_user_context(user_data['history'])} {message}".strip()
        kb_context = retrieve_context(message, segment, extra_query=blended)

        user_data["history"].append({"role": "user", "content": message})
        reply, unanswered = generate_reply(user_data, kb_context, profile_block=None, segment=segment)
        upload_manager.log_question(session_key, "web", message, unanswered, segment)
        if should_ask_segment:
            reply += SEGMENT_QUESTION
        if show_disclaimer:
            reply += FIRST_MESSAGE_DISCLAIMER
        user_data["history"].append({"role": "assistant", "content": reply})
        touch_last_message(user_data)
        save_user_data(session_key, user_data)

        return JSONResponse({"reply": reply, "segment": segment})

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/chat/reset")
async def chat_reset(current_user: dict = Depends(get_current_user)):
    """Clear the web chat session's history for the logged-in user"""
    session_key = f"web-{current_user['username']}"
    save_user_data(session_key, {"phone": session_key, "email": None, "history": []})
    return JSONResponse({"success": True})


# ============================================================================
# FILE UPLOAD / HR WEB INTERFACE
# ============================================================================

upload_manager = FileUploadManager(
    upload_dir=f"{PROJECT_DIR}/uploads",
    db_path=f"{PROJECT_DIR}/chatbot.db"
)
file_processor = UploadedFileProcessor(knowledge_dir=PROJECT_DIR)
init_auth(upload_manager.get_user_by_username)

# Text formats we can merge directly into the knowledge base. PDFs/DOCX are
# still saved and tracked as general uploads, just not merged (no extraction
# library wired up).
KNOWLEDGE_MERGEABLE_EXTENSIONS = {'.md', '.txt'}


def rebuild_and_reload_knowledge():
    """Regenerate all three knowledge files from knowledge_base.md + active
    HR-uploaded documents, filtered by audience tag, then reload them into
    their vector stores. knowledge.md (segment unknown) only gets 'both'-tagged
    docs; the pre/post-join files get their tag plus every 'both'-tagged doc."""
    all_active = upload_manager.list_knowledge_documents(active_only=True)
    common_docs = [d for d in all_active if d.get('audience', 'both') == 'both']
    pre_join_docs = [d for d in all_active if d.get('audience') in ('pre_join', 'both')]
    post_join_docs = [d for d in all_active if d.get('audience') in ('post_join', 'both')]

    results = [
        file_processor.rebuild_knowledge_md(common_docs, output_file=KNOWLEDGE_FILE),
        file_processor.rebuild_knowledge_md(pre_join_docs, output_file=KNOWLEDGE_FILE_PRE_JOIN),
        file_processor.rebuild_knowledge_md(post_join_docs, output_file=KNOWLEDGE_FILE_POST_JOIN),
    ]
    if all(ok for ok, _ in results):
        reload_knowledge_base()
    else:
        logger.error(f"Could not rebuild knowledge files: {[msg for ok, msg in results if not ok]}")


@app.get("/upload-interface")
async def get_upload_interface():
    """Serve the upload interface HTML"""
    interface_path = APP_DIR / "upload_interface.html"
    if interface_path.exists():
        return FileResponse(interface_path, media_type="text/html")
    else:
        return JSONResponse(
            {"error": "Upload interface not found"},
            status_code=404
        )


@app.post("/register")
async def register(
    fullname: str = Form(...),
    email: str = Form(...),
    username: str = Form(...),
    password: str = Form(...)
):
    """Register a new user"""
    try:
        success, message = upload_manager.register_user(
            username=username,
            email=email,
            password=password,
            fullname=fullname
        )

        if success:
            return JSONResponse({"success": True, "message": message})
        else:
            return JSONResponse(
                {"success": False, "error": message},
                status_code=400
            )

    except Exception as e:
        logger.error(f"Registration error: {e}")
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


@app.post("/login")
async def login(
    username: str = Form(...),
    password: str = Form(...)
):
    """Authenticate user"""
    try:
        success, message = upload_manager.authenticate_user(
            username=username,
            password=password
        )

        if success:
            token = create_token(username)
            user = upload_manager.get_user_by_username(username)
            return JSONResponse({
                "success": True,
                "message": message,
                "username": username,
                "role": user["role"] if user else "admin",
                "access_token": token,
                "token_type": "bearer"
            })
        else:
            return JSONResponse(
                {"success": False, "error": message},
                status_code=401
            )

    except Exception as e:
        logger.error(f"Login error: {e}")
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


@app.post("/upload")
async def upload_files(
    request: Request,
    audience: str = Form('both'),
    current_user: dict = Depends(require_admin)
):
    """Upload files to the chatbot. audience: 'pre_join'/'post_join'/'both' —
    which segment(s) of users these files' content should be visible to."""
    username = current_user["username"]
    if audience not in ('pre_join', 'post_join', 'both'):
        audience = 'both'
    try:
        form = await request.form()

        if 'files' not in form:
            return JSONResponse(
                {"error": "No files provided"},
                status_code=400
            )

        files = form.getlist('files')
        uploaded_count = 0
        errors = []
        notes = []
        added_to_knowledge = False

        for file in files:
            if not file.filename:
                continue

            content = await file.read()

            valid, msg = upload_manager.validate_file(file.filename, len(content))
            if not valid:
                errors.append(f"{file.filename}: {msg}")
                continue

            success, msg, file_path = upload_manager.save_file(
                username=username,
                file_content=content,
                filename=file.filename
            )

            if success:
                uploaded_count += 1
                logger.info(f"File uploaded: {file.filename} by {username}")

                ext = Path(file.filename).suffix.lower()
                if ext in KNOWLEDGE_MERGEABLE_EXTENSIONS:
                    try:
                        text = content.decode('utf-8')
                        ok, kmsg, _ = upload_manager.add_knowledge_document(
                            title=file.filename, content=text, uploaded_by=username, audience=audience
                        )
                        if ok:
                            added_to_knowledge = True
                        else:
                            notes.append(f"{file.filename}: saved, but not added to knowledge base ({kmsg})")
                    except UnicodeDecodeError:
                        notes.append(f"{file.filename}: saved, but not added to knowledge base (not valid text)")
                else:
                    notes.append(f"{file.filename}: saved, but not added to knowledge base ({ext} text extraction not supported)")
            else:
                errors.append(f"{file.filename}: {msg}")

        if added_to_knowledge:
            rebuild_and_reload_knowledge()

        response = {
            "success": uploaded_count > 0,
            "uploaded": uploaded_count,
            "total": len(files),
            "message": f"Successfully uploaded {uploaded_count} file(s)"
        }

        if errors:
            response["errors"] = errors
        if notes:
            response["notes"] = notes

        return JSONResponse(response)

    except Exception as e:
        logger.error(f"Upload error: {e}")
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


@app.post("/upload/text")
async def upload_text(request: Request, current_user: dict = Depends(require_admin)):
    """Upload text content as markdown"""
    try:
        data = await request.json()

        username = current_user["username"]
        title = data.get('title')
        content = data.get('content')
        audience = data.get('audience') if data.get('audience') in ('pre_join', 'post_join', 'both') else 'both'

        if not all([title, content]):
            return JSONResponse(
                {"error": "Missing required fields"},
                status_code=400
            )

        success, message = file_processor.process_text(
            filename=title,
            content=content,
            username=username
        )

        if success:
            ok, kmsg, _ = upload_manager.add_knowledge_document(
                title=title, content=content, uploaded_by=username, audience=audience
            )
            if ok:
                rebuild_and_reload_knowledge()
            else:
                logger.error(f"Could not add knowledge document from text upload: {kmsg}")

            return JSONResponse({
                "success": True,
                "message": message
            })
        else:
            return JSONResponse(
                {"success": False, "error": message},
                status_code=400
            )

    except Exception as e:
        logger.error(f"Text upload error: {e}")
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


@app.get("/upload/history")
async def get_upload_history(current_user: dict = Depends(require_admin)):
    """Get upload history for the logged-in user"""
    try:
        history = upload_manager.get_upload_history(current_user["username"])

        return JSONResponse({
            "success": True,
            "files": history
        })

    except Exception as e:
        logger.error(f"History error: {e}")
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


@app.get("/knowledge/documents")
async def list_knowledge_documents(current_user: dict = Depends(require_admin)):
    """List every document contributing to (or retired from) the bot's knowledge base"""
    try:
        docs = upload_manager.list_knowledge_documents()
        for d in docs:
            d.pop('content', None)  # keep the listing light; full text not needed here
        return JSONResponse({"success": True, "documents": docs})
    except Exception as e:
        logger.error(f"List knowledge documents error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/knowledge/documents/{doc_id}/toggle")
async def toggle_knowledge_document(doc_id: int, active: bool = Form(...), current_user: dict = Depends(require_admin)):
    """Activate or deactivate a knowledge document without deleting it"""
    try:
        success, message = upload_manager.set_knowledge_document_active(doc_id, active)
        if not success:
            return JSONResponse({"success": False, "error": message}, status_code=404)
        rebuild_and_reload_knowledge()
        return JSONResponse({"success": True, "active": active})
    except Exception as e:
        logger.error(f"Toggle knowledge document error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/knowledge/documents/{doc_id}/audience")
async def set_knowledge_document_audience(doc_id: int, audience: str = Form(...), current_user: dict = Depends(require_admin)):
    """Re-tag which segment(s) a knowledge document is visible to"""
    try:
        success, message = upload_manager.set_knowledge_document_audience(doc_id, audience)
        if not success:
            return JSONResponse({"success": False, "error": message}, status_code=400)
        rebuild_and_reload_knowledge()
        return JSONResponse({"success": True, "audience": audience})
    except Exception as e:
        logger.error(f"Set knowledge document audience error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.delete("/knowledge/documents/{doc_id}")
async def delete_knowledge_document(doc_id: int, current_user: dict = Depends(require_admin)):
    """Permanently remove a knowledge document"""
    try:
        success, message = upload_manager.delete_knowledge_document(doc_id)
        if not success:
            return JSONResponse({"success": False, "error": message}, status_code=404)
        rebuild_and_reload_knowledge()
        return JSONResponse({"success": True})
    except Exception as e:
        logger.error(f"Delete knowledge document error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================================
# OUTREACH (HR-initiated welcome/update messages)
# ============================================================================

@app.get("/candidates")
async def list_candidates(current_user: dict = Depends(require_admin)):
    """List candidates from the onboarding log (if one is configured), each
    flagged with whether they've ever messaged the bot — a rough signal for
    whether a free-form update would actually be deliverable (WhatsApp only
    allows free-form sends within 24h of the recipient's last message)."""
    try:
        if not DIRECTORY.ready:
            return JSONResponse({
                "success": True,
                "configured": False,
                "welcome_template_configured": _welcome_template_configured(),
                "candidates": [],
            })

        messaged_phones = {normalize_phone(f.stem) for f in DATA_DIR.glob("*.json")}

        candidates = []
        for entry in DIRECTORY.list_all():
            row = entry["row"]
            candidates.append({
                "phone_normalized": entry["phone_normalized"],
                "phone_raw": entry["phone_raw"],
                "name": DIRECTORY.name_of(row),
                "segment": DIRECTORY.segment_of(row),
                "has_messaged": entry["phone_normalized"] in messaged_phones,
            })

        return JSONResponse({
            "success": True,
            "configured": True,
            "welcome_template_configured": _welcome_template_configured(),
            "candidates": candidates,
        })
    except Exception as e:
        logger.error(f"List candidates error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/candidates/send-welcome")
async def send_welcome_messages(request: Request, current_user: dict = Depends(require_admin)):
    """Bulk-send the approved welcome template. Works regardless of the 24h
    window — that's the point of using a template. content_variables sends
    {"1": first_name} to match the template's first placeholder; adjust to
    match however the actual approved template numbers its variables."""
    try:
        data = await request.json()
        phones = data.get("phones") or []
        if not phones:
            return JSONResponse({"error": "No phone numbers provided"}, status_code=400)

        results = []
        for phone in phones:
            candidate = DIRECTORY.lookup(phone)
            name = DIRECTORY.name_of(candidate) if candidate else None
            first_name = name.split()[0] if name else "there"
            ok, detail = send_whatsapp_template(phone, {"1": first_name})
            results.append({"phone": phone, "success": ok, "detail": detail})
            logger.info(f"Welcome send to {phone} by {current_user['username']}: {ok} ({detail})")

        return JSONResponse({"success": True, "results": results})
    except Exception as e:
        logger.error(f"Send welcome error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/candidates/send-update")
async def send_update_messages(request: Request, current_user: dict = Depends(require_admin)):
    """Bulk-send a free-form update. Only actually reaches people who
    messaged the bot within the last 24h — WhatsApp rejects free-form sends
    outside that window regardless of what this code does."""
    try:
        data = await request.json()
        phones = data.get("phones") or []
        message = (data.get("message") or "").strip()
        if not phones or not message:
            return JSONResponse({"error": "phones and message are required"}, status_code=400)

        results = []
        for phone in phones:
            ok, detail = send_whatsapp_freeform(phone, message)
            results.append({"phone": phone, "success": ok, "detail": detail})
            logger.info(f"Update send to {phone} by {current_user['username']}: {ok} ({detail})")

        return JSONResponse({"success": True, "results": results})
    except Exception as e:
        logger.error(f"Send update error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================================
# USER MANAGEMENT (admin-only — chat-only 'user' accounts for real employees/
# candidates trying the bot, plus visibility into what they're asking it)
# ============================================================================

@app.post("/users")
async def create_user(request: Request, current_user: dict = Depends(require_admin)):
    """Create a chat-only 'user' account. Share the username/password with
    them out of band (email, WhatsApp, etc.) — there's no self-service
    signup for this role, an admin creates it deliberately."""
    try:
        data = await request.json()
        username = (data.get("username") or "").strip()
        password = data.get("password") or ""
        fullname = (data.get("fullname") or "").strip()
        email = (data.get("email") or f"{username}@users.local").strip()

        if not username or not password:
            return JSONResponse({"error": "username and password are required"}, status_code=400)

        success, message = upload_manager.register_user(
            username=username, email=email, password=password, fullname=fullname, role='user'
        )
        if success:
            return JSONResponse({"success": True, "message": message})
        return JSONResponse({"success": False, "error": message}, status_code=400)
    except Exception as e:
        logger.error(f"Create user error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/users")
async def list_regular_users(current_user: dict = Depends(require_admin)):
    """List chat-only 'user' accounts"""
    try:
        users = upload_manager.list_users(role='user')
        return JSONResponse({"success": True, "users": users})
    except Exception as e:
        logger.error(f"List users error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.delete("/users/{username}")
async def revoke_user(username: str, current_user: dict = Depends(require_admin)):
    """Revoke a user account's access"""
    try:
        success, message = upload_manager.delete_user_by_username(username)
        if not success:
            return JSONResponse({"success": False, "error": message}, status_code=404)
        return JSONResponse({"success": True})
    except Exception as e:
        logger.error(f"Revoke user error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/users/{username}/chat")
async def get_user_chat_transcript(username: str, current_user: dict = Depends(require_admin)):
    """Admin-only: read a user's web-chat conversation history — the same
    session storage /chat itself reads and writes, just for a given username
    instead of the currently-authenticated one."""
    try:
        data = get_user_data(f"web-{username}")
        return JSONResponse({"success": True, "history": data.get("history", [])})
    except Exception as e:
        logger.error(f"Get user chat transcript error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================================
# STATUS / HEALTH / ROOT
# ============================================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "Recykal Onboarding Agent",
        "timestamp": datetime.now().isoformat(),
        "knowledge_loaded": bool(KNOWLEDGE_BASE),
        "knowledge_chars": len(KNOWLEDGE_BASE),
        "vector_memory": VSTORE.ready,
        "vector_chunks": len(VSTORE.chunks),
        "vector_chunks_pre_join": len(VSTORE_PRE_JOIN.chunks),
        "vector_chunks_post_join": len(VSTORE_POST_JOIN.chunks),
        "retrieval_top_k": TOP_K,
        "candidate_directory": DIRECTORY.ready,
        "candidate_rows": len(DIRECTORY._index),
        "directory_source": DIRECTORY.source,
        "outreach": {
            "provider": WHATSAPP_PROVIDER,
            "twilio_configured": bool(get_twilio_client()),
            "meta_configured": bool(META_ACCESS_TOKEN and META_PHONE_NUMBER_ID),
            "meta_signature_verification": bool(META_APP_SECRET),
            "welcome_template_configured": _welcome_template_configured(),
            "sender_configured": bool(TWILIO_WHATSAPP_FROM) if WHATSAPP_PROVIDER != "meta" else bool(META_PHONE_NUMBER_ID),
        },
        "google_drive_sync": DRIVE_SYNC.get_sync_status(),
        "upload_dir_exists": Path(f"{PROJECT_DIR}/uploads").exists(),
        "db_initialized": Path(f"{PROJECT_DIR}/chatbot.db").exists(),
    }


@app.get("/status")
async def status():
    """Alias of /health, kept for backward compatibility with existing tooling"""
    return await health()


@app.get("/")
async def root():
    return {
        "service": "Recykal Onboarding Agent",
        "status": "running",
        "engine": "deepseek-chat",
        "grounding": "knowledge.md only",
        "webhook": "/whatsapp",
        "upload_interface": "/upload-interface",
    }


@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("Recykal Onboarding Agent Starting")
    logger.info("=" * 60)
    logger.info(f"Upload directory: {PROJECT_DIR}/uploads")
    logger.info(f"Database: {PROJECT_DIR}/chatbot.db")
    rebuild_and_reload_knowledge()
    if DRIVE_SYNC.service:
        sync_knowledge_from_drive()
    else:
        logger.info(
            f"Google Drive sync not configured (no service account at "
            f"{GOOGLE_SERVICE_ACCOUNT_FILE}) — knowledge.md stays local-only"
        )
    logger.info(f"Knowledge base: {KNOWLEDGE_FILE} ({len(KNOWLEDGE_BASE)} chars)")
    logger.info(f"Vector store ready: {VSTORE.ready} ({len(VSTORE.chunks)} chunks)")
    if not DEEPSEEK_API_KEY:
        logger.warning("DEEPSEEK_API_KEY is not set — /whatsapp replies will fail")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Recykal Onboarding Agent Shutting Down")


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server on 0.0.0.0:8002")
    uvicorn.run(app, host="0.0.0.0", port=8002)
