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
import logging
import mimetypes
from pathlib import Path
from datetime import datetime

import requests
from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import Response, JSONResponse, FileResponse
from twilio.twiml.messaging_response import MessagingResponse

from vector_store import VectorStore
from lookup_store import CandidateDirectory
from google_drive_sync import GoogleDriveSync
from file_upload_handler import FileUploadManager, UploadedFileProcessor
from auth import init_auth, create_token, get_current_user

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Recykal Onboarding Agent")

APP_DIR = Path(__file__).parent
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
#   Google Drive sync if that's ever configured).
# knowledge.md: generated (gitignored) — base + every active HR-uploaded
#   document, rebuilt by rebuild_and_reload_knowledge(). This is what the bot
#   actually loads/embeds.
KNOWLEDGE_BASE_FILE = APP_DIR / "knowledge_base.md"
KNOWLEDGE_FILE = APP_DIR / "knowledge.md"
TOP_K = 8


def reload_knowledge_base():
    """(Re)load knowledge.md and rebuild the vector store from its current content."""
    global KNOWLEDGE_BASE, VSTORE
    try:
        KNOWLEDGE_BASE = KNOWLEDGE_FILE.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"Could not load knowledge.md: {e}")
        KNOWLEDGE_BASE = ""
    VSTORE = VectorStore(KNOWLEDGE_FILE)


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

# Shown once, appended to a brand-new user's first reply only.
FIRST_MESSAGE_DISCLAIMER = (
    "\n\n_Just so you know: this chat is for general guidance, not an official "
    "or compliance record — please confirm anything important with People & Culture._"
)

PERSONA_RULES = f"""You are Maya, the pre-onboarding assistant for Recykal (legal name \
Rapidue Technologies Pvt Ltd). You chat with brand-new hires over WhatsApp to welcome \
them and answer their onboarding questions.

################  ABSOLUTE GROUNDING RULE  ################
You must answer ONLY using facts found in the KNOWLEDGE BASE provided below. This is your \
single source of truth.
- NEVER use outside/general knowledge, prior training, or assumptions.
- NEVER invent, guess, estimate, or "fill in" details that are not explicitly in the \
KNOWLEDGE BASE — not even plausible-sounding ones (numbers, dates, policies, names, links).
- If the answer is not clearly in the KNOWLEDGE BASE, do NOT attempt an answer. Instead say, \
warmly and in your own words, exactly this idea: "{DEFLECTION}"
- If you are unsure whether something is in the KNOWLEDGE BASE, treat it as not there and \
deflect. Accuracy matters more than being helpful.

################  NEVER ANSWER THESE (always deflect to People & Culture)  ################
Even if related info appears in the KNOWLEDGE BASE, do not give individualized answers on:
- Individual salary details / a person's specific CTC or offer numbers
- Personal appraisal or performance information
- Confidential employee data
- Interpretation of an individual's offer letter
- Legal or medical advice
- Exceptions to any policy
For these, use the deflection to the People & Culture team.

################  STYLE  ################
- Warm, human, and genuinely welcoming — like a friendly P&C colleague, not a form or a bot.
- Professional and trustworthy. Vary your phrasing; never sound scripted or repetitive.
- Keep replies short and WhatsApp-friendly (usually 1-3 sentences). No walls of text.
- You may greet, acknowledge, and make light small talk naturally — but the moment a reply \
contains any fact/policy/number/link/contact, it MUST come from the KNOWLEDGE BASE.
- When helpful, share the exact contact email from the KNOWLEDGE BASE (e.g. \
peopleandculture@recykal.com, itsupport@recykal.com) rather than a vague "contact HR".
- Use an emoji only occasionally, when it feels natural.

################  YOUR GOAL  ################
Help the new hire feel welcomed and get their onboarding questions answered accurately from \
the KNOWLEDGE BASE. If they ask something out of scope, gently deflect and steer back to how \
you can help with onboarding."""


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


def build_system_prompt(kb_context: str, profile_block: str | None = None) -> str:
    parts = [PERSONA_RULES]
    if profile_block:
        parts.append(CANDIDATE_RULES + "\n\n" + profile_block +
                     "\n################  END OF CANDIDATE PROFILE  ################")
    parts.append(
        "################  KNOWLEDGE BASE (your only source of truth)  ################\n"
        + kb_context
        + "\n################  END OF KNOWLEDGE BASE  ################"
    )
    return "\n\n".join(parts)


def retrieve_context(query: str) -> str:
    """Top-k relevant chunks via vector memory; full KB as fallback."""
    if VSTORE.ready:
        chunks = VSTORE.retrieve(query, k=TOP_K)
        if chunks:
            return "\n\n---\n\n".join(chunks)
    return KNOWLEDGE_BASE


def get_user_data(phone: str) -> dict:
    user_file = DATA_DIR / f"{phone}.json"
    if user_file.exists():
        with open(user_file) as f:
            return json.load(f)
    return {"phone": phone, "email": None, "history": []}


def save_user_data(phone: str, data: dict):
    with open(DATA_DIR / f"{phone}.json", "w") as f:
        json.dump(data, f, indent=2)


# --- Media (document upload) handling, WhatsApp side -----------------------
TWILIO_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")


def parse_media(form) -> list[dict]:
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


def save_media(phone: str, media: list[dict]) -> list[dict]:
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


def generate_reply(user_data: dict, kb_context: str, profile_block: str | None = None) -> str:
    """Call DeepSeek with retrieved KB context + candidate profile + recent history."""
    messages = [{"role": "system", "content": build_system_prompt(kb_context, profile_block)}]
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
                "max_tokens": 350,
                "temperature": 0.3,  # low → faithful to KB
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return (
            "Sorry, I glitched for a second there — could you send that again? "
            "I'm here to help with your Recykal onboarding."
        )


@app.get("/whatsapp")
async def whatsapp_webhook_get():
    return Response(content="", status_code=200)


@app.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    try:
        form_data = await request.form()
        phone = form_data.get("From", "").replace("whatsapp:", "")
        incoming_message = (form_data.get("Body", "") or "").strip()
        media = parse_media(form_data)

        logger.info(f"📨 [WEBHOOK] From: {phone} | Message: {incoming_message} | media: {len(media)}")

        sync_knowledge_from_drive()

        user_data = get_user_data(phone)
        is_first_interaction = not user_data["history"]

        # Match the caller against the onboarding log (if configured).
        candidate = DIRECTORY.lookup(phone)
        profile_block = DIRECTORY.profile_block(candidate) if candidate else None
        first_name = None
        if candidate:
            full = DIRECTORY.name_of(candidate)
            first_name = full.split()[0] if full else None

        # Genuine empty ping (no text AND no attachment) → greeting.
        if not incoming_message and not media:
            hello = f"Hey {first_name}! 👋" if first_name else "Hey! 👋"
            reply = f"{hello} I'm Maya from Recykal's People & Culture team. How can I help with your onboarding today?"
            if is_first_interaction:
                reply += FIRST_MESSAGE_DISCLAIMER
            user_data["history"].append({"role": "assistant", "content": reply})
            save_user_data(phone, user_data)
            resp = MessagingResponse()
            resp.message(reply)
            return Response(content=str(resp), media_type="application/xml")

        # opportunistic email capture
        if incoming_message and not user_data.get("email"):
            m = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", incoming_message)
            if m:
                user_data["email"] = m.group(0)

        # Handle uploaded documents: save them and build a turn the LLM can act on.
        if media:
            saved = save_media(phone, media)
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
            retrieval_query = f"{caption} document upload onboarding verification".strip()
        else:
            prev_user = next(
                (m["content"] for m in reversed(user_data["history"])
                 if m["role"] == "user"),
                "",
            )
            user_turn = incoming_message
            retrieval_query = f"{prev_user} {incoming_message}".strip()

        kb_context = retrieve_context(retrieval_query)
        user_data["history"].append({"role": "user", "content": user_turn})
        reply = generate_reply(user_data, kb_context, profile_block)
        if is_first_interaction:
            reply += FIRST_MESSAGE_DISCLAIMER
        user_data["history"].append({"role": "assistant", "content": reply})
        save_user_data(phone, user_data)

        resp = MessagingResponse()
        resp.message(reply)
        return Response(content=str(resp), media_type="application/xml")

    except Exception as e:
        logger.error(f"ERROR in webhook: {e}")
        resp = MessagingResponse()
        resp.message("Sorry, something hiccuped on my end. Mind sending that again?")
        return Response(content=str(resp), media_type="application/xml")


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
    """Regenerate knowledge.md from knowledge_base.md + active HR-uploaded
    documents, then reload it into the running vector store."""
    active_docs = upload_manager.list_knowledge_documents(active_only=True)
    success, message = file_processor.rebuild_knowledge_md(active_docs)
    if success:
        reload_knowledge_base()
    else:
        logger.error(f"Could not rebuild knowledge.md: {message}")


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
            return JSONResponse({
                "success": True,
                "message": message,
                "username": username,
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
    current_user: dict = Depends(get_current_user)
):
    """Upload files to the chatbot"""
    username = current_user["username"]
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
                            title=file.filename, content=text, uploaded_by=username
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
async def upload_text(request: Request, current_user: dict = Depends(get_current_user)):
    """Upload text content as markdown"""
    try:
        data = await request.json()

        username = current_user["username"]
        title = data.get('title')
        content = data.get('content')

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
                title=title, content=content, uploaded_by=username
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
async def get_upload_history(current_user: dict = Depends(get_current_user)):
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
async def list_knowledge_documents(current_user: dict = Depends(get_current_user)):
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
async def toggle_knowledge_document(doc_id: int, active: bool = Form(...), current_user: dict = Depends(get_current_user)):
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


@app.delete("/knowledge/documents/{doc_id}")
async def delete_knowledge_document(doc_id: int, current_user: dict = Depends(get_current_user)):
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
        "retrieval_top_k": TOP_K,
        "candidate_directory": DIRECTORY.ready,
        "candidate_rows": len(DIRECTORY._index),
        "directory_source": DIRECTORY.source,
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
