"""
Recykal HR Chatbot - Updated with File Upload Interface
Add these endpoints to your existing agent.py
"""

from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from file_upload_handler import FileUploadManager, UploadedFileProcessor
import logging
from datetime import datetime
from pathlib import Path
import shutil
import os

# Try to import VectorStore (optional for local dev)
try:
    from vector_store_updated import VectorStore
except ImportError:
    VectorStore = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Recykal HR Chatbot with Uploads")

# Initialize managers
# Detect environment (local vs production)
IS_LOCAL = os.getenv('IS_LOCAL', 'true').lower() == 'true'

if IS_LOCAL:
    PROJECT_DIR = str(Path(__file__).parent)
else:
    PROJECT_DIR = "/home/chetan/apps/onboarding-agent"

logger.info(f"Running in {'LOCAL' if IS_LOCAL else 'PRODUCTION'} mode")
logger.info(f"Project directory: {PROJECT_DIR}")

upload_manager = FileUploadManager(
    upload_dir=f"{PROJECT_DIR}/uploads",
    db_path=f"{PROJECT_DIR}/chatbot.db"
)
file_processor = UploadedFileProcessor(knowledge_dir=PROJECT_DIR)

# Initialize vector store (optional for local dev)
vs = None
try:
    if not IS_LOCAL or os.path.exists(f"{PROJECT_DIR}/google-credentials.json"):
        vs = VectorStore(
            knowledge_file=f"{PROJECT_DIR}/knowledge.md",
            embeddings_file=f"{PROJECT_DIR}/knowledge.vec.npz",
            use_google_drive=not IS_LOCAL,
            service_account_file=f"{PROJECT_DIR}/google-credentials.json",
            folder_id="1aGaZa6N2i2CbZ1xY7k9AAzUO9b3hy4Ju"
        )
except Exception as e:
    logger.warning(f"Vector store initialization failed: {e}")
    logger.info("Chatbot will work without vector search in local mode")

# ============================================================================
# FILE UPLOAD ENDPOINTS
# ============================================================================

@app.get("/upload-interface")
async def get_upload_interface():
    """Serve the upload interface HTML"""
    interface_path = Path(__file__).parent / "upload_interface.html"
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
            return JSONResponse({
                "success": True,
                "message": message,
                "username": username
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
    username: str = Form(...)
):
    """Upload files to the chatbot"""
    try:
        # Get form data
        form = await request.form()

        if 'files' not in form:
            return JSONResponse(
                {"error": "No files provided"},
                status_code=400
            )

        files = form.getlist('files')
        uploaded_count = 0
        errors = []

        for file in files:
            if not file.filename:
                continue

            # Read file content
            content = await file.read()

            # Validate file
            valid, msg = upload_manager.validate_file(file.filename, len(content))
            if not valid:
                errors.append(f"{file.filename}: {msg}")
                continue

            # Save file
            success, msg, file_path = upload_manager.save_file(
                username=username,
                file_content=content,
                filename=file.filename
            )

            if success:
                # Process file
                if file.filename.endswith('.md'):
                    file_processor.process_markdown(file_path, username)

                uploaded_count += 1
                logger.info(f"File uploaded: {file.filename} by {username}")
            else:
                errors.append(f"{file.filename}: {msg}")

        # Return results
        response = {
            "success": uploaded_count > 0,
            "uploaded": uploaded_count,
            "total": len(files),
            "message": f"Successfully uploaded {uploaded_count} file(s)"
        }

        if errors:
            response["errors"] = errors

        return JSONResponse(response)

    except Exception as e:
        logger.error(f"Upload error: {e}")
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )

@app.post("/upload/text")
async def upload_text(request: Request):
    """Upload text content as markdown"""
    try:
        data = await request.json()

        username = data.get('username')
        title = data.get('title')
        content = data.get('content')

        if not all([username, title, content]):
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
async def get_upload_history(username: str):
    """Get upload history for user"""
    try:
        if not username:
            return JSONResponse(
                {"error": "Username required"},
                status_code=400
            )

        history = upload_manager.get_upload_history(username)

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

# ============================================================================
# EXISTING ENDPOINTS (from your original agent.py)
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "service": "Recykal HR Chatbot"
    }

@app.get("/status")
async def get_status():
    """Get chatbot status including Google Drive sync and upload stats"""
    try:
        upload_stats = {
            "upload_dir_exists": Path(f"{PROJECT_DIR}/uploads").exists(),
            "db_initialized": Path(f"{PROJECT_DIR}/chatbot.db").exists()
        }

        return {
            "timestamp": datetime.now().isoformat(),
            "service": "Recykal HR Chatbot with Uploads",
            "vector_store": vs.get_status() if vs else None,
            "uploads": upload_stats
        }
    except Exception as e:
        logger.error(f"Status error: {e}")
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )

@app.post("/chat")
async def chat(request: Request):
    """Chat endpoint - same as original agent.py"""
    try:
        data = await request.form() if request.headers.get('content-type') == 'application/x-www-form-urlencoded' else await request.json()
        user_message = data.get('Body') or data.get('message', '')

        if not user_message:
            return PlainTextResponse("Please send a message")

        logger.info(f"Received message: {user_message[:100]}")

        # Search knowledge base
        if vs:
            results = vs.search(user_message, top_k=3)

            if results:
                context = "\n".join([f"- {chunk[:200]}..." for chunk, score in results])
                response = generate_response(user_message, context)
            else:
                response = "I couldn't find relevant information. Please try rephrasing your question."
        else:
            response = "The knowledge base is not available."

        return PlainTextResponse(response)

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return PlainTextResponse(f"Error: {str(e)}")

@app.post("/twilio/webhook")
async def twilio_webhook(request: Request):
    """Twilio WhatsApp webhook - same as original agent.py"""
    try:
        form_data = await request.form()
        message_body = form_data.get('Body', '')

        logger.info(f"Twilio message: {message_body[:100]}")

        if vs:
            results = vs.search(message_body, top_k=3)
            if results:
                context = "\n".join([chunk[:150] for chunk, _ in results])
                response_text = generate_response(message_body, context)
            else:
                response_text = "I couldn't find that information. Please ask about company policies or onboarding."
        else:
            response_text = "The chatbot is temporarily unavailable."

        twilio_response = f'<?xml version="1.0" encoding="UTF-8"?><Response><Message>{response_text}</Message></Response>'
        return PlainTextResponse(twilio_response, media_type="application/xml")

    except Exception as e:
        logger.error(f"Twilio error: {e}")
        return PlainTextResponse('<?xml version="1.0" encoding="UTF-8"?><Response><Message>Error processing message</Message></Response>', media_type="application/xml")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_response(user_query: str, context: str) -> str:
    """Generate a response using retrieved context"""
    return f"""Based on our knowledge base:

{context}

For more information, contact HR at hr@recykal.com"""

# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("=" * 60)
    logger.info("Recykal HR Chatbot with File Uploads Starting")
    logger.info("=" * 60)
    logger.info(f"Upload directory: {PROJECT_DIR}/uploads")
    logger.info(f"Database: {PROJECT_DIR}/chatbot.db")
    if vs:
        logger.info(f"Vector store: {vs.get_status()}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Recykal HR Chatbot Shutting Down")

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server on 0.0.0.0:8002")
    uvicorn.run(app, host="0.0.0.0", port=8002)
