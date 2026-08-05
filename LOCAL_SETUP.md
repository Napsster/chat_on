# Local Development Setup

## Setting Up on a New Machine

To move this project to a different laptop, clone from GitHub rather than
copy-pasting the folder — `venv/` is built for a specific OS/CPU and won't
transfer, and the local checkout usually has test data mixed in
(`data/users/*.json`, `chatbot.db`, `uploads/*`) that shouldn't travel with it.

```bash
git clone <repo-url> chatbot-local
cd chatbot-local
```

Then carry over exactly one file by hand — **`.env`** (gitignored, holds
secrets — copy it directly via AirDrop/USB/secure notes, not through chat or
any channel that logs plaintext). Everything else regenerates on first run.

Optional, only if you want continuity with the old machine's data:
- `chatbot.db` — existing user accounts, uploaded knowledge documents, question logs
- `data/users/*.json` — real (non-test) WhatsApp conversation history

Requires **Python 3.10+** (developed/tested on 3.12) — the code uses
`X | None` union syntax and fastembed/numpy 2.x, which need it.

Then follow Quick Start below.

## Quick Start (5 minutes)

### 1. Set Up Virtual Environment

```bash
cd chatbot-local

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Create Project Directories

```bash
mkdir -p uploads
```

### 4. Create .env File

```bash
cp .env.example .env
```

Fill in at minimum `DEEPSEEK_API_KEY` (required — `/whatsapp` and `/chat`
replies fail without it) and `JWT_SECRET` (falls back to an insecure
placeholder if unset — fine to skip only for a throwaway local test).
Everything else in `.env.example` is optional and dormant until configured.

### 5. Run the Chatbot

```bash
# Make sure venv is activated
source venv/bin/activate

# Start the server
python agent.py

# Or with uvicorn directly:
uvicorn agent:app --reload --host 0.0.0.0 --port 8002
```

Server will be running at: **http://localhost:8002**

### 6. Test the Upload Interface

- Upload interface: http://localhost:8002/upload-interface
- Health check: http://localhost:8002/health
- Status: http://localhost:8002/status

---

## Workflow: Local → GitHub → Hostinger

### Local Development
```bash
# 1. Make changes to code
nano agent.py  # or use your editor

# 2. Test locally
# The server reloads automatically with --reload flag

# 3. Commit to git
git add .
git commit -m "feat: describe your changes"

# 4. Push to GitHub
git push origin claude/html-mockup-internal-screens-mhyf46
```

### Deploy to Hostinger
Once tested locally and pushed to GitHub:

```bash
# SSH into Hostinger
ssh root@187.127.157.195

# Navigate to chatbot directory
cd /home/chetan/apps/onboarding-agent

# Pull latest from GitHub
git pull origin claude/html-mockup-internal-screens-mhyf46

# Install/update dependencies
pip install -r requirements.txt

# Restart the service
systemctl restart onboarding-agent

# Check status
systemctl status onboarding-agent
```

---

## Project Structure

```
chatbot-local/
├── agent.py                    # Main FastAPI application (WhatsApp bot + upload interface)
├── vector_store.py             # RAG: chunks + embeds knowledge.md, cosine retrieval
├── lookup_store.py             # Candidate directory (onboarding log) lookup by phone
├── file_upload_handler.py      # File upload & user auth logic (SQLite-backed)
├── knowledge.md                # Knowledge base the WhatsApp bot answers from
├── upload_interface.html       # Web UI for HR to upload/manage knowledge docs
├── index.html                  # HTML mockup (from initial task)
├── requirements.txt            # Python dependencies
├── .env.example                # Example environment variables
├── venv/                       # Virtual environment (created after setup)
├── uploads/                    # HR-uploaded knowledge documents
├── data/users/                 # Per-phone WhatsApp conversation state (gitignored, PII)
├── data/uploads/               # Documents received over WhatsApp (gitignored, PII)
└── chatbot.db                  # SQLite database (created on first run)
```

This is the same codebase deployed to Hostinger (`/home/chetan/apps/onboarding-agent`,
service `onboarding-agent`) — keep local, GitHub, and Hostinger in sync via the
push/pull workflow below.

---

## Common Tasks

### Update Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Add New Dependency
```bash
pip install <package-name>
pip freeze > requirements.txt  # Update requirements.txt
```

### View Database
```bash
sqlite3 chatbot.db
SELECT * FROM users;
SELECT * FROM uploads;
```

### Clear Database
```bash
rm chatbot.db
# Database will be recreated on next run
```

### Push to GitHub
```bash
git add .
git commit -m "your message"
git push origin claude/html-mockup-internal-screens-mhyf46
```

---

## Email Reports (Optional)

`reports.py` is a standalone script, not part of the running server — it's
meant to be invoked by cron, and loads its own `.env` (cron doesn't inherit
the systemd service's environment). Dormant until `SMTP_HOST`,
`SMTP_USERNAME`, and `SMTP_PASSWORD` are set in `.env`.

```bash
# Daily digest of that day's unanswered questions, once at end of day
crontab -e
# add:
55 23 * * * cd /path/to/chatbot-local && ./venv/bin/python reports.py --daily

# Weekly digest of repeated questions (semantic clustering, trailing 7 days)
0 9 * * 1 cd /path/to/chatbot-local && ./venv/bin/python reports.py --weekly
```

Use absolute paths in the crontab entry — cron's working directory isn't the
repo, and `reports.py` resolves `.env`/`knowledge*.md` relative to its own
file location, not the cron shell's cwd. Test manually first with
`./venv/bin/python reports.py --daily` before trusting the schedule.

---

## Troubleshooting

### Port already in use
```bash
# Change port in command:
uvicorn agent:app --reload --host 0.0.0.0 --port 8003
```

### Module not found error
```bash
# Make sure venv is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Database locked
```bash
# Kill any existing processes
pkill -f "uvicorn agent"

# Delete database to reset
rm chatbot.db

# Restart
python agent.py
```

---

## Next Steps

1. ✅ Clone & setup locally
2. ✅ Run on local machine
3. Make changes/improvements
4. Test locally
5. Push to GitHub (`git push`)
6. Deploy to Hostinger (pull + restart)

Happy coding! 🚀
