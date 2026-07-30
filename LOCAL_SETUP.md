# Local Development Setup

## Quick Start (5 minutes)

### 1. Set Up Virtual Environment

```bash
cd /home/user/chatbot-local

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
mkdir -p knowledge
```

### 4. Create .env File

```bash
cp .env.example .env
# Edit .env if needed (usually defaults work for local dev)
```

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
├── agent.py                    # Main FastAPI application
├── file_upload_handler.py      # File upload & user auth logic
├── upload_interface.html       # Web UI for uploads
├── index.html                  # HTML mockup (from initial task)
├── requirements.txt            # Python dependencies
├── .env.example               # Example environment variables
├── venv/                      # Virtual environment (created after setup)
├── uploads/                   # User uploads directory
├── knowledge/                 # Knowledge base files
└── chatbot.db                # SQLite database (created on first run)
```

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
