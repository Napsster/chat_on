#!/bin/bash

# Chatbot Local Run Script
# Activates venv and starts the chatbot

set -e

if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Run ./setup.sh first"
    exit 1
fi

echo "🚀 Starting Recykal Chatbot..."
echo ""

source venv/bin/activate

# Set local mode
export IS_LOCAL=true

echo "📍 Chatbot starting on http://localhost:8002"
echo "📍 Upload interface: http://localhost:8002/upload-interface"
echo ""
echo "Press Ctrl+C to stop"
echo ""

uvicorn agent:app --reload --host 0.0.0.0 --port 8002
