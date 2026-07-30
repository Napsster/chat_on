#!/bin/bash

# Chatbot Local Setup Script
# Run this once to set up the project

set -e  # Exit on error

echo "=========================================="
echo "Recykal Chatbot - Local Setup"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "agent.py" ]; then
    echo "❌ Error: agent.py not found in current directory"
    echo "Please run this from the chatbot-local directory"
    exit 1
fi

echo "📁 Setting up in: $(pwd)"
echo ""

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

echo ""
echo "📦 Activating virtual environment and installing dependencies..."
source venv/bin/activate

pip install --upgrade pip setuptools wheel > /dev/null 2>&1
pip install -r requirements.txt

echo "✅ Dependencies installed"
echo ""

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file..."
    cp .env.example .env
    echo "✅ .env created (using default settings)"
else
    echo "✅ .env already exists"
fi

echo ""

# Create necessary directories
echo "📂 Creating directories..."
mkdir -p uploads
mkdir -p knowledge
echo "✅ Directories created"

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Run the chatbot:"
echo "   python agent.py"
echo ""
echo "3. Open in browser:"
echo "   http://localhost:8002/upload-interface"
echo ""
echo "For more info, see LOCAL_SETUP.md"
echo ""
