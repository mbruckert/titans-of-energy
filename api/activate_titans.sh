#!/bin/bash
# Titans API Environment Activation Script

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found. Please create it with your API keys."
fi

echo "✅ Titans API environment activated!"
echo "📁 Current directory: $(pwd)"
echo "🐍 Python: $(which python)"
echo "📦 Virtual environment: activated"
echo ""
echo "To start the API server, run:"
echo "  python app.py"
echo ""
echo "To deactivate the environment, run:"
echo "  deactivate"
