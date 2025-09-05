#!/bin/bash
# Activation script for Student Coach Q&A project
echo "Activating Student Coach Q&A environment..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

echo "Virtual environment activated!"
echo "Python path: $(which python)"
echo "Pip path: $(which pip)"
echo ""
echo "To deactivate, run: deactivate"
echo "To run the app: streamlit run app.py"
