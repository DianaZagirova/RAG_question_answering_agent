#!/bin/bash
# Fixed setup script for RAG Agent virtual environment (Python 3.12 compatible)

set -e  # Exit on error

echo "=================================="
echo "RAG Agent - Virtual Environment Setup"
echo "=================================="
echo ""

# Create virtual environment
VENV_DIR="venv"

if [ -d "$VENV_DIR" ]; then
    echo "⚠ Virtual environment already exists. Removing..."
    rm -rf "$VENV_DIR"
fi

echo "Creating virtual environment..."
python3 -m venv "$VENV_DIR"
echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
echo "✓ Activated"
echo ""

# Upgrade pip and install compatible setuptools for Python 3.12
echo "Installing compatible pip, setuptools, and wheel..."
pip install --upgrade pip
pip install "setuptools>=65.5.1" "wheel>=0.38.0"
echo "✓ Core tools installed"
echo ""

# Install PyTorch first (it's picky about dependencies)
echo "Installing PyTorch..."
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
echo "✓ PyTorch installed"
echo ""

# Install other requirements one by one to avoid conflicts
echo "Installing requirements..."

pip install chromadb==0.4.22
pip install "sentence-transformers==2.3.1"
pip install nltk==3.8.1
pip install langchain==0.1.4
pip install langchain-community==0.0.16
pip install tiktoken==0.5.2
pip install "numpy<2.0"
pip install pandas==2.0.3
pip install tqdm==4.66.1
pip install openai==1.10.0
pip install "transformers==4.36.2"
pip install peft
pip install python-dotenv

echo "✓ All packages installed"
echo ""

# Download NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"
echo "✓ NLTK data downloaded"
echo ""

echo "=================================="
echo "✅ Setup Complete!"
echo "=================================="
echo ""
echo "Virtual environment location: ./venv"
echo ""
echo "To activate in the future:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""
echo "Test the installation:"
echo "  source venv/bin/activate"
echo "  python src/core/llm_integration.py"
echo ""
