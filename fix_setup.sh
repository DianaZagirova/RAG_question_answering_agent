#!/bin/bash

# 🔧 Quick Fix for Python 3.12 Setup Issues
# =========================================

set -e

echo "🔧 Fixing Python 3.12 compatibility issues..."
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip and build tools
echo "📦 Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

echo ""
echo "📚 Installing dependencies one by one..."
echo ""

# Install core dependencies in order
echo "1/7 Installing chromadb..."
pip install "chromadb>=0.4.18"

echo "2/7 Installing numpy (Python 3.12 compatible)..."
pip install "numpy>=1.26.0"

echo "3/7 Installing pandas..."
pip install "pandas>=2.0.0"

echo "4/7 Installing sentence-transformers..."
pip install "sentence-transformers>=2.2.0"

echo "5/7 Installing nltk..."
pip install "nltk>=3.8"

echo "6/7 Installing openai..."
pip install "openai>=1.0.0"

echo "7/7 Installing utilities..."
pip install "python-dotenv>=1.0.0" "tqdm>=4.66.0"

echo ""
echo "📥 Downloading NLTK data..."
python3 << 'EOF'
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    print("✓ NLTK data downloaded")
except Exception as e:
    print(f"⚠ NLTK download warning: {e}")
EOF

echo ""
echo "🧪 Testing imports..."
python3 << 'EOF'
try:
    import chromadb
    print("✓ chromadb")
    import sentence_transformers
    print("✓ sentence-transformers")
    import nltk
    print("✓ nltk")
    import openai
    print("✓ openai")
    import pandas
    print("✓ pandas")
    import numpy
    print("✓ numpy", numpy.__version__)
    print("\n✅ All core libraries imported successfully!")
except ImportError as e:
    print(f"✗ Import error: {e}")
    exit(1)
EOF

echo ""
echo "========================================="
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo "========================================="
echo ""
echo "📝 Next Steps:"
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Run the demo:"
echo "   python demo.py"
echo ""
echo "3. Or check the README:"
echo "   cat README.md"
echo ""
