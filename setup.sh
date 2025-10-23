#!/bin/bash

# 🧬 Aging Theories RAG System - Setup Script
# Agentic AI Against Aging Hackathon
# ========================================

set -e  # Exit on error

echo "🧬 Aging Theories RAG System - Setup"
echo "====================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo "📋 Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}✗ Python 3.8+ required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo -e "${YELLOW}⚠ Virtual environment already exists${NC}"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing old virtual environment..."
        rm -rf venv
    else
        echo "Using existing virtual environment"
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment exists${NC}"
fi
echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Upgrade pip and build tools
echo "📦 Upgrading pip and build tools..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo -e "${GREEN}✓ pip and build tools upgraded${NC}"
echo ""

# Install requirements
echo "📚 Installing dependencies..."
if [ -f "requirements.txt" ]; then
    echo "  Installing core dependencies..."
    pip install -r requirements.txt
    
    # Check if installation was successful
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Dependencies installed${NC}"
    else
        echo -e "${YELLOW}⚠ Some dependencies failed to install${NC}"
        echo "  Trying with --no-cache-dir..."
        pip install --no-cache-dir -r requirements.txt
    fi
else
    echo -e "${RED}✗ requirements.txt not found${NC}"
    exit 1
fi
echo ""

# Download NLTK data
echo "📥 Downloading NLTK data..."
python3 << EOF
import nltk
import ssl

# Handle SSL certificate issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    print("✓ NLTK data downloaded")
except Exception as e:
    print(f"⚠ NLTK download warning: {e}")
EOF
echo ""

# Check GPU availability
echo "🎮 Checking GPU availability..."
python3 << EOF
import torch
if torch.cuda.is_available():
    print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠ No GPU available (will use CPU)")
EOF
echo ""

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p chroma_db_optimal
mkdir -p data
mkdir -p rag_results
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Check for .env file
echo "🔐 Checking configuration..."
if [ -f ".env" ]; then
    echo -e "${GREEN}✓ .env file exists${NC}"
else
    echo -e "${YELLOW}⚠ .env file not found${NC}"
    echo ""
    echo "Creating template .env file..."
    cat > .env << 'ENVEOF'
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4.1-mini

# Database Paths
DB_PATH=/path/to/papers.db
PERSIST_DIR=./chroma_db_optimal
COLLECTION_NAME=scientific_papers_optimal

# Chunking Configuration
CHUNK_SIZE=1500
CHUNK_OVERLAP=300
MIN_CHUNK_SIZE=200

# Embedding Model
EMBEDDING_MODEL=allenai/specter2
BACKUP_EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2

# Hardware
CUDA_DEVICE=3

# LLM Parameters
TEMPERATURE=0.2
MAX_TOKENS=1000
ENVEOF
    echo -e "${YELLOW}⚠ Please edit .env with your Azure OpenAI credentials${NC}"
fi
echo ""

# Test imports
echo "🧪 Testing imports..."
python3 << EOF
try:
    import chromadb
    import sentence_transformers
    import nltk
    import openai
    import pandas
    import tqdm
    print("✓ All core libraries imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    exit(1)
EOF
echo ""

# Print summary
echo "========================================="
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo "========================================="
echo ""
echo "📝 Next Steps:"
echo ""
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Configure .env file with your credentials:"
echo "   nano .env"
echo ""
echo "3. Run the demo:"
echo "   python demo.py"
echo ""
echo "4. Or process all papers:"
echo "   python scripts/run_rag_on_all_papers.py --help"
echo ""
echo "📚 Documentation:"
echo "   - README.md - Main documentation"
echo "   - documentation/ - Technical guides"
echo "   - config/optimal_config.py - Configuration details"
echo ""
echo "🎯 For hackathon judges:"
echo "   - Review README.md for technical details"
echo "   - Check documentation/ for advanced techniques"
echo "   - Run demo.py for quick demonstration"
echo ""
echo "========================================="
