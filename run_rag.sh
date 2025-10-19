#!/bin/bash
# Wrapper script to run RAG system with virtual environment
# Usage: ./run_rag.sh --question "Your question here"

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================="
echo "RAG Agent Wrapper"
echo -e "==================================${NC}"
echo ""

# Check if conda is available
if command -v conda &> /dev/null; then
    # Check if rag_agent environment exists
    if conda env list | grep -q "rag_agent"; then
        echo -e "${GREEN}✓ Using conda environment: rag_agent${NC}"
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate rag_agent
    else
        echo "⚠ Conda environment 'rag_agent' not found"
        echo "Create it with: conda create -n rag_agent python=3.10"
    fi
elif [ -d "venv" ]; then
    echo -e "${GREEN}✓ Using virtual environment: venv${NC}"
    source venv/bin/activate
else
    echo "⚠ No virtual environment found. Using system Python."
fi

echo ""

# Set GPU device (default to 3)
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE:-3}
echo "GPU Device: $CUDA_VISIBLE_DEVICES"
echo ""

# Run the RAG script with all arguments passed through
python scripts/rag_answer.py "$@"

# Deactivate if conda
if command -v conda &> /dev/null && [ ! -z "$CONDA_DEFAULT_ENV" ]; then
    conda deactivate
fi
