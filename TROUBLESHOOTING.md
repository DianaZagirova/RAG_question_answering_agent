# 🔧 Troubleshooting Guide

## Python 3.12 Setup Issues

### Problem: numpy installation fails with "Cannot import 'setuptools.build_meta'"

**Cause**: numpy 1.24.3 doesn't support Python 3.12. You need numpy 1.26.0+

**Solution 1: Use the fix script (Recommended)**
```bash
# Remove old venv
rm -rf venv

# Run the fix script
./fix_setup.sh
```

**Solution 2: Manual installation**
```bash
# Remove old venv
rm -rf venv

# Create new venv
python3 -m venv venv
source venv/bin/activate

# Upgrade build tools
pip install --upgrade pip setuptools wheel

# Install numpy first (Python 3.12 compatible)
pip install "numpy>=1.26.0"

# Install other dependencies
pip install -r requirements.txt
```

**Solution 3: Use Python 3.11 or 3.10**
```bash
# If you have Python 3.11 available
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Common Issues

### Issue 1: "ModuleNotFoundError: No module named 'chromadb'"

**Cause**: Dependencies not installed

**Solution**:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

---

### Issue 2: "FileNotFoundError: [Errno 2] No such file or directory: 'chroma_db_optimal'"

**Cause**: Vector database hasn't been created yet

**Solution**: This is expected if you haven't run ingestion yet. The demo will show a warning but continue.

---

### Issue 3: NLTK data not found

**Cause**: NLTK punkt tokenizer data not downloaded

**Solution**:
```bash
python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

### Issue 4: GPU not available

**Cause**: PyTorch not installed or CUDA not configured

**Solution**: The system will automatically fall back to CPU. To use GPU:
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

### Issue 5: Azure OpenAI authentication error

**Cause**: .env file not configured or invalid credentials

**Solution**:
```bash
# Create/edit .env file
nano .env

# Add your credentials:
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4.1-mini
```

---

## Quick Fixes

### Reset Everything
```bash
# Remove virtual environment
rm -rf venv

# Remove any cached files
rm -rf __pycache__ src/__pycache__ src/core/__pycache__

# Run fix script
./fix_setup.sh
```

### Check Installation
```bash
source venv/bin/activate

python3 << 'EOF'
import sys
print(f"Python version: {sys.version}")

import chromadb
print(f"✓ chromadb {chromadb.__version__}")

import sentence_transformers
print(f"✓ sentence-transformers")

import nltk
print(f"✓ nltk {nltk.__version__}")

import numpy
print(f"✓ numpy {numpy.__version__}")

import pandas
print(f"✓ pandas {pandas.__version__}")

import openai
print(f"✓ openai {openai.__version__}")
EOF
```

---

## Python Version Compatibility

| Python Version | Status | Notes |
|----------------|--------|-------|
| 3.8 | ✅ Supported | Use numpy>=1.24.0 |
| 3.9 | ✅ Supported | Use numpy>=1.24.0 |
| 3.10 | ✅ Supported | Use numpy>=1.24.0 |
| 3.11 | ✅ Supported | Use numpy>=1.24.0 |
| 3.12 | ✅ Supported | **Must use numpy>=1.26.0** |

---

## Dependency Versions

### Minimum Requirements (Python 3.12)
```
chromadb>=0.4.18
sentence-transformers>=2.2.0
nltk>=3.8
openai>=1.0.0
python-dotenv>=1.0.0
tqdm>=4.66.0
numpy>=1.26.0  # Critical for Python 3.12
pandas>=2.0.0
```

### Optional Dependencies
```
torch>=2.1.0  # For GPU acceleration
transformers>=4.36.0  # For advanced models
peft>=0.7.0  # For SPECTER2
langchain>=0.1.0  # For advanced RAG features
```

---

## Getting Help

1. **Check this guide first** - Most issues are covered here
2. **Run the fix script** - `./fix_setup.sh`
3. **Check Python version** - `python3 --version`
4. **Verify installation** - Run the check script above
5. **Review logs** - Look for specific error messages

---

## Contact

For persistent issues:
- Review the README.md for detailed documentation
- Check the demo.py for working examples
- Ensure all prerequisites are met (Python 3.8+, pip, venv)
