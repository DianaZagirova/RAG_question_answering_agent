# Virtual Environment Usage Guide

## Quick Start

### 1. Create Virtual Environment

```bash
cd /home/diana.z/hack/rag_agent
python3 -m venv venv
```

### 2. Activate Virtual Environment

```bash
source venv/bin/activate
```

You'll see `(venv)` prefix in your terminal:
```
(venv) user@host:~/hack/rag_agent$
```

### 3. Install Packages

**Option A: Install all at once (may have issues with Python 3.12)**
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Option B: Install step-by-step (recommended for Python 3.12)**
```bash
# Upgrade core tools
pip install --upgrade pip
pip install "setuptools>=65.5.1" wheel

# Install packages one by one
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install sentence-transformers
pip install chromadb
pip install openai
pip install nltk
pip install langchain langchain-community
pip install tiktoken numpy pandas tqdm
pip install peft python-dotenv

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### 4. Run Scripts with Virtual Environment

**Always activate venv first:**
```bash
source venv/bin/activate
```

**Then run your scripts:**
```bash
# Test Azure OpenAI connection
python src/core/llm_integration.py

# Run complete RAG
CUDA_VISIBLE_DEVICES=3 python scripts/rag_answer.py \
    --question "Does the paper suggest an aging biomarker?"

# Run ingestion
CUDA_VISIBLE_DEVICES=3 python src/ingestion/ingest_optimal.py --limit 100

# Run tests
python tests/test_system.py
```

### 5. Deactivate Virtual Environment

When done:
```bash
deactivate
```

---

## Alternative: Use Conda Environment

If you're using Miniconda/Anaconda (which you have), you can also use conda:

### Create Conda Environment

```bash
conda create -n rag_agent python=3.10 -y
conda activate rag_agent
```

**Note:** Using Python 3.10 instead of 3.12 avoids compatibility issues.

### Install Packages

```bash
# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other packages
pip install chromadb sentence-transformers nltk
pip install langchain langchain-community tiktoken
pip install openai transformers peft python-dotenv
pip install pandas numpy tqdm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Use Conda Environment

```bash
# Activate
conda activate rag_agent

# Run scripts
CUDA_VISIBLE_DEVICES=3 python scripts/rag_answer.py --question "..."

# Deactivate
conda deactivate
```

---

## Current Situation

You currently have packages installed in your **base conda environment**. This works fine, but using a virtual environment is better practice because:

✅ **Isolation**: Dependencies don't conflict with other projects  
✅ **Reproducibility**: Easy to recreate exact environment  
✅ **Safety**: Won't break other projects if you upgrade packages  

### Option 1: Keep Using Base Environment (Current)

```bash
# Just run scripts directly
CUDA_VISIBLE_DEVICES=3 python scripts/rag_answer.py --question "..."
```

**Pros:** Already working, no setup needed  
**Cons:** Not isolated, could affect other projects

### Option 2: Create Dedicated Environment (Recommended)

```bash
# Create conda env with Python 3.10 (more compatible)
conda create -n rag_agent python=3.10 -y
conda activate rag_agent

# Install packages
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt

# Use for all scripts
conda activate rag_agent
CUDA_VISIBLE_DEVICES=3 python scripts/rag_answer.py --question "..."
```

---

## Recommended Workflow

### Setup (One Time)

```bash
cd /home/diana.z/hack/rag_agent

# Create conda environment (Python 3.10 for compatibility)
conda create -n rag_agent python=3.10 -y
conda activate rag_agent

# Install PyTorch with CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other packages
pip install chromadb==0.4.22
pip install sentence-transformers==2.3.1
pip install transformers==4.36.2
pip install openai==1.10.0
pip install nltk==3.8.1
pip install langchain==0.1.4 langchain-community==0.0.16
pip install tiktoken==0.5.2
pip install pandas==2.0.3 tqdm==4.66.1
pip install peft python-dotenv

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Test
python src/core/llm_integration.py
```

### Daily Usage

```bash
# Activate environment
conda activate rag_agent

# Run your scripts
CUDA_VISIBLE_DEVICES=3 python scripts/rag_answer.py --all-questions

# Deactivate when done
conda deactivate
```

---

## Quick Reference

| Task | Command |
|------|---------|
| **Create venv** | `python3 -m venv venv` |
| **Activate venv** | `source venv/bin/activate` |
| **Deactivate venv** | `deactivate` |
| **Create conda env** | `conda create -n rag_agent python=3.10 -y` |
| **Activate conda** | `conda activate rag_agent` |
| **Deactivate conda** | `conda deactivate` |
| **List conda envs** | `conda env list` |
| **Delete conda env** | `conda env remove -n rag_agent` |
| **Export environment** | `conda env export > environment.yml` |
| **Create from export** | `conda env create -f environment.yml` |

---

## Troubleshooting

### Issue: "setuptools" error with Python 3.12

**Solution:** Use Python 3.10 instead
```bash
conda create -n rag_agent python=3.10 -y
```

### Issue: CUDA version mismatch

**Check CUDA version:**
```bash
nvidia-smi
```

**Install matching PyTorch:**
- CUDA 11.8: `pytorch-cuda=11.8`
- CUDA 12.1: `pytorch-cuda=12.1`

### Issue: Package conflicts

**Solution:** Install packages one by one
```bash
pip install package1
pip install package2
# etc.
```

---

## Best Practice

For this project, I recommend:

```bash
# One-time setup
conda create -n rag_agent python=3.10 -y
conda activate rag_agent
bash setup_venv_fixed.sh  # Or install packages manually

# Add to your .bashrc or .zshrc for convenience
alias rag='conda activate rag_agent'

# Then just use:
rag
CUDA_VISIBLE_DEVICES=3 python scripts/rag_answer.py --all-questions
```
