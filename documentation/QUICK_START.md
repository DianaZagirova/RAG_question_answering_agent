# 🚀 Quick Start Guide

## ✅ System is Ready!

Your RAG system is **fully configured and tested**. Everything works with your current Python environment.

---

## 🎯 Two Ways to Run

### Option 1: Direct Execution (Current - Works Now)

Just run scripts directly with your current environment:

```bash
cd /home/diana.z/hack/rag_agent

# Answer a single question
CUDA_VISIBLE_DEVICES=3 python scripts/rag_answer.py \
    --question "Does the paper suggest an aging biomarker?"

# Answer all 9 questions
CUDA_VISIBLE_DEVICES=3 python scripts/rag_answer.py --all-questions

# Run ingestion
CUDA_VISIBLE_DEVICES=3 python src/ingestion/ingest_optimal.py --reset
```

**Pros:** Works immediately, no setup needed  
**Cons:** Uses system Python, could affect other projects

### Option 2: Use Wrapper Scripts (Easier)

Use the wrapper scripts that handle virtual environment automatically:

```bash
# Answer questions
./run_rag.sh --question "Does the paper suggest an aging biomarker?"
./run_rag.sh --all-questions

# Run ingestion
./run_ingest.sh --limit 100 --reset
./run_ingest.sh --reset  # Full ingestion
```

**Pros:** Cleaner syntax, handles venv automatically  
**Cons:** Requires venv setup first (optional)

---

## 📦 Virtual Environment Setup (Optional but Recommended)

### Using Conda (Recommended - You Have Miniconda)

```bash
# 1. Create environment with Python 3.10 (more compatible than 3.12)
conda create -n rag_agent python=3.10 -y

# 2. Activate
conda activate rag_agent

# 3. Install PyTorch with CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 4. Install other packages
pip install chromadb==0.4.22 sentence-transformers==2.3.1 transformers==4.36.2
pip install openai==1.10.0 nltk==3.8.1 peft python-dotenv
pip install langchain==0.1.4 langchain-community==0.0.16 tiktoken==0.5.2
pip install pandas==2.0.3 numpy tqdm==4.66.1

# 5. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# 6. Test
python src/core/llm_integration.py
```

### Daily Usage with Conda

```bash
# Activate (do this once per terminal session)
conda activate rag_agent

# Run scripts (wrapper handles activation automatically)
./run_rag.sh --all-questions

# Or run directly
CUDA_VISIBLE_DEVICES=3 python scripts/rag_answer.py --all-questions

# Deactivate when done
conda deactivate
```

---

## 🎯 Common Commands

### Answer Questions

```bash
# Single question
./run_rag.sh --question "Does the paper suggest an aging biomarker?" \
    --n-results 12 --temperature 0.3

# All 9 critical questions
./run_rag.sh --all-questions --output my_results.json

# Specific question with custom parameters
./run_rag.sh \
    --question "Does the paper explain why naked mole rats live 40+ years?" \
    --n-results 15 \
    --temperature 0.3 \
    --max-tokens 800
```

### Ingestion

```bash
# Test with 100 papers
./run_ingest.sh --limit 100 --reset

# Full ingestion (all 42,735 papers)
./run_ingest.sh --reset

# Continue ingestion (don't reset)
./run_ingest.sh --limit 10000
```

### Testing

```bash
# Run all tests
python tests/test_system.py

# Test Azure OpenAI connection
python src/core/llm_integration.py

# Test retrieval only (no LLM)
CUDA_VISIBLE_DEVICES=3 python scripts/query_aging_papers.py \
    --question "test" --question-type biomarker
```

---

## 📊 Current Status

### Database
- **Papers available:** 42,735
- **Papers ingested:** 50 (test set)
- **Total chunks:** 1,352
- **Database:** `./chroma_db_optimal/`

### Configuration
- **Chunk size:** 1500 chars (optimal)
- **Overlap:** 300 chars
- **Embedding model:** all-mpnet-base-v2
- **LLM:** gpt-4.1-mini (Azure OpenAI)
- **GPU:** Device 3

### Files
```
✓ src/core/llm_integration.py       # Azure OpenAI integration
✓ scripts/rag_answer.py             # Complete RAG (retrieval + LLM)
✓ src/ingestion/ingest_optimal.py   # Optimal ingestion
✓ .env                               # Azure credentials
✓ run_rag.sh                         # Wrapper for answering
✓ run_ingest.sh                      # Wrapper for ingestion
```

---

## 🎓 Example Workflow

### First Time Setup (with Conda)

```bash
cd /home/diana.z/hack/rag_agent

# Create conda environment
conda create -n rag_agent python=3.10 -y
conda activate rag_agent

# Install packages
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt

# Test
python src/core/llm_integration.py
```

### Daily Usage

```bash
cd /home/diana.z/hack/rag_agent

# Activate environment (if using conda)
conda activate rag_agent

# Answer all 9 questions
./run_rag.sh --all-questions

# Check results
cat rag_answers.json
```

### Full Production Run

```bash
# 1. Ingest all papers (~20-25 minutes)
./run_ingest.sh --reset

# 2. Answer all questions
./run_rag.sh --all-questions --output final_answers.json

# 3. Review answers
less final_answers.json
```

---

## 💡 Pro Tips

### 1. Add Alias to .bashrc

```bash
echo 'alias rag="cd /home/diana.z/hack/rag_agent && conda activate rag_agent"' >> ~/.bashrc
source ~/.bashrc

# Then just use:
rag
./run_rag.sh --all-questions
```

### 2. Set Default GPU in .env

Already configured in `.env`:
```bash
CUDA_DEVICE=3
```

### 3. Save Common Queries

Create a file `queries.txt`:
```
Does the paper suggest an aging biomarker?
Does the paper suggest a molecular mechanism of aging?
Does the paper explain calorie restriction effects?
```

Run batch:
```bash
while read question; do
    ./run_rag.sh --question "$question" --output "${question// /_}.json"
done < queries.txt
```

---

## 🔧 Troubleshooting

### If wrapper scripts don't work:

```bash
# Check if executable
ls -l run_rag.sh
# Should show: -rwxr-xr-x

# Make executable if needed
chmod +x run_rag.sh run_ingest.sh

# Run directly
bash run_rag.sh --all-questions
```

### If Python packages missing:

```bash
# Install requirements
pip install -r requirements.txt

# Or install individually
pip install peft python-dotenv
```

### If GPU issues:

```bash
# Check GPU availability
nvidia-smi

# Use different GPU
export CUDA_DEVICE=2
./run_rag.sh --question "..."

# Use CPU
export CUDA_VISIBLE_DEVICES=""
./run_rag.sh --question "..."
```

---

## 📚 Documentation

- `README.md` - Complete system documentation
- `VENV_USAGE.md` - Virtual environment guide
- `docs/OPTIMAL_STRATEGY.md` - Chunking strategy deep dive
- `docs/CHUNKING_COMPARISON.md` - Chunker comparisons

---

## ✅ You're Ready!

Your system is fully operational. You can:

1. ✅ **Use it now** - Scripts work with current environment
2. 📦 **Set up venv later** - Optional but recommended for isolation
3. 🚀 **Run production** - Ready for full 42K paper ingestion

**Simplest command to get started:**
```bash
./run_rag.sh --all-questions
```

This will answer all 9 critical aging questions using your current setup! 🎉
