# 📚 Production Ingestion Guide

## Database Analysis Results

### Overview
- **Total papers in database:** 54,760
- **Papers with usable text:** 43,637 (79.7%)
- **Papers to ingest:** ~43,637

### Text Format Distribution
- **full_text_sections (JSON Dict):** 99.5% of cases
- **full_text_sections (JSON List):** 0.5% of cases  
- **full_text only:** 46.9% of papers also have this
- **Preprocessor handles:** ✅ All formats (dict, list, plain text)

### Common Section Names
Most frequent sections in papers:
1. Abstract (367/500 samples)
2. Introduction (166/500)
3. Discussion (115/500)
4. Results (58/500)
5. Conclusion (71/500)

### What Gets Ingested
✅ **Included:**
- Papers with `full_text_sections` (prioritized)
- Papers with `full_text` (fallback)
- All standard sections (Abstract, Introduction, Methods, Results, Discussion, Conclusion)

❌ **Excluded:**
- Papers with neither field populated (11,123 papers, 20.3%)
- Reference sections
- Acknowledgments, Funding, Author Contributions
- Papers with <100 characters after preprocessing

---

## Quick Start

### 1. Inspect Database First (Optional)

```bash
python scripts/inspect_database.py
```

**Output:**
- Total papers available
- Format distribution
- Section names
- Recommendations

### 2. Test with Small Sample

```bash
# Test with 100 papers (takes ~2-3 minutes)
CUDA_VISIBLE_DEVICES=3 python scripts/run_full_ingestion.py --limit 100 --reset

# Test with 1000 papers (takes ~20-25 minutes)
CUDA_VISIBLE_DEVICES=3 python scripts/run_full_ingestion.py --limit 1000 --reset
```

### 3. Run Full Production Ingestion

```bash
# All ~43,637 papers (takes ~18-24 hours)
CUDA_VISIBLE_DEVICES=3 python scripts/run_full_ingestion.py --reset
```

---

## Detailed Instructions

### Option 1: Using Wrapper Script (Easiest)

```bash
# The wrapper handles environment activation
./run_ingest.sh --reset
```

### Option 2: Using Direct Python (More Control)

```bash
# Activate environment if using one
conda activate rag_agent  # or: source venv/bin/activate

# Set GPU device
export CUDA_VISIBLE_DEVICES=3

# Run ingestion
python scripts/run_full_ingestion.py --reset
```

### Option 3: Dry Run (Check Configuration)

```bash
# See what would happen without actually ingesting
python scripts/run_full_ingestion.py --dry-run
```

**Output:**
```
📊 Configuration:
  Database: /home/diana.z/hack/download_papers_pubmed/...
  Collection: scientific_papers_optimal
  Chunk Size: 1500
  Chunk Overlap: 300
  ...

📚 Papers:
  Available in database: 43,637
  Will process: 43,637
```

---

## Command Reference

### Test Runs

```bash
# Small test (100 papers)
CUDA_VISIBLE_DEVICES=3 python scripts/run_full_ingestion.py --limit 100 --reset

# Medium test (1000 papers)
CUDA_VISIBLE_DEVICES=3 python scripts/run_full_ingestion.py --limit 1000 --reset

# Large test (10000 papers)
CUDA_VISIBLE_DEVICES=3 python scripts/run_full_ingestion.py --limit 10000 --reset
```

### Production Runs

```bash
# Full ingestion (new collection)
CUDA_VISIBLE_DEVICES=3 python scripts/run_full_ingestion.py --reset

# Continue interrupted ingestion (don't reset)
CUDA_VISIBLE_DEVICES=3 python scripts/run_full_ingestion.py

# Use CPU instead of GPU (slower)
CUDA_VISIBLE_DEVICES="" python scripts/run_full_ingestion.py --reset
```

---

## Time & Resource Estimates

### Processing Speed

**GPU (Device 3):**
- ~40-50 papers/minute
- ~2,400-3,000 papers/hour

**CPU:**
- ~10-15 papers/minute  
- ~600-900 papers/hour

### Time Estimates for Full Ingestion (43,637 papers)

| Setup | Papers/min | Total Time |
|-------|-----------|------------|
| **GPU (recommended)** | 40-50 | 14-18 hours |
| **CPU** | 10-15 | 48-72 hours |

### Storage Requirements

- **Vector database:** ~8-10 GB
- **Metadata:** ~500 MB
- **Total:** ~10 GB

### Memory Requirements

- **RAM:** ~16 GB recommended
- **GPU VRAM:** ~4-6 GB
- **Disk space:** ~15 GB free (for safety)

---

## Progress Monitoring

### During Ingestion

The script shows real-time progress:

```
Processing papers: 1000it [00:20, 48.5it/s, processed=950, skipped=50, chunks=25650]
Processing papers: 2000it [00:41, 48.2it/s, processed=1900, skipped=100, chunks=51300]
...
```

**Legend:**
- `1000it` = 1000 papers checked
- `[00:20]` = 20 seconds elapsed
- `48.5it/s` = 48.5 papers/second
- `processed=950` = 950 successfully ingested
- `skipped=50` = 50 skipped (no valid text)
- `chunks=25650` = 25,650 chunks created

### After Completion

```
INGESTION COMPLETE!
========================================

📊 Final Statistics:
  Papers processed: 43,250
  Papers skipped: 387
  Total chunks created: 1,167,750
  Average chunks/paper: 27.0

⏱️  Time:
  Total time: 18h 24m 35s
  Processing rate: 39.2 papers/minute

💾 Database:
  Collection: scientific_papers_optimal
  Total chunks: 1,167,750
  Embedding model: all-mpnet-base-v2
  Location: ./chroma_db_optimal
```

---

## Handling Interruptions

### If Ingestion is Interrupted

**DON'T PANIC!** Your progress is saved.

**Option 1: Continue from where you left off**
```bash
# Run without --reset to continue
CUDA_VISIBLE_DEVICES=3 python scripts/run_full_ingestion.py
```

**Option 2: Start fresh**
```bash
# Use --reset to start over
CUDA_VISIBLE_DEVICES=3 python scripts/run_full_ingestion.py --reset
```

### Check Current Progress

```python
python -c "
from src.core.rag_system import ScientificRAG
rag = ScientificRAG(collection_name='scientific_papers_optimal')
stats = rag.get_statistics()
print(f'Current chunks: {stats[\"total_chunks\"]:,}')
print(f'Estimated papers: {stats[\"total_chunks\"] / 27:,.0f}')
"
```

---

## Troubleshooting

### Issue: GPU Out of Memory

**Error:** `CUDA out of memory`

**Solution 1:** Use a different GPU
```bash
# Try GPU 2 or 1
CUDA_VISIBLE_DEVICES=2 python scripts/run_full_ingestion.py --reset
```

**Solution 2:** Use CPU (slower)
```bash
CUDA_VISIBLE_DEVICES="" python scripts/run_full_ingestion.py --reset
```

**Solution 3:** Process in smaller batches
```bash
# Process 10,000 at a time
CUDA_VISIBLE_DEVICES=3 python scripts/run_full_ingestion.py --limit 10000 --reset
# Wait for completion
CUDA_VISIBLE_DEVICES=3 python scripts/run_full_ingestion.py --limit 20000
# Repeat...
```

### Issue: Database Locked

**Error:** `database is locked`

**Cause:** Another process is using the database

**Solution:**
```bash
# Check for other processes
ps aux | grep python

# Kill if needed
kill <process_id>

# Or wait and try again
```

### Issue: Disk Space

**Error:** `No space left on device`

**Check space:**
```bash
df -h .
```

**Solutions:**
- Free up space: `rm -rf old_data/`
- Use different location: Edit `PERSIST_DIR` in `.env`

### Issue: Slow Processing

**If processing is slower than expected:**

1. **Check GPU usage:**
```bash
nvidia-smi
```

2. **Check CPU usage:**
```bash
top
```

3. **Check if using GPU:**
```bash
# Should see GPU activity
watch -n 1 nvidia-smi
```

4. **Optimize batch size:** Already optimized in code

---

## Verification After Ingestion

### 1. Check Statistics

```bash
python -c "
from src.core.rag_system import ScientificRAG
rag = ScientificRAG(
    collection_name='scientific_papers_optimal',
    persist_directory='./chroma_db_optimal'
)
stats = rag.get_statistics()
print(f'Total chunks: {stats[\"total_chunks\"]:,}')
print(f'Embedding model: {stats[\"embedding_model\"]}')
print(f'Collection: {stats[\"collection_name\"]}')
"
```

**Expected output:**
```
Total chunks: ~1,167,750
Embedding model: all-mpnet-base-v2
Collection: scientific_papers_optimal
```

### 2. Test a Query

```bash
./run_rag.sh --question "Does the paper suggest an aging biomarker?"
```

**Expected:** 
- Should retrieve relevant chunks
- Should generate answer with sources

### 3. Run Full Test Suite

```bash
python tests/test_system.py
```

**Expected:**
```
All tests passed: 7/7 ✅
```

---

## Production Checklist

Before running full ingestion:

- [ ] Database path correct in `.env`
- [ ] GPU device set (default: 3)
- [ ] Enough disk space (~15 GB free)
- [ ] Test run completed successfully
- [ ] Virtual environment activated (if using)
- [ ] Backup important data (if overwriting)

Run ingestion:

- [ ] `--reset` flag if starting fresh
- [ ] Monitor progress for first hour
- [ ] Check GPU usage with `nvidia-smi`
- [ ] Have patience (14-18 hours on GPU)

After completion:

- [ ] Verify statistics
- [ ] Test with sample query
- [ ] Run test suite
- [ ] Answer all 9 questions
- [ ] Save results

---

## Expected Results

### For 100 Papers (Test)

```
Papers processed: 100
Total chunks: ~2,700
Time: ~2-3 minutes (GPU)
Storage: ~15 MB
```

### For 1,000 Papers (Validation)

```
Papers processed: 1,000
Total chunks: ~27,000
Time: ~20-25 minutes (GPU)
Storage: ~150 MB
```

### For All ~43,637 Papers (Production)

```
Papers processed: ~43,250
Total chunks: ~1,167,750
Time: ~14-18 hours (GPU)
Storage: ~8-10 GB
```

---

## What Happens During Ingestion

1. **Database Connection**
   - Opens SQLite database
   - Counts available papers

2. **Text Extraction** (Per Paper)
   - Load `full_text_sections` (priority)
   - Parse JSON (dict or list format)
   - Concatenate sections in order
   - Fall back to `full_text` if needed

3. **Preprocessing**
   - Remove reference sections
   - Clean special characters
   - Normalize whitespace
   - Check minimum length (100 chars)

4. **Chunking**
   - Use NLTK sentence tokenizer
   - Create 1500-char chunks
   - Add 300-char overlap
   - Preserve section boundaries
   - Create ~27 chunks per paper

5. **Embedding**
   - Generate embeddings with mpnet
   - Batch processing for efficiency
   - GPU acceleration if available

6. **Storage**
   - Add to ChromaDB collection
   - Include metadata (DOI, title, section, etc.)
   - Persist to disk

7. **Progress Tracking**
   - Update every 100 papers
   - Save statistics
   - Write to JSON file

---

## Configuration Details

All settings are in `.env`:

```bash
# Database
DB_PATH=/home/diana.z/hack/download_papers_pubmed/paper_collection/data/papers.db

# Collection
COLLECTION_NAME=scientific_papers_optimal
PERSIST_DIR=./chroma_db_optimal

# Chunking
CHUNK_SIZE=1500      # 50% larger than standard
CHUNK_OVERLAP=300    # 50% more overlap
MIN_CHUNK_SIZE=200   # Minimum to keep

# Embeddings
EMBEDDING_MODEL=allenai/specter2           # Try first
BACKUP_EMBEDDING_MODEL=all-mpnet-base-v2   # Fallback (current)

# Hardware
CUDA_DEVICE=3  # GPU device number
```

**To modify:** Edit `.env` file, then re-run ingestion with `--reset`

---

## Next Steps After Ingestion

### 1. Answer Single Question

```bash
./run_rag.sh --question "Does the paper suggest an aging biomarker that is quantitatively shown?"
```

### 2. Answer All 9 Critical Questions

```bash
./run_rag.sh --all-questions --output final_answers.json
```

### 3. Review Results

```bash
# View JSON results
cat final_answers.json | jq '.Q1.answer'

# Or open in editor
nano final_answers.json
```

### 4. Iterate and Improve

- Adjust `temperature` for more/less creativity
- Change `n_results` for more/less context
- Try different question phrasings
- Fine-tune prompts in `llm_integration.py`

---

## Summary

**To run full production ingestion:**

```bash
cd /home/diana.z/hack/rag_agent

# Activate environment (if using)
conda activate rag_agent

# Run ingestion
CUDA_VISIBLE_DEVICES=3 python scripts/run_full_ingestion.py --reset
```

**Expected:**
- Duration: 14-18 hours
- Papers: ~43,250
- Chunks: ~1,167,750
- Storage: ~8-10 GB

**After completion:**
```bash
./run_rag.sh --all-questions
```

**You'll have answers to all 9 critical aging questions!** 🎉
