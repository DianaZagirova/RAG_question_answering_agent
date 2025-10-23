# Batch Processing Guide - RAG on All Papers

## Overview

This system processes all validated papers from your database through the RAG system, answering all questions for each paper and storing results incrementally.

---

## Features

✅ **Incremental Processing**: Resume from where it left off  
✅ **Abstract Inclusion**: Paper abstract added to LLM context  
✅ **Database Storage**: Results stored in SQLite with full metadata  
✅ **Progress Tracking**: Real-time progress with tqdm  
✅ **Error Handling**: Errors logged, processing continues  
✅ **Predefined Queries**: Uses optimized queries from JSON file  

---

## Quick Start

### 1. Run Batch Processing

```bash
python scripts/run_rag_on_all_papers.py \
  --evaluations-db /path/to/evaluations.db \
  --papers-db /path/to/papers.db \
  --results-db rag_results.db \
  --limit 10  # Optional: test with 10 papers first
```

### 2. Analyze Results

```bash
python scripts/analyze_rag_results.py \
  --results-db rag_results.db \
  --export-json results.json \
  --export-csv results.csv
```

---

## Database Schema

### Input Databases

**evaluations.db** (source):
```sql
paper_evaluations (
  doi TEXT,
  pmid TEXT,
  title TEXT,
  result TEXT,  -- 'valid', 'doubted', 'not_valid'
  confidence_score INTEGER
)
```

**papers.db** (source):
```sql
papers (
  doi TEXT,
  pmid TEXT,
  title TEXT,
  abstract TEXT,
  full_text TEXT,
  full_text_sections TEXT
)
```

### Output Database

**rag_results.db** (created):

```sql
-- Paper metadata
paper_metadata (
  doi TEXT PRIMARY KEY,
  pmid TEXT,
  title TEXT,
  abstract TEXT,
  validation_result TEXT,
  confidence_score INTEGER,
  used_full_text BOOLEAN,
  n_chunks_retrieved INTEGER,
  timestamp TEXT
)

-- Question answers
paper_answers (
  id INTEGER PRIMARY KEY,
  doi TEXT,
  question_key TEXT,
  question_text TEXT,
  answer TEXT,
  confidence REAL,
  reasoning TEXT,
  parse_error BOOLEAN,
  n_sources INTEGER,
  UNIQUE(doi, question_key)
)

-- Processing log
processing_log (
  id INTEGER PRIMARY KEY,
  doi TEXT,
  status TEXT,  -- 'success' or 'error'
  error_message TEXT,
  timestamp TEXT
)
```

---

## Paper Selection Criteria

Papers are selected from `evaluations.db` if they meet ANY of:
- `result = 'valid'`
- `result = 'doubted'`
- `result = 'not_valid' AND confidence_score <= 7`

This ensures we process papers that are likely relevant to aging research.

---

## Processing Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. LOAD VALIDATED PAPERS                                    │
│    - Query evaluations.db for validated DOIs               │
│    - Join with papers.db to get full text + abstract       │
│    - Check rag_results.db for already processed papers     │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. FOR EACH PAPER (with progress bar)                       │
│                                                              │
│    For each question:                                       │
│      a. Load predefined queries for question key           │
│      b. Retrieve top N unique chunks (12 per query)        │
│      c. Add paper abstract to context                      │
│      d. Generate answer with LLM                           │
│      e. Parse structured JSON response                     │
│                                                              │
│    Save results to database immediately                     │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. STORE RESULTS                                             │
│    - Insert/update paper_metadata                           │
│    - Insert all answers for paper                           │
│    - Log success/error in processing_log                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Context Construction

For each question, the LLM receives:

```
======================================================================
PAPER ABSTRACT
======================================================================
<paper abstract text>
======================================================================

[Source 1] (DOI: 10.xxx, Section: Results)
<retrieved chunk 1>

[Source 2] (DOI: 10.xxx, Section: Discussion)
<retrieved chunk 2>

...

[Source N] (DOI: 10.xxx, Section: Methods)
<retrieved chunk N>
```

**Why include abstract?**
- Provides high-level overview of paper
- Helps LLM understand context
- Improves answer quality for broad questions

---

## Command-Line Options

### Required Arguments

```bash
--evaluations-db PATH    # Path to evaluations.db
--papers-db PATH         # Path to papers.db
```

### Optional Arguments

```bash
--results-db PATH        # Output database (default: rag_results.db)
--questions-file PATH    # Questions JSON (default: data/questions_part2.json)
--predefined-queries PATH # Queries JSON (default: data/queries_extended.json)
--limit N                # Process only N papers (for testing)
--temperature FLOAT      # LLM temperature (default: 0.2)
--max-tokens INT         # Max tokens per answer (default: 300)
--collection-name STR    # ChromaDB collection name
--persist-dir PATH       # ChromaDB persistence directory
```

---

## Examples

### Test with 10 Papers

```bash
python scripts/run_rag_on_all_papers.py \
  --evaluations-db ../theories_extraction_agent/data/evaluations.db \
  --papers-db ../theories_extraction_agent/data/papers.db \
  --results-db test_results.db \
  --limit 10
```

### Process All Papers

```bash
python scripts/run_rag_on_all_papers.py \
  --evaluations-db ../theories_extraction_agent/data/evaluations.db \
  --papers-db ../theories_extraction_agent/data/papers.db \
  --results-db rag_results.db
```

### Resume Processing

If processing was interrupted, simply run the same command again:

```bash
python scripts/run_rag_on_all_papers.py \
  --evaluations-db ../theories_extraction_agent/data/evaluations.db \
  --papers-db ../theories_extraction_agent/data/papers.db \
  --results-db rag_results.db
```

The system automatically skips papers already in `rag_results.db`.

---

## Analysis Commands

### View Statistics

```bash
python scripts/analyze_rag_results.py \
  --results-db rag_results.db
```

**Output:**
```
📊 Overall Statistics:
  Total papers processed: 1500
  Papers with full text: 1200 (80.0%)
  Total answers: 13500
  Successful answers: 12800 (94.8%)
  Parse errors: 700 (5.2%)

📋 Answer Distribution by Question:
  aging_biomarker:
    No: 800
    Yes, but not shown: 500
    Yes, quantitatively shown: 200

🎯 Average Confidence by Question:
  aging_biomarker: 0.87 (n=1500)
  molecular_mechanism_of_aging: 0.92 (n=1500)
  ...
```

### Export to JSON

```bash
python scripts/analyze_rag_results.py \
  --results-db rag_results.db \
  --export-json results.json
```

**Format:**
```json
[
  {
    "doi": "10.1089/ars.2012.5111",
    "pmid": "23025434",
    "title": "Oxidative stress and aging...",
    "abstract": "...",
    "validation_result": "valid",
    "confidence_score": 9,
    "used_full_text": true,
    "n_chunks_retrieved": 90,
    "timestamp": "2025-10-20T14:30:00",
    "answers": {
      "aging_biomarker": {
        "answer": "Yes, but not shown",
        "confidence": 0.85,
        "reasoning": "The paper discusses...",
        "parse_error": false,
        "n_sources": 10
      },
      ...
    }
  },
  ...
]
```

### Export to CSV

```bash
python scripts/analyze_rag_results.py \
  --results-db rag_results.db \
  --export-csv results.csv
```

**Format:** Flattened table with one row per (paper, question) pair.

---

## Performance Considerations

### Processing Speed

- **Per paper**: ~30-60 seconds (9 questions)
- **Per question**: ~3-7 seconds
- **1000 papers**: ~8-16 hours

### Rate Limiting

Simple delay between papers (0.5 seconds) to avoid overwhelming the API.

For production, consider:
- Async processing
- Batch API calls
- More sophisticated rate limiting

### Memory Usage

- Minimal (processes one paper at a time)
- Database writes are incremental
- Safe to interrupt and resume

---

## Error Handling

### Automatic Recovery

- **Parse errors**: Logged, processing continues
- **API errors**: Logged, processing continues
- **Database errors**: Logged, processing continues

### Error Logs

All errors stored in `processing_log` table:

```sql
SELECT doi, error_message, timestamp
FROM processing_log
WHERE status = 'error'
ORDER BY timestamp DESC;
```

### Retry Failed Papers

To retry papers that failed:

```sql
-- Delete failed papers from results
DELETE FROM paper_metadata
WHERE doi IN (
  SELECT doi FROM processing_log WHERE status = 'error'
);

DELETE FROM paper_answers
WHERE doi IN (
  SELECT doi FROM processing_log WHERE status = 'error'
);
```

Then run the batch script again.

---

## Monitoring Progress

### During Processing

```
[150/1500] 10.1089/ars.2012.5111
🔍 Retrieving top 12 unique chunks using 2 predefined queries...
🤖 Generating answer with gpt-4.1...
✓ Answered 9/9 questions
```

### Check Database

```bash
sqlite3 rag_results.db "SELECT COUNT(*) FROM paper_metadata"
sqlite3 rag_results.db "SELECT COUNT(*) FROM paper_answers"
```

### View Recent Results

```sql
SELECT doi, timestamp, n_chunks_retrieved
FROM paper_metadata
ORDER BY timestamp DESC
LIMIT 10;
```

---

## Validation Against Ground Truth

If you have ground truth answers (e.g., from `qa_validation_set_extended.json`):

```python
import sqlite3
import json

# Load ground truth
with open('data/qa_validation_set_extended.json') as f:
    ground_truth = {p['doi']: p for p in json.load(f)}

# Compare with RAG results
conn = sqlite3.connect('rag_results.db')
cur = conn.cursor()

for doi, gt in ground_truth.items():
    cur.execute("""
        SELECT question_key, answer
        FROM paper_answers
        WHERE doi = ?
    """, (doi,))
    
    rag_answers = dict(cur.fetchall())
    
    # Compare answers
    for q_key, gt_answer in gt.items():
        if q_key.startswith('Q'):
            rag_answer = rag_answers.get(q_key)
            if rag_answer == gt_answer:
                print(f"✓ {doi} {q_key}")
            else:
                print(f"✗ {doi} {q_key}: {rag_answer} != {gt_answer}")
```

---

## Troubleshooting

### Issue: "No papers to process"

**Cause**: All papers already processed or no validated papers found.

**Solution**:
```bash
# Check validated papers
sqlite3 evaluations.db "SELECT COUNT(*) FROM paper_evaluations WHERE result='valid'"

# Check already processed
sqlite3 rag_results.db "SELECT COUNT(*) FROM paper_metadata"
```

### Issue: High parse error rate

**Cause**: LLM not returning valid JSON.

**Solution**:
- Check LLM model (gpt-4.1 recommended)
- Increase max_tokens if responses are truncated
- Review prompt in `src/core/llm_integration.py`

### Issue: Low confidence scores

**Cause**: Retrieved chunks not relevant.

**Solution**:
- Check if papers are in ChromaDB
- Verify predefined queries are appropriate
- Increase n_results for better coverage

---

## Files Created

```
scripts/
├── run_rag_on_all_papers.py      # Main batch processing script
└── analyze_rag_results.py        # Results analysis script

BATCH_PROCESSING_GUIDE.md         # This guide
```

---

## Next Steps

1. **Test with small batch**: Use `--limit 10` to verify setup
2. **Run full processing**: Process all validated papers
3. **Analyze results**: Use analysis script to review
4. **Export data**: Generate JSON/CSV for further analysis
5. **Validate accuracy**: Compare with ground truth if available

---

## Summary

**To process all papers:**
```bash
python scripts/run_rag_on_all_papers.py \
  --evaluations-db /path/to/evaluations.db \
  --papers-db /path/to/papers.db \
  --results-db rag_results.db
```

**To analyze results:**
```bash
python scripts/analyze_rag_results.py \
  --results-db rag_results.db \
  --export-json results.json
```

The system handles everything automatically:
- ✅ Loads validated papers
- ✅ Retrieves with predefined queries
- ✅ Includes abstract in context
- ✅ Stores results incrementally
- ✅ Handles errors gracefully
- ✅ Resumes from interruptions

Ready to process thousands of papers! 🚀
