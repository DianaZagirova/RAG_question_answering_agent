# LLM Voter - Production Usage Guide

## Overview

The LLM Voter combines RAG and full-text evaluation results using the **RAG YES OVERRIDE** strategy, filtered by theory mapping and enriched with theory names.

## Quick Start

### Basic Usage (with defaults from .env)
```bash
./run_llm_voter.sh
```

### Production Run
```bash
python scripts/llm_voter.py --output combined_results_final.db
```

### Custom Paths
```bash
python scripts/llm_voter.py \
  --fulltext-db /path/to/qa_results.db \
  --rag-db /path/to/rag_results_fast.db \
  --theory-mapping /path/to/final_theory_to_dois_mapping.json \
  --validation-set data/validation_set_qa \
  --questions data/questions_part2.json \
  --output combined_results.db
```

## Configuration

Add to `.env`:
```bash
# LLM Voter Configuration
FULLTEXT_EVAL_DB=/home/diana.z/hack/theories_extraction_agent/qa_results/qa_results.db
RAG_EVAL_DB=rag_results_fast.db
THEORY_MAPPING=/home/diana.z/hack/theories_extraction_agent/output/final_output/final_theory_to_dois_mapping.json
```

## Processing Logic

### 1. Theory Filtering
- **Only processes DOIs** from `final_theory_to_dois_mapping.json`
- Ensures all results are theory-mapped

### 2. Data Sources Priority
1. **Full-text evaluations** (baseline - broad coverage)
2. **RAG evaluations** (override when "Yes" - high precision)
3. **Validation set** (fallback for missing DOIs)

### 3. RAG YES OVERRIDE Strategy
```
IF RAG says "Yes" for any question:
    → Use RAG answer (high precision for positive findings)
ELSE:
    → Use full-text answer (comprehensive baseline)
```

### 4. Missing DOI Handling
```
IF DOI in theory mapping BUT missing from both DBs:
    → Try validation set (data/validation_set_qa/*.json)
    → If found: Use validation set data
    → If not found: Print WARNING and skip
```

## Output Files

### 1. Database: `combined_results_final.db`

**Three tables:**

#### `combined_answers_short`
- One row per DOI
- Columns: doi, theory, [9 question answers], timestamp

#### `combined_answers_extended`
- One row per DOI-question pair
- Columns: doi, theory, question_key, question_text, answer, confidence, reasoning, source, timestamp

#### `theory_statistics`
- One row per theory
- Columns: theory_name, doi_count

### 2. CSV Files

#### `*_short.csv` (15,821 rows)
```csv
doi,theory,aging_biomarker,molecular_mechanism_of_aging,...
10.1001/...,Mitochondrial ROS Theory,Yes,Yes,...
```

#### `*_extended.csv` (142,389 rows)
```csv
doi,theory,question_key,question_text,answer,confidence,reasoning,source
10.1001/...,Mitochondrial ROS Theory,aging_biomarker,"Does it...",Yes,0.95,"The paper...",rag
```

#### `*_theory_stats.csv` (2,141 theories)
```csv
theory_name,doi_count
Mitochondrial ROS-Induced Free Radical Theory,261
Mitochondrial ROS-Induced Oxidative Stress Theory,114
```

## Example Queries

### Database Queries

```bash
# View short results with theory
sqlite3 combined_results_final.db \
  "SELECT doi, theory, aging_biomarker, molecular_mechanism_of_aging 
   FROM combined_answers_short 
   LIMIT 5"

# Top theories by DOI count
sqlite3 combined_results_final.db \
  "SELECT * FROM theory_statistics 
   ORDER BY doi_count DESC 
   LIMIT 10"

# Find RAG overrides for specific theory
sqlite3 combined_results_final.db \
  "SELECT doi, question_key, answer, confidence 
   FROM combined_answers_extended 
   WHERE theory = 'Mitochondrial ROS-Induced Free Radical Theory' 
     AND source = 'rag' 
   LIMIT 10"

# Count answers by source
sqlite3 combined_results_final.db \
  "SELECT source, COUNT(*) as count 
   FROM combined_answers_extended 
   GROUP BY source"

# Export specific theory to CSV
sqlite3 -header -csv combined_results_final.db \
  "SELECT * FROM combined_answers_extended 
   WHERE theory = 'Mitochondrial ROS-Induced Free Radical Theory'" \
  > mitochondrial_ros_theory.csv
```

### Python Analysis

```python
import pandas as pd
import sqlite3

# Load data
conn = sqlite3.connect('combined_results_final.db')
df_short = pd.read_sql("SELECT * FROM combined_answers_short", conn)
df_extended = pd.read_sql("SELECT * FROM combined_answers_extended", conn)
df_theories = pd.read_sql("SELECT * FROM theory_statistics", conn)

# Analyze by theory
theory_analysis = df_extended.groupby('theory').agg({
    'answer': lambda x: (x.str.contains('Yes', case=False)).sum(),
    'confidence': 'mean',
    'source': lambda x: (x == 'rag').sum()
})

# Find high-confidence RAG "Yes" answers
rag_yes = df_extended[
    (df_extended['source'] == 'rag') & 
    (df_extended['answer'].str.contains('Yes', case=False)) &
    (df_extended['confidence'] > 0.9)
]

# Top theories
print(df_theories.sort_values('doi_count', ascending=False).head(10))
```

### CSV Analysis

```bash
# Count DOIs per theory
cut -d',' -f2 combined_results_final_short.csv | sort | uniq -c | sort -rn | head -10

# Find specific theory papers
grep "Mitochondrial ROS" combined_results_final_short.csv | wc -l

# Extract validation set usage
grep "validation_set" combined_results_final_extended.csv | cut -d',' -f1 | sort -u
```

## Statistics Output

After running, you'll see:
```
📊 Combination Statistics:
  Total theory DOIs: 15,821
  Processed DOIs: 15,821
  Full-text only: 1,527
  RAG only: 1
  Both sources: 14,293
  Validation set used: 22
  Missing DOIs: 0
  RAG overrides applied: 38,484
```

## Validation Set Format

Validation set files: `data/validation_set_qa/{doi_with_underscores}.json`

Example: `10.1007_BF02008340.json`
```json
{
  "aging_biomarker": {
    "answer": "Yes, quantitatively shown",
    "confidence": 1,
    "reasoning": "The paper presents quantitative data..."
  },
  "molecular_mechanism_of_aging": {
    "answer": "Yes",
    "confidence": 0.9,
    "reasoning": "The paper discusses..."
  }
}
```

## Troubleshooting

### Missing DOIs Warning
```
⚠️  WARNING: DOI 10.xxxx/yyyy missing from all sources (fulltext, RAG, validation set)
```
**Solution:** Add the DOI's answers to validation set or check if it should be in theory mapping.

### File Not Found Errors
**Check paths in `.env`:**
- FULLTEXT_EVAL_DB
- RAG_EVAL_DB  
- THEORY_MAPPING

### Empty Results
**Verify:**
1. Theory mapping file has DOIs
2. At least one source DB has data
3. Question keys match between sources

## Production Checklist

Before running in production:

- [ ] Verify all paths in `.env` are correct
- [ ] Check theory mapping file exists and has data
- [ ] Ensure validation set directory exists
- [ ] Confirm both evaluation DBs are accessible
- [ ] Test with `--help` to verify configuration
- [ ] Run with small output first to verify
- [ ] Check disk space for output files (~100MB+)

## Performance

- **Processing time:** ~30-60 seconds for 15,821 DOIs
- **Output size:** 
  - Database: ~80MB
  - Short CSV: ~1.5MB
  - Extended CSV: ~50MB
  - Theory stats CSV: ~100KB

## Version History

- **v2.0** (2025-10-22): Added theory filtering, validation set fallback, theory enrichment, theory statistics table
- **v1.0** (2025-10-22): Initial RAG YES OVERRIDE implementation
