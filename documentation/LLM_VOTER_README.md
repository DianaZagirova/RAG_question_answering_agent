# LLM Voter - Combining RAG and Full-Text Evaluation Results

## Overview

The LLM Voter script combines evaluation results from two sources:
1. **Full-text evaluations** - Comprehensive analysis of complete papers
2. **RAG evaluations** - Targeted analysis using retrieval-augmented generation

## Strategy: RAG YES OVERRIDE

The combination strategy prioritizes precision over recall:

- **Baseline**: Use full-text answers (broad coverage)
- **Override**: When RAG says "Yes" for any question, trust RAG's answer
- **Rationale**: Leverages full-text's comprehensive coverage while trusting RAG's precision for positive findings

## Configuration

Add to `.env`:
```bash
# LLM Voter Configuration
FULLTEXT_EVAL_DB=/home/diana.z/hack/theories_extraction_agent/qa_results/qa_results.db
RAG_EVAL_DB=rag_results_fast.db
```

## Usage

### Basic Usage
```bash
./run_llm_voter.sh
```

### With Custom Paths
```bash
python scripts/llm_voter.py \
  --fulltext-db /path/to/qa_results.db \
  --rag-db /path/to/rag_results_fast.db \
  --questions data/questions_part2.json \
  --output combined_results.db
```

## Output Files

The script creates three output files:
1. **Database**: `combined_results.db` (SQLite)
2. **Short CSV**: `combined_results_short.csv`
3. **Extended CSV**: `combined_results_extended.csv`

### Database Tables

The database contains two tables:

### 1. `combined_answers_short`
One row per DOI with all answers in columns:
- `doi` (PRIMARY KEY)
- `aging_biomarker`
- `molecular_mechanism_of_aging`
- `longevity_intervention_to_test`
- `aging_cannot_be_reversed`
- `cross_species_longevity_biomarker`
- `naked_mole_rat_lifespan_explanation`
- `birds_lifespan_explanation`
- `large_animals_lifespan_explanation`
- `calorie_restriction_lifespan_explanation`
- `timestamp`

### 2. `combined_answers_extended`
Detailed view with one row per DOI-question pair:
- `id` (PRIMARY KEY)
- `doi`
- `question_key`
- `question_text`
- `answer`
- `confidence`
- `reasoning`
- `source` (either "rag" or "fulltext")
- `timestamp`

### CSV Files

#### 1. `*_short.csv`
One row per DOI with answers in columns:
- Header: `doi, aging_biomarker, molecular_mechanism_of_aging, ...`
- 18,167 rows (one per DOI)
- Easy to open in Excel/Google Sheets
- Ideal for quick overview and filtering

#### 2. `*_extended.csv`
One row per DOI-question pair with full details:
- Header: `doi, question_key, question_text, answer, confidence, reasoning, source`
- 163,503 rows (one per DOI-question combination)
- Includes confidence scores and reasoning
- Shows which source (RAG or full-text) provided each answer
- Ideal for detailed analysis and auditing

## Example Queries

### View short results
```bash
sqlite3 combined_results.db "SELECT * FROM combined_answers_short LIMIT 5"
```

### Find RAG overrides
```bash
sqlite3 combined_results.db \
  "SELECT doi, question_key, answer, confidence 
   FROM combined_answers_extended 
   WHERE source='rag' 
   LIMIT 10"
```

### Count answers by source
```bash
sqlite3 combined_results.db \
  "SELECT source, COUNT(*) as count 
   FROM combined_answers_extended 
   GROUP BY source"
```

### Find high-confidence RAG "Yes" answers
```bash
sqlite3 combined_results.db \
  "SELECT doi, question_key, answer, confidence, reasoning 
   FROM combined_answers_extended 
   WHERE source='rag' 
     AND answer LIKE '%Yes%' 
     AND confidence > 0.9 
   ORDER BY confidence DESC 
   LIMIT 20"
```

### Work with CSV files directly
```bash
# View short CSV in terminal
head -n 10 combined_results_short.csv

# Count RAG overrides in extended CSV
grep ",rag" combined_results_extended.csv | wc -l

# Open in Excel/LibreOffice
libreoffice combined_results_short.csv

# Load in Python
import pandas as pd
df_short = pd.read_csv('combined_results_short.csv')
df_extended = pd.read_csv('combined_results_extended.csv')

# Filter RAG answers with high confidence
df_rag = df_extended[
    (df_extended['source'] == 'rag') & 
    (df_extended['confidence'] > 0.9)
]
```

## Answer Validation

The script validates that all answers match the allowed options defined in `data/questions_part2.json`:

- **aging_biomarker**: "Yes, quantitatively shown" / "Yes, but not shown" / "No"
- **Other questions**: "Yes" / "No"

Invalid answers are flagged in the output with warnings.

## Statistics

After running, the script displays:
- Total DOIs processed
- DOIs with full-text only
- DOIs with RAG only
- DOIs with both sources
- Number of RAG overrides applied
- Validation warnings (if any)

## Example Output

```
======================================================================
LLM VOTER - Combining RAG and Full-Text Results
======================================================================
Strategy: RAG YES OVERRIDE
  → Full-text answers as baseline
  → RAG 'Yes' answers override full-text
======================================================================

📥 Loading data...
✓ Loaded full-text answers for 18166 DOIs
✓ Loaded RAG answers for 16320 DOIs

🔄 Combining answers...

📊 Combination Statistics:
  Total DOIs: 18167
  Full-text only: 1847
  RAG only: 1
  Both sources: 16319
  RAG overrides applied: 42537

💾 Creating output database...
✓ Created output database: combined_results.db
  - combined_answers_short: 18167 rows
  - combined_answers_extended: 163503 rows

======================================================================
✅ COMPLETE!
======================================================================
```

## Question Mapping

The script uses `data/questions_part2.json` to map question keys to full question text and validate answers. All 9 questions from the aging research domain are supported.
