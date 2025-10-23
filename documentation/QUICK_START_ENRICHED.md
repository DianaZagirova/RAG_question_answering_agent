# Quick Start - Enriched LLM Voter

## Run the Script

```bash
# With defaults from .env
python scripts/llm_voter.py --output combined_results_final.db

# Or use the shell script
./run_llm_voter.sh
```

## What You Get

**4 Output Files:**

1. `combined_results_final.db` - SQLite database (74 MB)
2. `combined_results_final_short.csv` - One row per paper (3.4 MB)
3. `combined_results_final_extended.csv` - One row per paper-question (80 MB)
4. `combined_results_final_theory_stats.csv` - Theory statistics (109 KB)

## CSV Columns

### Short CSV (15,813 papers)
```
doi | theory_id | theory | title | year | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 | Q9
```

### Extended CSV (142,317 rows)
```
doi | theory_id | theory | title | year | question_key | question_text | 
answer | confidence | reasoning | source | journal | date_published | 
cited_by_count | fwci | openalex_topic_name | keywords
```

### Theory Stats CSV (2,141 theories)
```
theory_name | doi_count
```

## Quick Analysis

### Python
```python
import pandas as pd

# Load data
df_short = pd.read_csv('combined_results_final_short.csv')
df_extended = pd.read_csv('combined_results_final_extended.csv')
df_theories = pd.read_csv('combined_results_final_theory_stats.csv')

# Top 10 theories by paper count
print(df_theories.head(10))

# High-impact papers (FWCI > 2, citations > 100)
high_impact = df_extended[
    (df_extended['fwci'] > 2.0) & 
    (df_extended['cited_by_count'] > 100)
].drop_duplicates('doi')

# RAG overrides with high confidence
rag_yes = df_extended[
    (df_extended['source'] == 'rag') & 
    (df_extended['answer'].str.contains('Yes')) &
    (df_extended['confidence'] > 0.9)
]

# Papers per theory per year
timeline = df_short.groupby(['theory_id', 'year']).size().reset_index(name='count')
```

### Excel
1. Open `combined_results_final_short.csv`
2. Apply filters to header row
3. Sort by `year` or `theory_id`
4. Create pivot table: theories vs questions

### Command Line
```bash
# Count papers per theory
cut -d',' -f2 combined_results_final_short.csv | sort | uniq -c | sort -rn

# Find specific theory papers
grep "T0001" combined_results_final_short.csv

# Extract high-confidence RAG answers
awk -F',' '$11=="rag" && $9>0.9' combined_results_final_extended.csv
```

## Key Features

✅ **Theory IDs:** Unique identifiers (T0001-T2141) for each theory  
✅ **Paper Metadata:** Title, year, journal, citations, FWCI, topics  
✅ **Original Questions:** Full question text from research protocol  
✅ **Source Tracking:** Know if answer came from RAG, full-text, or validation set  
✅ **Confidence Scores:** Assess answer reliability  
✅ **Empty Fields:** Missing data left empty (not N/A)

## Configuration

Edit `.env` to customize paths:
```bash
FULLTEXT_EVAL_DB=/path/to/qa_results.db
RAG_EVAL_DB=rag_results_fast.db
THEORY_MAPPING=/path/to/final_theory_to_dois_mapping.json
PAPERS_DB=/path/to/papers.db
```

## Help

```bash
python scripts/llm_voter.py --help
```

## Documentation

- `CSV_FORMAT_DOCUMENTATION.md` - Detailed format specification
- `LLM_VOTER_USAGE.md` - Complete usage guide
- `PRODUCTION_RUN_EXAMPLE.md` - Step-by-step production run

## Example Output

```
✓ Loaded 9 question mappings
✓ Loaded 9 original question mappings
✓ Loaded 2141 theories with 15813 DOIs
✓ Papers DB: /path/to/papers.db

📊 Combination Statistics:
  Total theory DOIs: 15,813
  Processed DOIs: 15,813
  Validation set used: 22
  RAG overrides applied: 38,483

✓ Created short CSV: combined_results_final_short.csv (15813 rows)
✓ Created extended CSV: combined_results_final_extended.csv (142317 rows)
✓ Created theory statistics CSV: combined_results_final_theory_stats.csv (2141 theories)
```

## Top 5 Theories

1. **T0001** - Mitochondrial ROS-Induced Free Radical Theory (261 papers)
2. **T0002** - Mitochondrial ROS-Induced Oxidative Stress Theory (114 papers)
3. **T0003** - Somatic DNA Damage Theory (112 papers)
4. **T0004** - Mitochondrial ROS-Induced Mitochondrial Decline Theory (84 papers)
5. **T0005** - Insulin/IGF-1 Signaling Disposable Soma Theory (72 papers)
