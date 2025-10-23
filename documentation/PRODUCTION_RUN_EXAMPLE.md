# Production Run Example - LLM Voter

## Complete Production Run

### Step 1: Verify Configuration

```bash
# Check .env settings
cat .env | grep -A 3 "LLM Voter"
```

Expected output:
```
# LLM Voter Configuration
FULLTEXT_EVAL_DB=/home/diana.z/hack/theories_extraction_agent/qa_results/qa_results.db
RAG_EVAL_DB=rag_results_fast.db
THEORY_MAPPING=/home/diana.z/hack/theories_extraction_agent/output/final_output/final_theory_to_dois_mapping.json
```

### Step 2: Run the Script

```bash
# Simple run with defaults
./run_llm_voter.sh

# OR with explicit output name
python scripts/llm_voter.py --output combined_results_final.db
```

### Step 3: Expected Output

```
✓ Loaded 9 question mappings
✓ Loaded 2141 theories with 15821 DOIs
✓ Full-text DB: /home/diana.z/hack/theories_extraction_agent/qa_results/qa_results.db
✓ RAG DB: rag_results_fast.db
✓ Validation set: data/validation_set_qa
✓ Output DB: combined_results_final.db

======================================================================
LLM VOTER - Combining RAG and Full-Text Results
======================================================================
Strategy: RAG YES OVERRIDE
  → Full-text answers as baseline
  → RAG 'Yes' answers override full-text
======================================================================

📥 Loading data...
✓ Loaded full-text answers for 18166 DOIs
✓ Loaded RAG answers for 16321 DOIs

🔄 Combining answers...

📊 Combination Statistics:
  Total theory DOIs: 15,821
  Processed DOIs: 15,821
  Full-text only: 1,527
  RAG only: 1
  Both sources: 14,293
  Validation set used: 22
  Missing DOIs: 0
  RAG overrides applied: 38,484

💾 Creating output database...
✓ Created output database: combined_results_final.db
  - combined_answers_short: 15821 rows
  - combined_answers_extended: 142389 rows
✓ Created theory_statistics table with 2141 theories

📄 Exporting to CSV...
✓ Created short CSV: combined_results_final_short.csv
  - 15821 rows
✓ Created extended CSV: combined_results_final_extended.csv
  - 142389 rows
✓ Created theory statistics CSV: combined_results_final_theory_stats.csv
  - 2141 theories

======================================================================
✅ COMPLETE!
======================================================================

Output files:
  Database: combined_results_final.db
  Short CSV: combined_results_final_short.csv
  Extended CSV: combined_results_final_extended.csv
  Theory Stats CSV: combined_results_final_theory_stats.csv
```

### Step 4: Verify Output Files

```bash
# Check files were created
ls -lh combined_results_final*

# Expected:
# -rw-r--r-- 1 user user  80M combined_results_final.db
# -rw-r--r-- 1 user user  50M combined_results_final_extended.csv
# -rw-r--r-- 1 user user 1.5M combined_results_final_short.csv
# -rw-r--r-- 1 user user 100K combined_results_final_theory_stats.csv
```

### Step 5: Quick Validation

```bash
# Check row counts
wc -l combined_results_final_*.csv

# Expected:
#    15822 combined_results_final_short.csv (15821 + header)
#   142390 combined_results_final_extended.csv (142389 + header)
#     2142 combined_results_final_theory_stats.csv (2141 + header)

# Check database tables
sqlite3 combined_results_final.db ".tables"

# Expected:
# combined_answers_extended  combined_answers_short     theory_statistics

# Check theory column exists
head -n 2 combined_results_final_short.csv

# Expected:
# doi,theory,aging_biomarker,molecular_mechanism_of_aging,...
# 10.1001/...,Mitochondrial ROS Theory,Yes,Yes,...
```

### Step 6: Sample Queries

```bash
# Top 10 theories by DOI count
sqlite3 combined_results_final.db \
  "SELECT theory_name, doi_count 
   FROM theory_statistics 
   ORDER BY doi_count DESC 
   LIMIT 10"

# Count by source
sqlite3 combined_results_final.db \
  "SELECT source, COUNT(*) as count 
   FROM combined_answers_extended 
   GROUP BY source"

# Expected:
# fulltext|103905
# rag|38484
# validation_set|0 (or small number)

# Check validation set usage
sqlite3 combined_results_final.db \
  "SELECT COUNT(DISTINCT doi) 
   FROM combined_answers_extended 
   WHERE source='validation_set'"

# Expected: 22 (or similar small number)
```

### Step 7: Export for Analysis

```bash
# Export top theory to separate CSV
sqlite3 -header -csv combined_results_final.db \
  "SELECT * FROM combined_answers_extended 
   WHERE theory = 'Mitochondrial ROS-Induced Free Radical Theory'" \
  > mitochondrial_ros_theory.csv

# Export summary statistics
sqlite3 -header -csv combined_results_final.db \
  "SELECT 
     theory,
     COUNT(DISTINCT doi) as paper_count,
     SUM(CASE WHEN answer LIKE '%Yes%' THEN 1 ELSE 0 END) as yes_count,
     AVG(confidence) as avg_confidence,
     SUM(CASE WHEN source='rag' THEN 1 ELSE 0 END) as rag_count
   FROM combined_answers_extended
   GROUP BY theory
   ORDER BY paper_count DESC
   LIMIT 20" \
  > theory_summary.csv
```

## Common Use Cases

### 1. Find All Papers for a Specific Theory

```bash
sqlite3 -header -csv combined_results_final.db \
  "SELECT doi, aging_biomarker, molecular_mechanism_of_aging 
   FROM combined_answers_short 
   WHERE theory = 'Mitochondrial ROS-Induced Free Radical Theory'" \
  > theory_papers.csv
```

### 2. Find High-Confidence RAG Overrides

```bash
sqlite3 -header -csv combined_results_final.db \
  "SELECT doi, theory, question_key, answer, confidence, reasoning 
   FROM combined_answers_extended 
   WHERE source = 'rag' 
     AND confidence > 0.9 
     AND answer LIKE '%Yes%'
   ORDER BY confidence DESC" \
  > high_confidence_rag.csv
```

### 3. Compare Sources for Same DOI

```bash
# This requires joining with original DBs or checking extended table
sqlite3 combined_results_final.db \
  "SELECT doi, question_key, answer, source, confidence 
   FROM combined_answers_extended 
   WHERE doi = '10.1007/s11357-016-9895-0'
   ORDER BY question_key"
```

### 4. Theory Statistics Analysis

```python
import pandas as pd

# Load theory stats
df = pd.read_csv('combined_results_final_theory_stats.csv')

# Summary statistics
print(f"Total theories: {len(df)}")
print(f"Total DOIs: {df['doi_count'].sum()}")
print(f"Avg DOIs per theory: {df['doi_count'].mean():.1f}")
print(f"Median DOIs per theory: {df['doi_count'].median():.0f}")

# Top 10 theories
print("\nTop 10 Theories:")
print(df.nlargest(10, 'doi_count'))

# Distribution
print("\nDOI Count Distribution:")
print(df['doi_count'].describe())
```

## Troubleshooting

### Issue: "FileNotFoundError: Theory mapping not found"

**Solution:**
```bash
# Check if file exists
ls -l /home/diana.z/hack/theories_extraction_agent/output/final_output/final_theory_to_dois_mapping.json

# Update .env if path is different
nano .env
```

### Issue: "WARNING: DOI X missing from all sources"

**Solution:**
1. Check if DOI should be in theory mapping
2. Add to validation set if needed:
   ```bash
   # Create validation set file
   cp data/validation_set_qa/template.json data/validation_set_qa/10.xxxx_yyyy.json
   # Edit with answers
   ```

### Issue: No RAG overrides

**Solution:**
- Verify RAG DB has data: `sqlite3 rag_results_fast.db "SELECT COUNT(*) FROM paper_answers"`
- Check if RAG answers contain "Yes": `sqlite3 rag_results_fast.db "SELECT answer FROM paper_answers WHERE answer LIKE '%Yes%' LIMIT 5"`

## Next Steps

After successful run:

1. **Analyze results** using provided queries
2. **Share CSVs** with team for review
3. **Validate** sample of RAG overrides
4. **Document** any theory-specific findings
5. **Archive** output files with timestamp

## Automation

For regular runs, create a cron job or script:

```bash
#!/bin/bash
# run_llm_voter_daily.sh

cd /home/diana.z/hack/rag_agent
timestamp=$(date +%Y%m%d_%H%M%S)
output="combined_results_${timestamp}.db"

python scripts/llm_voter.py --output "$output"

# Archive old results
mkdir -p archive
mv combined_results_*.db archive/ 2>/dev/null || true
mv combined_results_*.csv archive/ 2>/dev/null || true

# Keep latest in main directory
cp "archive/$output" combined_results_latest.db
```
