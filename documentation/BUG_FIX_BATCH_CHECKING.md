# Bug Fix: Batch Checking Not Finding Existing Papers

## Problem
When running `python scripts/run_full_ingestion.py --validated-only`, the optimization step reported:
```
Step 2: Checking Chroma for already-ingested papers...
  ✓ Found 0 already ingested, 26576 remaining
```

Despite having **1,349,045 chunks** already in the database.

## Root Cause
**Invalid Chroma API parameter**: `include=['ids']`

### Location
Two files had the bug:
1. `scripts/run_full_ingestion.py` line 159
2. `src/ingestion/ingest_papers.py` line 178

### The Bug
```python
result = collection.get(where={'doi': doi}, limit=1, include=['ids'])
```

### Why It Failed
- Chroma's `get()` method doesn't support `'ids'` in the `include` parameter
- Valid include values: `['embeddings', 'documents', 'metadatas', 'uris', 'data']`
- IDs are **always returned by default** (no need to include them)
- The invalid parameter caused a `ValueError` exception
- Exception was silently caught by `except Exception: pass`
- Result: Every DOI check failed → reported 0 already ingested

## The Fix (v2 - Final Optimized Version)

### Initial Fix
Remove the invalid `include=['ids']` parameter:
```python
# Before (broken)
result = collection.get(where={'doi': doi}, limit=1, include=['ids'])

# After (fixed)
result = collection.get(where={'doi': doi}, limit=1)
```

### Performance Optimization
Even with the fix, checking 26,000+ DOIs individually was too slow (~hours).

**Final solution**: Query Chroma's SQLite database directly for all unique DOIs, then use fast set operations:

```python
# Query Chroma's SQLite DB directly (FAST!)
chroma_db_path = f"{persist_dir}/chroma.sqlite3"
conn = sqlite3.connect(chroma_db_path)
cursor = conn.cursor()

query = """
    SELECT DISTINCT string_value
    FROM embedding_metadata
    WHERE key = 'doi'
      AND string_value IS NOT NULL
      AND string_value NOT IN ('#N/A', 'Unknown', 'unknown')
"""

cursor.execute(query)
existing_dois_in_chroma = {row[0] for row in cursor.fetchall() if row[0]}
conn.close()

# Fast set intersection (milliseconds)
all_candidate_dois = validated_dois | validated_pmids
existing_dois = all_candidate_dois & existing_dois_in_chroma
```

**Performance**: Extracts 47,075 unique DOIs in **~3 seconds** vs hours with individual API calls.

## Files Changed
1. **`scripts/run_full_ingestion.py`** - Line 159
2. **`src/ingestion/ingest_papers.py`** - Line 178

## Verification
Test with known DOIs:
```python
import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(path='chroma_db_optimal', settings=Settings(anonymized_telemetry=False))
collection = client.get_collection('scientific_papers_optimal')

# This now works correctly
result = collection.get(where={'doi': '10.1002/adma.202413096'}, limit=1)
print(f"Found: {len(result.get('ids', [])) > 0}")  # Should print: Found: True
```

## Expected Behavior After Fix
When you rerun:
```bash
python scripts/run_full_ingestion.py --validated-only
```

You should now see:
```
🔍 Optimizing paper selection...
  Step 1: Querying evaluations + papers DB for validated papers with full text...
    ✓ Found 26,576 validated papers with full text
  Step 2: Checking Chroma for already-ingested papers...
    ✓ Found ~23,000+ already ingested, ~3,000 remaining

📚 Papers:
  Validated with full text (after Chroma check): ~3,000
  Will process: ~3,000
```

(Exact numbers depend on overlap between validated papers and already-ingested papers)

## Impact
- **Before fix**: Would attempt to reprocess all 26,576 papers (wasting ~11 hours)
- **After fix**: Only processes papers not yet in Chroma (likely ~3,000 or fewer)
- **Time saved**: ~8-10 hours on reruns

## Lesson Learned
- Always validate API parameters against documentation
- Don't use bare `except Exception: pass` - it hides bugs
- Test optimization logic with known data before production runs
