import sqlite3

evaluations_db = '/home/diana.z/hack/llm_judge/data/evaluations.db'
papers_db = '/home/diana.z/hack/download_papers_pubmed/paper_collection/data/papers.db'
chroma_db = './chroma_db_optimal/chroma.sqlite3'

print("Validated Papers Analysis")
print("="*60)

# Step 1: Count validated papers with full text
eval_conn = sqlite3.connect(evaluations_db)
eval_cursor = eval_conn.cursor()
eval_cursor.execute(f"ATTACH DATABASE '{papers_db}' AS papers_db")

query = """
    SELECT COUNT(DISTINCT COALESCE(e.doi, e.pmid))
    FROM paper_evaluations e
    INNER JOIN papers_db.papers p ON (e.doi = p.doi OR e.pmid = p.pmid)
    WHERE (e.result = 'valid'
       OR e.result = 'doubted'
       OR (e.result = 'not_valid' AND e.confidence_score <= 7))
      AND ((p.full_text_sections IS NOT NULL AND p.full_text_sections != '' AND p.full_text_sections != 'null')
       OR (p.full_text IS NOT NULL AND p.full_text != '' AND p.full_text != 'null'))
"""
eval_cursor.execute(query)
validated_count = eval_cursor.fetchone()[0]
print(f"\n1. Validated papers with full text: {validated_count:,}")

# Step 2: Get validated DOIs
query = """
    SELECT DISTINCT e.doi, e.pmid
    FROM paper_evaluations e
    INNER JOIN papers_db.papers p ON (e.doi = p.doi OR e.pmid = p.pmid)
    WHERE (e.result = 'valid'
       OR e.result = 'doubted'
       OR (e.result = 'not_valid' AND e.confidence_score <= 7))
      AND ((p.full_text_sections IS NOT NULL AND p.full_text_sections != '' AND p.full_text_sections != 'null')
       OR (p.full_text IS NOT NULL AND p.full_text != '' AND p.full_text != 'null'))
"""
eval_cursor.execute(query)
validated_papers = eval_cursor.fetchall()

validated_dois = set()
for doi, pmid in validated_papers:
    if doi:
        validated_dois.add(doi)
    if pmid:
        validated_dois.add(str(pmid))

print(f"   Unique identifiers: {len(validated_dois):,}")

eval_conn.close()

# Step 3: Check what's in Chroma
chroma_conn = sqlite3.connect(chroma_db)
chroma_cursor = chroma_conn.cursor()

chroma_cursor.execute("""
    SELECT COUNT(DISTINCT string_value)
    FROM embedding_metadata
    WHERE key = 'doi'
      AND string_value IS NOT NULL
      AND string_value NOT IN ('#N/A', 'Unknown', 'unknown')
""")
total_in_chroma = chroma_cursor.fetchone()[0]
print(f"\n2. Total papers in ChromaDB: {total_in_chroma:,}")

# Get all DOIs from Chroma
chroma_cursor.execute("""
    SELECT DISTINCT string_value
    FROM embedding_metadata
    WHERE key = 'doi'
      AND string_value IS NOT NULL
      AND string_value NOT IN ('#N/A', 'Unknown', 'unknown')
""")
chroma_dois = {row[0] for row in chroma_cursor.fetchall()}

# Step 4: Calculate overlap
validated_in_chroma = validated_dois & chroma_dois
non_validated_in_chroma = chroma_dois - validated_dois

print(f"\n3. Overlap Analysis:")
print(f"   Validated papers in Chroma: {len(validated_in_chroma):,}")
print(f"   NON-validated papers in Chroma: {len(non_validated_in_chroma):,}")
print(f"   Validated papers NOT yet in Chroma: {len(validated_dois - chroma_dois):,}")

print(f"\n4. Action Required:")
print(f"   ✓ Keep in Chroma: {len(validated_in_chroma):,} papers")
print(f"   ✗ Remove from Chroma: {len(non_validated_in_chroma):,} papers")
print(f"   + Add to Chroma: {len(validated_dois - chroma_dois):,} papers")

chroma_conn.close()

print("\n" + "="*60)
print("RECOMMENDATION:")
print("="*60)
print("1. Run ingestion with --validated-only flag:")
print("   python scripts/run_full_ingestion.py --validated-only")
print()
print("2. Clean up non-validated papers from ChromaDB")
print(f"   (Will remove {len(non_validated_in_chroma):,} papers)")
print("="*60)
