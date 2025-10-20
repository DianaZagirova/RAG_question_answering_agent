import sqlite3

evaluations_db = '/home/diana.z/hack/llm_judge/data/evaluations.db'
papers_db = '/home/diana.z/hack/download_papers_pubmed/paper_collection/data/papers.db'

print("="*70)
print("Debugging Validation Query Discrepancy")
print("="*70)

eval_conn = sqlite3.connect(evaluations_db)
eval_cursor = eval_conn.cursor()
eval_cursor.execute(f"ATTACH DATABASE '{papers_db}' AS papers_db")

# Query 1: Exactly what run_full_ingestion.py uses (lines 111-120)
print("\n1. Query from run_full_ingestion.py (lines 111-120):")
query1 = """
    SELECT DISTINCT e.doi, e.pmid
    FROM paper_evaluations e
    INNER JOIN papers_db.papers p ON (e.doi = p.doi OR e.pmid = p.pmid)
    WHERE (e.result = 'valid'
       OR e.result = 'doubted'
       OR (e.result = 'not_valid' AND e.confidence_score <= 7))
      AND ((p.full_text_sections IS NOT NULL AND p.full_text_sections != '' AND p.full_text_sections != 'null')
       OR (p.full_text IS NOT NULL AND p.full_text != '' AND p.full_text != 'null'))
"""
eval_cursor.execute(query1)
results1 = eval_cursor.fetchall()
print(f"   Results: {len(results1):,} rows")

# Count unique identifiers
dois1 = set()
pmids1 = set()
for doi, pmid in results1:
    if doi:
        dois1.add(doi)
    if pmid:
        pmids1.add(str(pmid))

print(f"   Unique DOIs: {len(dois1):,}")
print(f"   Unique PMIDs: {len(pmids1):,}")
print(f"   Total unique identifiers: {len(dois1) + len(pmids1):,}")

# Query 2: What I used in verify_validated_counts.py
print("\n2. Query from verify_validated_counts.py:")
query2 = """
    SELECT COUNT(DISTINCT COALESCE(e.doi, e.pmid))
    FROM paper_evaluations e
    INNER JOIN papers_db.papers p ON (e.doi = p.doi OR e.pmid = p.pmid)
    WHERE (e.result = 'valid'
       OR e.result = 'doubted'
       OR (e.result = 'not_valid' AND e.confidence_score <= 7))
      AND ((p.full_text_sections IS NOT NULL AND p.full_text_sections != '' AND p.full_text_sections != 'null')
       OR (p.full_text IS NOT NULL AND p.full_text != '' AND p.full_text != 'null'))
"""
eval_cursor.execute(query2)
count2 = eval_cursor.fetchone()[0]
print(f"   Unique papers: {count2:,}")

# Query 3: Check for papers with BOTH doi and pmid
print("\n3. Papers with BOTH DOI and PMID:")
eval_cursor.execute("""
    SELECT COUNT(*)
    FROM paper_evaluations e
    INNER JOIN papers_db.papers p ON (e.doi = p.doi OR e.pmid = p.pmid)
    WHERE (e.result = 'valid'
       OR e.result = 'doubted'
       OR (e.result = 'not_valid' AND e.confidence_score <= 7))
      AND ((p.full_text_sections IS NOT NULL AND p.full_text_sections != '' AND p.full_text_sections != 'null')
       OR (p.full_text IS NOT NULL AND p.full_text != '' AND p.full_text != 'null'))
      AND e.doi IS NOT NULL 
      AND e.pmid IS NOT NULL
""")
both_count = eval_cursor.fetchone()[0]
print(f"   Papers with both DOI and PMID: {both_count:,}")

# Query 4: Check for duplicates in the join
print("\n4. Checking for duplicate matches in JOIN:")
eval_cursor.execute("""
    SELECT e.doi, e.pmid, COUNT(*) as match_count
    FROM paper_evaluations e
    INNER JOIN papers_db.papers p ON (e.doi = p.doi OR e.pmid = p.pmid)
    WHERE (e.result = 'valid'
       OR e.result = 'doubted'
       OR (e.result = 'not_valid' AND e.confidence_score <= 7))
      AND ((p.full_text_sections IS NOT NULL AND p.full_text_sections != '' AND p.full_text_sections != 'null')
       OR (p.full_text IS NOT NULL AND p.full_text != '' AND p.full_text != 'null'))
    GROUP BY e.doi, e.pmid
    HAVING COUNT(*) > 1
    LIMIT 10
""")
duplicates = eval_cursor.fetchall()
if duplicates:
    print(f"   Found {len(duplicates)} papers matching multiple times:")
    for doi, pmid, count in duplicates[:5]:
        print(f"     DOI: {doi}, PMID: {pmid}, matches: {count}")
else:
    print("   No duplicates found in first 10")

# Query 5: The REAL issue - papers can match on BOTH doi AND pmid
print("\n5. THE PROBLEM - Double counting:")
print("   When a paper has both DOI and PMID, the OR join can match twice:")
eval_cursor.execute("""
    SELECT 
        COUNT(*) as total_matches,
        COUNT(DISTINCT e.doi) as unique_dois,
        COUNT(DISTINCT e.pmid) as unique_pmids,
        COUNT(DISTINCT COALESCE(e.doi, e.pmid)) as unique_papers
    FROM paper_evaluations e
    INNER JOIN papers_db.papers p ON (e.doi = p.doi OR e.pmid = p.pmid)
    WHERE (e.result = 'valid'
       OR e.result = 'doubted'
       OR (e.result = 'not_valid' AND e.confidence_score <= 7))
      AND ((p.full_text_sections IS NOT NULL AND p.full_text_sections != '' AND p.full_text_sections != 'null')
       OR (p.full_text IS NOT NULL AND p.full_text != '' AND p.full_text != 'null'))
""")
stats = eval_cursor.fetchone()
print(f"   Total JOIN matches: {stats[0]:,}")
print(f"   Unique DOIs: {stats[1]:,}")
print(f"   Unique PMIDs: {stats[2]:,}")
print(f"   Unique papers (COALESCE): {stats[3]:,}")

print("\n" + "="*70)
print("EXPLANATION")
print("="*70)
print(f"The script counts DOIs and PMIDs separately:")
print(f"  - {len(dois1):,} DOIs")
print(f"  - {len(pmids1):,} PMIDs")
print(f"  - Total: {len(dois1) + len(pmids1):,} identifiers")
print()
print(f"But many papers have BOTH DOI and PMID, so they're counted twice!")
print(f"Actual unique papers: {count2:,}")
print()
print(f"Difference: {len(dois1) + len(pmids1) - count2:,} papers counted twice")

eval_conn.close()

print("\n" + "="*70)
print("SOLUTION")
print("="*70)
print("The script should use DISTINCT COALESCE(doi, pmid) instead of")
print("counting DOIs and PMIDs separately.")
print("="*70)
