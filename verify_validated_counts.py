import sqlite3

evaluations_db = '/home/diana.z/hack/llm_judge/data/evaluations.db'
papers_db = '/home/diana.z/hack/download_papers_pubmed/paper_collection/data/papers.db'

print("="*70)
print("Detailed Validation Analysis")
print("="*70)

# Step 1: Check evaluations DB
eval_conn = sqlite3.connect(evaluations_db)
eval_cursor = eval_conn.cursor()

print("\n1. Evaluations Database Breakdown:")
eval_cursor.execute("""
    SELECT result, COUNT(*) as count 
    FROM paper_evaluations 
    GROUP BY result 
    ORDER BY count DESC
""")
for result, count in eval_cursor.fetchall():
    result_label = result if result else '(null)'
    print(f"   {result_label:15} {count:,}")

# Step 2: Count by confidence for not_valid
print("\n2. Not Valid Papers by Confidence Score:")
eval_cursor.execute("""
    SELECT 
        CASE 
            WHEN confidence_score <= 7 THEN 'Low confidence (<=7)'
            ELSE 'High confidence (>7)'
        END as confidence_group,
        COUNT(*) as count
    FROM paper_evaluations 
    WHERE result = 'not_valid'
    GROUP BY confidence_group
""")
for group, count in eval_cursor.fetchall():
    print(f"   {group:25} {count:,}")

# Step 3: Count validated papers (valid + doubted + low confidence not_valid)
print("\n3. Validated Papers (by criteria):")
eval_cursor.execute("""
    SELECT COUNT(*) 
    FROM paper_evaluations 
    WHERE result = 'valid'
       OR result = 'doubted'
       OR (result = 'not_valid' AND confidence_score <= 7)
""")
validated_total = eval_cursor.fetchone()[0]
print(f"   Total validated: {validated_total:,}")

# Breakdown
eval_cursor.execute("SELECT COUNT(*) FROM paper_evaluations WHERE result = 'valid'")
valid_count = eval_cursor.fetchone()[0]
eval_cursor.execute("SELECT COUNT(*) FROM paper_evaluations WHERE result = 'doubted'")
doubted_count = eval_cursor.fetchone()[0]
eval_cursor.execute("SELECT COUNT(*) FROM paper_evaluations WHERE result = 'not_valid' AND confidence_score <= 7")
low_conf_count = eval_cursor.fetchone()[0]

print(f"     - valid: {valid_count:,}")
print(f"     - doubted: {doubted_count:,}")
print(f"     - not_valid (confidence <= 7): {low_conf_count:,}")

# Step 4: Join with papers DB to check full_text availability
print("\n4. Validated Papers WITH Full Text:")
eval_cursor.execute(f"ATTACH DATABASE '{papers_db}' AS papers_db")

query = """
    SELECT 
        e.result,
        COUNT(*) as total,
        SUM(CASE WHEN p.full_text_sections IS NOT NULL AND p.full_text_sections != '' AND p.full_text_sections != 'null' THEN 1 ELSE 0 END) as with_sections,
        SUM(CASE WHEN p.full_text IS NOT NULL AND p.full_text != '' AND p.full_text != 'null' THEN 1 ELSE 0 END) as with_full_text,
        SUM(CASE WHEN (p.full_text_sections IS NOT NULL AND p.full_text_sections != '' AND p.full_text_sections != 'null')
                   OR (p.full_text IS NOT NULL AND p.full_text != '' AND p.full_text != 'null') THEN 1 ELSE 0 END) as with_any_text
    FROM paper_evaluations e
    LEFT JOIN papers_db.papers p ON (e.doi = p.doi OR e.pmid = p.pmid)
    WHERE e.result = 'valid'
       OR e.result = 'doubted'
       OR (e.result = 'not_valid' AND e.confidence_score <= 7)
    GROUP BY e.result
    ORDER BY total DESC
"""
eval_cursor.execute(query)

print(f"\n   {'Result':<15} {'Total':<10} {'With Sections':<15} {'With Full Text':<15} {'With Any Text':<15}")
print(f"   {'-'*15} {'-'*10} {'-'*15} {'-'*15} {'-'*15}")

total_with_text = 0
for result, total, with_sections, with_full_text, with_any_text in eval_cursor.fetchall():
    result_label = result if result else '(null)'
    print(f"   {result_label:<15} {total:<10,} {with_sections:<15,} {with_full_text:<15,} {with_any_text:<15,}")
    total_with_text += with_any_text

print(f"   {'-'*15} {'-'*10} {'-'*15} {'-'*15} {'-'*15}")
print(f"   {'TOTAL':<15} {validated_total:<10,} {'':<15} {'':<15} {total_with_text:<15,}")

# Step 5: Check unique DOIs/PMIDs
print("\n5. Unique Identifiers:")
eval_cursor.execute("""
    SELECT COUNT(DISTINCT COALESCE(e.doi, e.pmid))
    FROM paper_evaluations e
    INNER JOIN papers_db.papers p ON (e.doi = p.doi OR e.pmid = p.pmid)
    WHERE (e.result = 'valid'
       OR e.result = 'doubted'
       OR (e.result = 'not_valid' AND e.confidence_score <= 7))
      AND ((p.full_text_sections IS NOT NULL AND p.full_text_sections != '' AND p.full_text_sections != 'null')
       OR (p.full_text IS NOT NULL AND p.full_text != '' AND p.full_text != 'null'))
""")
unique_papers = eval_cursor.fetchone()[0]
print(f"   Unique papers with full text: {unique_papers:,}")

# Step 6: Check what's missing
print("\n6. Papers WITHOUT Full Text:")
eval_cursor.execute("""
    SELECT COUNT(*)
    FROM paper_evaluations e
    LEFT JOIN papers_db.papers p ON (e.doi = p.doi OR e.pmid = p.pmid)
    WHERE (e.result = 'valid'
       OR e.result = 'doubted'
       OR (e.result = 'not_valid' AND e.confidence_score <= 7))
      AND (p.full_text_sections IS NULL OR p.full_text_sections = '' OR p.full_text_sections = 'null')
      AND (p.full_text IS NULL OR p.full_text = '' OR p.full_text = 'null')
""")
without_text = eval_cursor.fetchone()[0]
print(f"   Validated papers without full text: {without_text:,}")
print(f"   Percentage with full text: {total_with_text/validated_total*100:.1f}%")

eval_conn.close()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Total validated papers (by criteria): {validated_total:,}")
print(f"  - With full text: {total_with_text:,}")
print(f"  - Without full text: {without_text:,}")
print(f"  - Unique papers to ingest: {unique_papers:,}")
print("="*70)
