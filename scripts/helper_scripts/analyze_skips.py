import sqlite3

db_path = '/home/diana.z/hack/download_papers_pubmed/paper_collection/data/papers.db'
chroma_db_path = './chroma_db_optimal/chroma.sqlite3'

# Get DOIs from papers DB
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("""
    SELECT doi, pmid 
    FROM papers 
    WHERE (full_text_sections IS NOT NULL AND full_text_sections != '') 
       OR (full_text IS NOT NULL AND full_text != '')
    LIMIT 100
""")
sample_papers = cursor.fetchall()
conn.close()

print(f"Sample of {len(sample_papers)} papers from DB:")
for i, (doi, pmid) in enumerate(sample_papers[:5]):
    print(f"  {i+1}. DOI: {doi}, PMID: {pmid}")

# Get DOIs from Chroma
chroma_conn = sqlite3.connect(chroma_db_path)
chroma_cursor = chroma_conn.cursor()

chroma_cursor.execute("""
    SELECT DISTINCT string_value
    FROM embedding_metadata
    WHERE key = 'doi'
      AND string_value IS NOT NULL
    LIMIT 100
""")
chroma_dois = {row[0] for row in chroma_cursor.fetchall()}
print(f"\nSample DOIs in Chroma: {len(chroma_dois)}")
for i, doi in enumerate(list(chroma_dois)[:5]):
    print(f"  {i+1}. {doi}")

# Check overlap
paper_dois = {doi if doi else pmid for doi, pmid in sample_papers}
overlap = paper_dois & chroma_dois
print(f"\nOverlap in sample: {len(overlap)}/{len(sample_papers)}")

# Check if papers are being skipped because they're already in Chroma
papers_already_in_chroma = 0
for doi, pmid in sample_papers:
    identifier = doi if doi else pmid
    if identifier in chroma_dois:
        papers_already_in_chroma += 1

print(f"Papers already in Chroma (from sample): {papers_already_in_chroma}/{len(sample_papers)}")

chroma_conn.close()
