import sqlite3

db_path = '/home/diana.z/hack/download_papers_pubmed/paper_collection/data/papers.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Total papers
cursor.execute("SELECT COUNT(*) FROM papers")
total = cursor.fetchone()[0]
print(f"Total papers in DB: {total:,}")

# Papers with full_text_sections
cursor.execute("SELECT COUNT(*) FROM papers WHERE full_text_sections IS NOT NULL AND full_text_sections != ''")
with_sections = cursor.fetchone()[0]
print(f"Papers with full_text_sections: {with_sections:,}")

# Papers with full_text
cursor.execute("SELECT COUNT(*) FROM papers WHERE full_text IS NOT NULL AND full_text != ''")
with_full_text = cursor.fetchone()[0]
print(f"Papers with full_text: {with_full_text:,}")

# Papers with either
cursor.execute("""
    SELECT COUNT(*) FROM papers 
    WHERE (full_text_sections IS NOT NULL AND full_text_sections != '') 
       OR (full_text IS NOT NULL AND full_text != '')
""")
with_any = cursor.fetchone()[0]
print(f"Papers with any full text: {with_any:,}")

# Check Chroma DB
chroma_db_path = './chroma_db_optimal/chroma.sqlite3'
try:
    chroma_conn = sqlite3.connect(chroma_db_path)
    chroma_cursor = chroma_conn.cursor()
    
    # Get unique DOIs in Chroma
    chroma_cursor.execute("""
        SELECT COUNT(DISTINCT string_value)
        FROM embedding_metadata
        WHERE key = 'doi'
          AND string_value IS NOT NULL
          AND string_value NOT IN ('#N/A', 'Unknown', 'unknown')
    """)
    unique_dois = chroma_cursor.fetchone()[0]
    print(f"\nUnique DOIs in Chroma: {unique_dois:,}")
    
    chroma_conn.close()
except Exception as e:
    print(f"\nCould not check Chroma: {e}")

conn.close()
