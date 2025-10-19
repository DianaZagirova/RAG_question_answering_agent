import sqlite3

conn = sqlite3.connect('/home/diana.z/hack/download_papers_pubmed/paper_collection/data/papers.db')
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM papers WHERE full_text_sections IS NOT NULL AND full_text_sections != ''")
print('Papers with full_text_sections:', cursor.fetchone()[0])

cursor.execute("SELECT COUNT(*) FROM papers WHERE full_text IS NOT NULL AND full_text != ''")
print('Papers with full_text:', cursor.fetchone()[0])

cursor.execute("SELECT COUNT(*) FROM papers WHERE (full_text_sections IS NOT NULL AND full_text_sections != '') OR (full_text IS NOT NULL AND full_text != '')")
print('Papers with either:', cursor.fetchone()[0])

conn.close()
