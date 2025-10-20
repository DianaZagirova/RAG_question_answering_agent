"""
Move non-validated papers from main ChromaDB to a separate archive database.
This will significantly speed up ingestion and queries.
"""
import sqlite3
import chromadb
from chromadb.config import Settings
from pathlib import Path
import sys
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def get_validated_dois():
    """Get set of validated DOIs from evaluations database."""
    evaluations_db = '/home/diana.z/hack/llm_judge/data/evaluations.db'
    papers_db = '/home/diana.z/hack/download_papers_pubmed/paper_collection/data/papers.db'
    
    eval_conn = sqlite3.connect(evaluations_db)
    eval_cursor = eval_conn.cursor()
    eval_cursor.execute(f"ATTACH DATABASE '{papers_db}' AS papers_db")
    
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
    
    eval_conn.close()
    return validated_dois


def move_non_validated_papers(
    source_persist_dir: str = './chroma_db_optimal',
    source_collection: str = 'scientific_papers_optimal',
    archive_persist_dir: str = './chroma_db_archive',
    archive_collection: str = 'scientific_papers_non_validated',
    batch_size: int = 1000,
    dry_run: bool = False
):
    """
    Move non-validated papers to archive database.
    
    Args:
        source_persist_dir: Main ChromaDB directory
        source_collection: Main collection name
        archive_persist_dir: Archive ChromaDB directory
        archive_collection: Archive collection name
        batch_size: Number of chunks to process at once
        dry_run: If True, only show what would be done
    """
    print("\n" + "="*70)
    print("ChromaDB Cleanup - Move Non-Validated Papers to Archive")
    print("="*70)
    
    if dry_run:
        print("\n⚠️  DRY RUN MODE - No changes will be made\n")
    
    # Get validated DOIs
    print("Loading validated papers list...")
    validated_dois = get_validated_dois()
    print(f"✓ Found {len(validated_dois):,} validated paper identifiers\n")
    
    # Connect to source ChromaDB
    print(f"Connecting to source: {source_persist_dir}/{source_collection}")
    source_client = chromadb.PersistentClient(
        path=source_persist_dir,
        settings=Settings(anonymized_telemetry=False)
    )
    source_coll = source_client.get_collection(source_collection)
    
    # Get all DOIs from source
    print("Analyzing source database...")
    all_data = source_coll.get(include=['metadatas'])
    total_chunks = len(all_data['ids'])
    print(f"✓ Total chunks in source: {total_chunks:,}")
    
    # Identify non-validated chunks
    non_validated_ids = []
    validated_ids = []
    
    for chunk_id, metadata in zip(all_data['ids'], all_data['metadatas']):
        doi = metadata.get('doi', 'Unknown')
        if doi in validated_dois:
            validated_ids.append(chunk_id)
        else:
            non_validated_ids.append(chunk_id)
    
    print(f"\n📊 Analysis:")
    print(f"   Validated chunks: {len(validated_ids):,}")
    print(f"   Non-validated chunks: {len(non_validated_ids):,}")
    
    # Count unique papers
    validated_paper_dois = set()
    non_validated_paper_dois = set()
    for metadata in all_data['metadatas']:
        doi = metadata.get('doi', 'Unknown')
        if doi in validated_dois:
            validated_paper_dois.add(doi)
        else:
            non_validated_paper_dois.add(doi)
    
    print(f"\n   Validated papers: {len(validated_paper_dois):,}")
    print(f"   Non-validated papers: {len(non_validated_paper_dois):,}")
    
    if dry_run:
        print("\n✅ Dry run complete")
        print(f"\nTo proceed with cleanup, run:")
        print(f"  python scripts/cleanup_non_validated.py")
        return
    
    if len(non_validated_ids) == 0:
        print("\n✅ No non-validated papers to move!")
        return
    
    # Confirm action
    print(f"\n⚠️  WARNING: This will:")
    print(f"   1. Move {len(non_validated_ids):,} chunks ({len(non_validated_paper_dois):,} papers) to archive")
    print(f"   2. Delete them from main database")
    print(f"   3. Keep {len(validated_ids):,} chunks ({len(validated_paper_dois):,} papers) in main database")
    
    response = input("\nContinue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Aborted.")
        return
    
    # Create archive database
    print(f"\nCreating archive database: {archive_persist_dir}/{archive_collection}")
    archive_client = chromadb.PersistentClient(
        path=archive_persist_dir,
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Get or create archive collection with same embedding function
    try:
        archive_coll = archive_client.get_collection(archive_collection)
        print("✓ Using existing archive collection")
    except:
        # Get embedding function from source
        archive_coll = archive_client.create_collection(
            name=archive_collection,
            metadata={"description": "Non-validated papers archive"}
        )
        print("✓ Created new archive collection")
    
    # Move chunks in batches
    print(f"\nMoving {len(non_validated_ids):,} chunks to archive...")
    
    for i in tqdm(range(0, len(non_validated_ids), batch_size), desc="Moving chunks"):
        batch_ids = non_validated_ids[i:i+batch_size]
        
        # Get batch data from source
        batch_data = source_coll.get(
            ids=batch_ids,
            include=['documents', 'metadatas', 'embeddings']
        )
        
        # Add to archive
        archive_coll.add(
            ids=batch_data['ids'],
            documents=batch_data['documents'],
            metadatas=batch_data['metadatas'],
            embeddings=batch_data['embeddings']
        )
        
        # Delete from source
        source_coll.delete(ids=batch_ids)
    
    print("✓ Move complete!")
    
    # Verify
    print("\nVerifying...")
    source_count = source_coll.count()
    archive_count = archive_coll.count()
    
    print(f"\n📊 Final Statistics:")
    print(f"   Main database: {source_count:,} chunks ({len(validated_paper_dois):,} papers)")
    print(f"   Archive database: {archive_count:,} chunks ({len(non_validated_paper_dois):,} papers)")
    
    print("\n" + "="*70)
    print("✅ Cleanup Complete!")
    print("="*70)
    print(f"\nMain database now contains only validated papers.")
    print(f"Non-validated papers archived to: {archive_persist_dir}")
    print("\nNext steps:")
    print("  1. Continue ingestion with validated papers:")
    print("     python scripts/run_full_ingestion.py --validated-only")
    print("\n  2. Archive location:")
    print(f"     {archive_persist_dir}/{archive_collection}")
    print("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Move non-validated papers to archive database"
    )
    parser.add_argument(
        '--source-dir',
        default='./chroma_db_optimal',
        help='Source ChromaDB directory'
    )
    parser.add_argument(
        '--source-collection',
        default='scientific_papers_optimal',
        help='Source collection name'
    )
    parser.add_argument(
        '--archive-dir',
        default='./chroma_db_archive',
        help='Archive ChromaDB directory'
    )
    parser.add_argument(
        '--archive-collection',
        default='scientific_papers_non_validated',
        help='Archive collection name'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    
    args = parser.parse_args()
    
    move_non_validated_papers(
        source_persist_dir=args.source_dir,
        source_collection=args.source_collection,
        archive_persist_dir=args.archive_dir,
        archive_collection=args.archive_collection,
        dry_run=args.dry_run
    )
