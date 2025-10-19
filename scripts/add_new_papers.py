"""
Incremental ingestion script to add newly available papers.
Adds papers that now have full_text but weren't processed before.
"""
import sys
import os
from pathlib import Path
import argparse
import sqlite3

# CRITICAL: Set GPU device BEFORE any CUDA/PyTorch imports
from dotenv import load_dotenv
load_dotenv()

if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    cuda_device = os.getenv('CUDA_DEVICE', '3')
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
    print(f"🔧 Setting CUDA_VISIBLE_DEVICES={cuda_device} (from .env)")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.ingest_optimal import PaperIngestionPipeline


def get_existing_dois(collection_name, persist_dir):
    """Get list of DOIs already in the collection."""
    from src.core.rag_system import ScientificRAG
    
    try:
        rag = ScientificRAG(
            collection_name=collection_name,
            persist_directory=persist_dir
        )
        
        # Get all unique DOIs from metadata
        collection = rag.collection
        results = collection.get()
        
        existing_dois = set()
        if results and 'metadatas' in results:
            for metadata in results['metadatas']:
                if metadata and 'doi' in metadata:
                    existing_dois.add(metadata['doi'])
        
        return existing_dois
    except Exception as e:
        print(f"⚠️  Could not load existing collection: {e}")
        print("  Assuming no existing papers (fresh collection)")
        return set()


def count_new_papers(db_path, existing_dois):
    """Count how many papers have text now but weren't processed before."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all papers with text
    cursor.execute("""
        SELECT doi
        FROM papers 
        WHERE (full_text_sections IS NOT NULL AND full_text_sections != '' AND full_text_sections != 'null')
        OR (full_text IS NOT NULL AND full_text != '' AND full_text != 'null')
    """)
    
    all_available_dois = set(row[0] for row in cursor.fetchall())
    conn.close()
    
    new_dois = all_available_dois - existing_dois
    
    return len(new_dois), new_dois


def main():
    parser = argparse.ArgumentParser(
        description="Add newly available papers to existing collection"
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default=os.getenv('DB_PATH', '/home/diana.z/hack/download_papers_pubmed/paper_collection/data/papers.db'),
        help='Path to papers.db database'
    )
    parser.add_argument(
        '--collection-name',
        type=str,
        default=os.getenv('COLLECTION_NAME', 'scientific_papers_optimal'),
        help='ChromaDB collection name'
    )
    parser.add_argument(
        '--persist-dir',
        type=str,
        default=os.getenv('PERSIST_DIR', './chroma_db_optimal'),
        help='ChromaDB persist directory'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only show what would be added'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("INCREMENTAL INGESTION - Add New Papers")
    print("="*70)
    
    print(f"\n1️⃣  Checking existing collection...")
    existing_dois = get_existing_dois(args.collection_name, args.persist_dir)
    print(f"   Found {len(existing_dois):,} papers already in collection")
    
    print(f"\n2️⃣  Checking database for new papers...")
    new_count, new_dois = count_new_papers(args.db_path, existing_dois)
    
    print(f"   Papers with text in DB: {len(existing_dois) + new_count:,}")
    print(f"   Already processed: {len(existing_dois):,}")
    print(f"   ✨ New papers to add: {new_count:,}")
    
    if new_count == 0:
        print("\n✅ No new papers to add. Collection is up to date!")
        return
    
    if args.dry_run:
        print("\n⚠️  DRY RUN - showing first 10 new DOIs:")
        for i, doi in enumerate(list(new_dois)[:10], 1):
            print(f"   {i}. {doi}")
        if new_count > 10:
            print(f"   ... and {new_count - 10} more")
        print("\n✅ Dry run complete")
        return
    
    print(f"\n3️⃣  Starting ingestion of {new_count:,} new papers...")
    print("   (This will NOT reset existing data)")
    
    response = input(f"\n   Continue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Aborted.")
        return
    
    # Note: This is a simplified approach
    # A better implementation would filter papers at the SQL level
    print("\n⚠️  Note: This will process all papers and skip duplicates.")
    print("   For better performance, consider manual SQL filtering.")
    
    # Run ingestion WITHOUT reset
    pipeline = PaperIngestionPipeline(
        db_path=args.db_path,
        rag_collection_name=args.collection_name,
        rag_persist_dir=args.persist_dir,
        chunk_size=int(os.getenv('CHUNK_SIZE', 1500)),
        chunk_overlap=int(os.getenv('CHUNK_OVERLAP', 300)),
        embedding_model=os.getenv('EMBEDDING_MODEL', 'allenai/specter2'),
        backup_model=os.getenv('BACKUP_EMBEDDING_MODEL', 'sentence-transformers/all-mpnet-base-v2')
    )
    
    # Do NOT reset - this preserves existing data
    pipeline.run(limit=None)
    
    print("\n" + "="*70)
    print("✅ Incremental ingestion complete!")
    print("="*70)
    print(f"\nNew total chunks in collection:")
    stats = pipeline.rag.get_statistics()
    print(f"  {stats['total_chunks']:,} chunks")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
