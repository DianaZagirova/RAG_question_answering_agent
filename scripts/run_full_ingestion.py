"""
Production ingestion script for all papers from database.
Ingests papers with full_text_sections (prioritized) or full_text.
"""
import sys
import os
from pathlib import Path
import argparse
import time
from datetime import datetime, timedelta

# CRITICAL: Set GPU device BEFORE any CUDA/PyTorch imports
from dotenv import load_dotenv
load_dotenv()  # Load .env first

# Set CUDA device from environment if not already set
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    cuda_device = os.getenv('CUDA_DEVICE', '3')
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
    print(f"🔧 Setting CUDA_VISIBLE_DEVICES={cuda_device} (from .env)")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# NOW import modules that use CUDA
from src.ingestion.ingest_optimal import PaperIngestionPipeline


def format_time(seconds):
    """Format seconds to human-readable time."""
    return str(timedelta(seconds=int(seconds)))


def estimate_remaining_time(processed, total, elapsed):
    """Estimate remaining time based on current progress."""
    if processed == 0:
        return "calculating..."
    rate = processed / elapsed
    remaining = total - processed
    remaining_seconds = remaining / rate
    return format_time(remaining_seconds)


def run_full_ingestion(
    reset: bool = False,
    batch_size: int = 1000,
    limit: int = None,
    dry_run: bool = False,
    evaluations_db: str = None,
    validated_only: bool = False
):
    """
    Run full ingestion on all papers from database.
    
    Args:
        reset: Whether to reset existing collection
        batch_size: How many papers to process before showing progress
        limit: Optional limit for testing
        dry_run: If True, only show what would be ingested
    """
    
    # Configuration from .env
    db_path = os.getenv('DB_PATH', '/home/diana.z/hack/download_papers_pubmed/paper_collection/data/papers.db')
    collection_name = os.getenv('COLLECTION_NAME', 'scientific_papers_optimal')
    persist_dir = os.getenv('PERSIST_DIR', './chroma_db_optimal')
    chunk_size = int(os.getenv('CHUNK_SIZE', 1500))
    chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 300))
    embedding_model = os.getenv('EMBEDDING_MODEL', 'allenai/specter2')
    backup_model = os.getenv('BACKUP_EMBEDDING_MODEL', 'sentence-transformers/all-mpnet-base-v2')
    
    print("\n" + "="*80)
    print("PRODUCTION INGESTION - Scientific Papers RAG System")
    print("="*80)
    print(f"\n📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if dry_run:
        print("\n⚠️  DRY RUN MODE - No data will be ingested")
    
    print("\n📊 Configuration:")
    print(f"  Database: {db_path}")
    print(f"  Collection: {collection_name}")
    print(f"  Persist Dir: {persist_dir}")
    print(f"  Chunk Size: {chunk_size}")
    print(f"  Chunk Overlap: {chunk_overlap}")
    print(f"  Embedding Model: {embedding_model}")
    print(f"  Backup Model: {backup_model}")
    print(f"  Limit: {limit if limit else 'None (all papers)'}")
    print(f"  Reset Collection: {reset}")
    
    # Estimate papers to process
    import sqlite3
    import chromadb
    from chromadb.config import Settings
    
    validated_dois = set()
    validated_pmids = set()
    
    if validated_only and evaluations_db:
        print("\n🔍 Optimizing paper selection...")
        print("  Step 1: Querying evaluations + papers DB for validated papers with full text...")
        
        # OPTIMIZATION 1: Join evaluations with papers DB to filter for full_text availability
        eval_conn = sqlite3.connect(evaluations_db)
        papers_conn = sqlite3.connect(db_path)
        
        # Attach papers DB to evaluations DB for cross-database query
        eval_cursor = eval_conn.cursor()
        eval_cursor.execute(f"ATTACH DATABASE '{db_path}' AS papers_db")
        
        # Single optimized query: join evaluations with papers, filter for full_text
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
        candidates = eval_cursor.fetchall()
        
        # Track unique papers (not double-counting DOI+PMID)
        unique_papers = set()
        for doi, pmid in candidates:
            # Use DOI as primary identifier, fallback to PMID
            paper_id = doi if doi else str(pmid)
            unique_papers.add(paper_id)
            
            # Also add to separate sets for filtering
            if doi:
                validated_dois.add(doi)
            if pmid:
                validated_pmids.add(str(pmid))
        
        eval_conn.close()
        papers_conn.close()
        
        print(f"    ✓ Found {len(unique_papers):,} validated papers with full text")
        
        if not unique_papers:
            print("\n⚠️  Validated-only mode requested, but no validated papers with full text were found")
            print("   Nothing to process. Exiting.")
            return
        
        # OPTIMIZATION 2: Batch-check Chroma for existing DOIs to exclude already-ingested
        print("  Step 2: Checking Chroma for already-ingested papers...")
        try:
            # FAST METHOD: Query Chroma's SQLite DB directly for all unique DOIs
            chroma_db_path = f"{persist_dir}/chroma.sqlite3"
            chroma_conn = sqlite3.connect(chroma_db_path)
            chroma_cursor = chroma_conn.cursor()
            
            # Get all unique DOIs from metadata (much faster than API calls)
            # Chroma stores metadata in embedding_metadata table with key-value pairs
            query = """
                SELECT DISTINCT string_value
                FROM embedding_metadata
                WHERE key = 'doi'
                  AND string_value IS NOT NULL
                  AND string_value NOT IN ('#N/A', 'Unknown', 'unknown')
            """
            chroma_cursor.execute(query)
            existing_dois_in_chroma = {row[0] for row in chroma_cursor.fetchall() if row[0]}
            chroma_conn.close()
            
            print(f"    ✓ Found {len(existing_dois_in_chroma):,} unique DOIs in Chroma")
            
            # Fast set intersection to find which validated papers are already ingested
            all_candidate_dois = validated_dois | validated_pmids
            existing_dois = all_candidate_dois & existing_dois_in_chroma
            
            # Remove already-ingested DOIs from validated sets
            validated_dois -= existing_dois
            validated_pmids -= existing_dois
            
            # Recalculate remaining unique papers
            remaining_papers = unique_papers - existing_dois
            print(f"    ✓ Found {len(existing_dois):,} already ingested, {len(remaining_papers):,} remaining")
            
            if not remaining_papers:
                print("\n✅ All validated papers are already ingested!")
                print("   Nothing to process. Exiting.")
                return
                
        except Exception as e:
            print(f"    ⚠ Could not check Chroma (collection may not exist yet): {e}")
            print(f"    → Proceeding with all {len(validated_dois) + len(validated_pmids)} validated papers")

    # Count papers to process (already filtered and optimized above for validated_only)
    if validated_only and 'remaining_papers' in locals():
        total_available = len(remaining_papers)
    elif validated_only and unique_papers:
        total_available = len(unique_papers)
    else:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) 
            FROM papers 
            WHERE (full_text_sections IS NOT NULL AND full_text_sections != '' AND full_text_sections != 'null')
            OR (full_text IS NOT NULL AND full_text != '' AND full_text != 'null')
            """
        )
        total_available = cursor.fetchone()[0]
        conn.close()
    
    papers_to_process = min(limit, total_available) if limit else total_available
    
    print(f"\n📚 Papers:")
    if validated_only:
        print(f"  Validated with full text: {len(unique_papers):,}")
        if 'remaining_papers' in locals():
            print(f"  Already ingested: {len(existing_dois):,}")
            print(f"  Remaining to process: {len(remaining_papers):,}")
        else:
            print(f"  To process: {len(unique_papers):,}")
    else:
        print(f"  Available in database: {total_available:,}")
    print(f"  Will process: {papers_to_process:,}")
    
    if dry_run:
        print("\n✅ Dry run complete - configuration verified")
        return
    
    # Confirm if processing large number
    if papers_to_process > 5000 and not limit:
        print(f"\n⚠️  You are about to process {papers_to_process:,} papers")
        print(f"  Estimated time: ~{int(papers_to_process / 40 / 60)} hours")
        print(f"  Storage needed: ~8-10 GB")
        response = input("\n  Continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Aborted.")
            return
    
    print("\n" + "="*80)
    print("Starting Ingestion...")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    # Initialize pipeline
    pipeline = PaperIngestionPipeline(
        db_path=db_path,
        rag_collection_name=collection_name,
        rag_persist_dir=persist_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        backup_model=backup_model,
        allowed_dois=list(validated_dois) if (validated_only and validated_dois) else None,
        allowed_pmids=list(validated_pmids) if (validated_only and validated_pmids) else None
    )
    
    # Reset if requested
    if reset:
        print("🔄 Resetting collection...")
        pipeline.rag.reset_collection()
        print("✓ Collection reset\n")
    
    # Run ingestion with progress tracking
    print("Processing papers...\n")
    
    try:
        pipeline.run(batch_size=batch_size, limit=limit)
        
        elapsed = time.time() - start_time
        
        # Final statistics
        print("\n" + "="*80)
        print("INGESTION COMPLETE!")
        print("="*80)
        
        stats = pipeline.stats
        print(f"\n📊 Final Statistics:")
        print(f"  Papers processed: {stats['processed_papers']:,}")
        print(f"  Papers skipped: {stats['skipped_papers']:,}")
        print(f"  Total chunks created: {stats['total_chunks']:,}")
        print(f"  Average chunks/paper: {stats['total_chunks']/max(stats['processed_papers'],1):.1f}")
        
        print(f"\n⏱️  Time:")
        print(f"  Total time: {format_time(elapsed)}")
        print(f"  Processing rate: {stats['processed_papers']/elapsed*60:.1f} papers/minute")
        
        print(f"\n💾 Database:")
        rag_stats = pipeline.rag.get_statistics()
        print(f"  Collection: {rag_stats['collection_name']}")
        print(f"  Total chunks: {rag_stats['total_chunks']:,}")
        print(f"  Embedding model: {rag_stats['embedding_model']}")
        print(f"  Location: {persist_dir}")
        
        if stats['errors']:
            print(f"\n⚠️  Errors: {len(stats['errors'])}")
            print("  (See ingestion_complete.json for details)")
        
        print("\n" + "="*80)
        print("✅ Ingestion successful!")
        print("="*80)
        
        print(f"\n📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n🎯 Next Steps:")
        print("  1. Test with a question:")
        print("     ./run_rag.sh --question 'Does the paper suggest an aging biomarker?'")
        print("\n  2. Answer all 9 critical questions:")
        print("     ./run_rag.sh --all-questions")
        print("\n" + "="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Ingestion interrupted by user")
        print(f"  Processed so far: {pipeline.stats['processed_papers']:,} papers")
        print(f"  Chunks created: {pipeline.stats['total_chunks']:,}")
        print("\n  You can resume by running again without --reset")
    except Exception as e:
        print(f"\n\n❌ Error during ingestion: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Production ingestion for RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to check configuration
  python run_full_ingestion.py --dry-run
  
  # Test with 100 papers
  python run_full_ingestion.py --limit 100 --reset
  
  # Test with 1000 papers
  python run_full_ingestion.py --limit 1000 --reset
  
  # Full production ingestion (all ~43,637 papers)
  python run_full_ingestion.py --reset
  
  # Continue interrupted ingestion (don't reset)
  python run_full_ingestion.py
        """
    )
    
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset the collection before ingesting (deletes existing data)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of papers to process (for testing)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Progress update frequency (default: 1000)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually ingesting'
    )
    parser.add_argument(
        '--evaluations-db',
        type=str,
        default='/home/diana.z/hack/llm_judge/data/evaluations.db',
        help='Path to evaluations.db with validation results'
    )
    parser.add_argument(
        '--validated-only',
        action='store_true',
        help='Process only papers validated in evaluations.db'
    )
    
    args = parser.parse_args()
    
    run_full_ingestion(
        reset=args.reset,
        batch_size=args.batch_size,
        limit=args.limit,
        dry_run=args.dry_run,
        evaluations_db=args.evaluations_db,
        validated_only=args.validated_only
    )


if __name__ == "__main__":
    main()
