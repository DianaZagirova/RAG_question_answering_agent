"""
Custom ingestion script for validated papers from evaluations.db.
Processes papers based on validation criteria:
- "valid" papers
- "doubted" papers  
- "not_valid" papers with confidence_score <= 7
"""
import sys
import os
from pathlib import Path
import sqlite3
import argparse
import time
from datetime import datetime, timedelta

# CRITICAL: Set GPU device BEFORE any CUDA/PyTorch imports
from dotenv import load_dotenv
load_dotenv()

if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    cuda_device = os.getenv('CUDA_DEVICE', '3')
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
    print(f"🔧 Setting CUDA_VISIBLE_DEVICES={cuda_device} (from .env)")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.ingest_papers import PaperIngestionPipeline
from tqdm import tqdm


def get_validated_dois(evaluations_db_path):
    """
    Get DOIs of papers that meet validation criteria.
    
    Criteria:
    - result = 'valid' OR
    - result = 'doubted' OR
    - (result = 'not_valid' AND confidence_score <= 7)
    
    Returns:
        dict: Statistics and list of DOIs
    """
    conn = sqlite3.connect(evaluations_db_path)
    cursor = conn.cursor()
    
    # Get counts for each category
    print("\n" + "="*70)
    print("VALIDATION DATABASE ANALYSIS")
    print("="*70)
    
    cursor.execute("SELECT COUNT(*) FROM paper_evaluations WHERE result = 'valid'")
    valid_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM paper_evaluations WHERE result = 'doubted'")
    doubted_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM paper_evaluations WHERE result = 'not_valid' AND confidence_score <= 7")
    low_confidence_invalid_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM paper_evaluations WHERE result = 'not_valid' AND confidence_score > 7")
    high_confidence_invalid_count = cursor.fetchone()[0]
    
    total_evaluated = valid_count + doubted_count + low_confidence_invalid_count + high_confidence_invalid_count
    
    print(f"\n📊 Evaluation Results:")
    print(f"  Total evaluated papers: {total_evaluated:,}")
    print(f"  ✅ Valid: {valid_count:,}")
    print(f"  ⚠️  Doubted: {doubted_count:,}")
    print(f"  ❓ Not valid (low confidence ≤7): {low_confidence_invalid_count:,}")
    print(f"  ❌ Not valid (high confidence >7): {high_confidence_invalid_count:,}")
    
    # Get DOIs matching criteria
    query = """
        SELECT doi 
        FROM paper_evaluations 
        WHERE result = 'valid' 
           OR result = 'doubted'
           OR (result = 'not_valid' AND confidence_score <= 7)
    """
    
    cursor.execute(query)
    dois = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    
    papers_to_ingest = len(dois)
    papers_to_skip = high_confidence_invalid_count
    
    print(f"\n🎯 Ingestion Plan:")
    print(f"  Papers to ingest: {papers_to_ingest:,}")
    print(f"  Papers to skip: {papers_to_skip:,}")
    print(f"  Percentage to ingest: {papers_to_ingest/total_evaluated*100:.1f}%")
    
    return {
        'dois': set(dois),
        'valid_count': valid_count,
        'doubted_count': doubted_count,
        'low_confidence_invalid_count': low_confidence_invalid_count,
        'high_confidence_invalid_count': high_confidence_invalid_count,
        'total_to_ingest': papers_to_ingest,
        'total_to_skip': papers_to_skip
    }


def check_papers_in_main_db(papers_db_path, dois):
    """
    Check which DOIs from evaluations exist in papers.db with full text.
    
    Returns:
        dict: DOIs with text available, missing DOIs
    """
    conn = sqlite3.connect(papers_db_path)
    cursor = conn.cursor()
    
    print("\n" + "="*70)
    print("CHECKING PAPERS DATABASE")
    print("="*70)
    
    # Check which DOIs exist with full text
    dois_with_text = set()
    dois_without_text = set()
    dois_not_found = set()
    
    print(f"\nChecking {len(dois):,} DOIs in papers.db...")
    
    for doi in tqdm(dois, desc="Checking DOIs"):
        cursor.execute("""
            SELECT doi, full_text, full_text_sections
            FROM papers
            WHERE doi = ?
        """, (doi,))
        
        result = cursor.fetchone()
        
        if result:
            _, full_text, full_text_sections = result
            has_text = (
                (full_text_sections and full_text_sections.strip() and full_text_sections != 'null') or
                (full_text and full_text.strip() and full_text != 'null')
            )
            
            if has_text:
                dois_with_text.add(doi)
            else:
                dois_without_text.add(doi)
        else:
            dois_not_found.add(doi)
    
    conn.close()
    
    print(f"\n📊 Results:")
    print(f"  ✅ DOIs with full text: {len(dois_with_text):,}")
    print(f"  ⚠️  DOIs without full text: {len(dois_without_text):,}")
    print(f"  ❌ DOIs not found in papers.db: {len(dois_not_found):,}")
    
    return {
        'dois_with_text': dois_with_text,
        'dois_without_text': dois_without_text,
        'dois_not_found': dois_not_found
    }


def ingest_validated_papers(
    papers_db_path,
    evaluations_db_path,
    dois_to_ingest,
    collection_name='scientific_papers_optimal',
    persist_dir='./chroma_db_optimal',
    reset=False,
    dry_run=False
):
    """
    Ingest only papers that match the DOI list.
    """
    
    if dry_run:
        print("\n⚠️  DRY RUN - No data will be ingested")
        return
    
    print("\n" + "="*70)
    print("STARTING SELECTIVE INGESTION")
    print("="*70)
    
    if not reset:
        print("\n⚠️  Running in INCREMENTAL mode (keeping existing papers)")
    else:
        print("\n⚠️  Running in RESET mode (will delete existing collection)")
        response = input("Are you sure? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Aborted.")
            return
    
    # Create a temporary filtered database or use SQL filtering
    # For simplicity, we'll modify the ingestion to filter by DOI
    
    print(f"\n📚 Will ingest {len(dois_to_ingest):,} papers")
    print(f"   Collection: {collection_name}")
    print(f"   Persist dir: {persist_dir}")
    
    start_time = time.time()
    
    # Initialize pipeline
    pipeline = PaperIngestionPipeline(
        db_path=papers_db_path,
        rag_collection_name=collection_name,
        rag_persist_dir=persist_dir,
        chunk_size=int(os.getenv('CHUNK_SIZE', 1500)),
        chunk_overlap=int(os.getenv('CHUNK_OVERLAP', 300)),
        embedding_model=os.getenv('EMBEDDING_MODEL', 'allenai/specter2'),
        backup_model=os.getenv('BACKUP_EMBEDDING_MODEL', 'sentence-transformers/all-mpnet-base-v2')
    )
    
    if reset:
        print("\n🔄 Resetting collection...")
        pipeline.rag.reset_collection()
        print("✓ Collection reset\n")
    
    # Custom ingestion with DOI filtering
    print("Processing papers...\n")
    
    processed = 0
    skipped = 0
    total_chunks = 0
    
    for paper in tqdm(pipeline.fetch_papers(), desc="Processing papers"):
        doi = paper.get('doi')
        
        # Skip if DOI not in our validated list
        if doi not in dois_to_ingest:
            skipped += 1
            continue
        
        # Process paper
        chunks = pipeline.process_paper(paper)
        
        if chunks:
            pipeline.rag.add_chunks(chunks)
            processed += 1
            total_chunks += len(chunks)
        else:
            skipped += 1
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("INGESTION COMPLETE!")
    print("="*70)
    
    print(f"\n📊 Statistics:")
    print(f"  Papers processed: {processed:,}")
    print(f"  Papers skipped: {skipped:,}")
    print(f"  Total chunks: {total_chunks:,}")
    print(f"  Average chunks/paper: {total_chunks/max(processed,1):.1f}")
    print(f"  Time: {timedelta(seconds=int(elapsed))}")
    print(f"  Rate: {processed/elapsed*60:.1f} papers/minute")
    
    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest validated papers from evaluations.db",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Validation Criteria:
  - result = 'valid'
  - result = 'doubted'
  - result = 'not_valid' AND confidence_score <= 7

Examples:
  # Check what would be ingested
  python ingest_validated_papers.py --dry-run
  
  # Ingest validated papers (incremental)
  python ingest_validated_papers.py
  
  # Reset and ingest only validated papers
  python ingest_validated_papers.py --reset
        """
    )
    
    parser.add_argument(
        '--papers-db',
        type=str,
        default='/home/diana.z/hack/download_papers_pubmed/paper_collection/data/papers.db',
        help='Path to papers.db'
    )
    parser.add_argument(
        '--evaluations-db',
        type=str,
        default='/home/diana.z/hack/llm_judge/data/evaluations.db',
        help='Path to evaluations.db'
    )
    parser.add_argument(
        '--collection-name',
        type=str,
        default='scientific_papers_optimal',
        help='ChromaDB collection name'
    )
    parser.add_argument(
        '--persist-dir',
        type=str,
        default='./chroma_db_optimal',
        help='ChromaDB persist directory'
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset collection before ingesting'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually ingesting'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("VALIDATED PAPERS INGESTION")
    print("="*70)
    print(f"\n📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Get validated DOIs from evaluations.db
    validation_stats = get_validated_dois(args.evaluations_db)
    validated_dois = validation_stats['dois']
    
    # Step 2: Check which DOIs have full text in papers.db
    availability_stats = check_papers_in_main_db(args.papers_db, validated_dois)
    dois_to_ingest = availability_stats['dois_with_text']
    
    print("\n" + "="*70)
    print("FINAL INGESTION PLAN")
    print("="*70)
    
    print(f"\n📊 Summary:")
    print(f"  Validated papers (from evaluations.db): {len(validated_dois):,}")
    print(f"    ├─ Valid: {validation_stats['valid_count']:,}")
    print(f"    ├─ Doubted: {validation_stats['doubted_count']:,}")
    print(f"    └─ Not valid (low confidence): {validation_stats['low_confidence_invalid_count']:,}")
    print(f"\n  Papers with full text (in papers.db): {len(dois_to_ingest):,}")
    print(f"  Papers without full text: {len(availability_stats['dois_without_text']):,}")
    print(f"  Papers not found in papers.db: {len(availability_stats['dois_not_found']):,}")
    
    print(f"\n🎯 Will ingest: {len(dois_to_ingest):,} papers")
    
    if args.dry_run:
        print("\n✅ Dry run complete - no data ingested")
        return
    
    # Step 3: Ingest papers
    ingest_validated_papers(
        papers_db_path=args.papers_db,
        evaluations_db_path=args.evaluations_db,
        dois_to_ingest=dois_to_ingest,
        collection_name=args.collection_name,
        persist_dir=args.persist_dir,
        reset=args.reset,
        dry_run=args.dry_run
    )
    
    print(f"\n📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
