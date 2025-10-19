"""
Optimal ingestion script using improved chunking parameters.
Configured for maximum quality on complex scientific questions.
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ingest_papers import PaperIngestionPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Ingest papers with optimal configuration for aging research questions"
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default='/home/diana.z/hack/download_papers_pubmed/paper_collection/data/papers.db',
        help='Path to papers.db database'
    )
    parser.add_argument(
        '--collection-name',
        type=str,
        default='scientific_papers_optimal',
        help='ChromaDB collection name (use different name to compare)'
    )
    parser.add_argument(
        '--persist-dir',
        type=str,
        default='./chroma_db_optimal',
        help='Directory to persist ChromaDB'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of papers (for testing)'
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset existing collection'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("OPTIMAL CONFIGURATION FOR AGING RESEARCH QUESTIONS")
    print("="*70)
    print("\nConfiguration:")
    print("  • Chunk size: 1500 chars (better context)")
    print("  • Chunk overlap: 300 chars (avoid missing key statements)")
    print("  • Sentence tokenizer: NLTK (better accuracy on scientific text)")
    print("  • Embedding model: SPECTER2 → mpnet-base-v2 (better quality)")
    print("="*70 + "\n")
    
    # Initialize pipeline with optimal parameters
    pipeline = PaperIngestionPipeline(
        db_path=args.db_path,
        rag_collection_name=args.collection_name,
        rag_persist_dir=args.persist_dir,
        chunk_size=1500,           # OPTIMAL: Larger chunks for better context
        chunk_overlap=300,         # OPTIMAL: More overlap to avoid missing key info
        embedding_model='allenai/specter2',
        backup_model='sentence-transformers/all-mpnet-base-v2'  # Better backup than MiniLM
    )
    
    # Reset if requested
    if args.reset:
        print("⚠ Resetting existing collection...")
        pipeline.rag.reset_collection()
        print()
    
    # Run ingestion
    pipeline.run(limit=args.limit)
    
    # Print comparison
    print("\n" + "="*70)
    print("CONFIGURATION COMPARISON")
    print("="*70)
    print("\nPrevious (Standard):")
    print("  Chunk size: 1000 chars")
    print("  Overlap: 200 chars")
    print("  Model: all-MiniLM-L6-v2")
    print("  Avg chunks/paper: ~32")
    
    if pipeline.stats['processed_papers'] > 0:
        avg_chunks = pipeline.stats['total_chunks'] / pipeline.stats['processed_papers']
        print("\nNew (Optimal):")
        print("  Chunk size: 1500 chars")
        print("  Overlap: 300 chars")
        print("  Model: SPECTER2 or all-mpnet-base-v2")
        print(f"  Avg chunks/paper: ~{avg_chunks:.1f}")
        
        reduction = ((32 - avg_chunks) / 32) * 100 if avg_chunks < 32 else 0
        print(f"\n  → {reduction:.0f}% fewer chunks (better quality per chunk)")
        print(f"  → {50}% more overlap (better continuity)")
        print(f"  → Better embeddings (higher retrieval quality)")
    
    print("="*70 + "\n")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Test with aging questions:")
    print("   python query_aging_papers.py --all-questions \\")
    print(f"       --collection-name {args.collection_name} \\")
    print(f"       --persist-dir {args.persist_dir}")
    
    print("\n2. Compare with standard configuration:")
    print("   python query_rag.py --question 'Does the paper suggest an aging biomarker?' \\")
    print("       --collection-name scientific_papers --n-results 5")
    print("   vs")
    print("   python query_aging_papers.py --question 'Does the paper suggest an aging biomarker?' \\")
    print(f"       --collection-name {args.collection_name} --question-type biomarker")
    
    print("\n3. Full production ingestion:")
    print("   python ingest_optimal.py --reset")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
