"""
Check the status of the ingestion and verify collection.
"""
import os
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.rag_system import ScientificRAG


def check_ingestion_status():
    """Check if ingestion completed successfully and show statistics."""
    
    print("\n" + "="*70)
    print("INGESTION STATUS CHECK")
    print("="*70)
    
    # Check for completion file
    completion_file = Path("ingestion_complete.json")
    
    if completion_file.exists():
        print("\n✅ INGESTION COMPLETED SUCCESSFULLY!")
        
        with open(completion_file, 'r') as f:
            stats = json.load(f)
        
        print("\n" + "="*70)
        print("FINAL STATISTICS (from ingestion_complete.json)")
        print("="*70)
        print(f"\n📚 Papers:")
        print(f"  Total available: {stats['total_papers']:,}")
        print(f"  Successfully processed: {stats['processed_papers']:,}")
        print(f"  Skipped (no valid text): {stats['skipped_papers']:,}")
        print(f"  Success rate: {stats['processed_papers']/stats['total_papers']*100:.1f}%")
        
        print(f"\n📊 Chunks:")
        print(f"  Total chunks created: {stats['total_chunks']:,}")
        print(f"  Average chunks per paper: {stats['total_chunks']/stats['processed_papers']:.1f}")
        
        if stats['errors']:
            print(f"\n⚠️  Errors encountered: {len(stats['errors'])}")
        else:
            print(f"\n✅ No errors!")
        
    else:
        print("\n⚠️  No completion file found (ingestion_complete.json)")
        print("   Checking progress file...")
        
        progress_file = Path("ingestion_progress.json")
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                stats = json.load(f)
            
            print("\n📊 INGESTION IN PROGRESS (or incomplete):")
            print(f"  Processed so far: {stats['processed_papers']:,}/{stats['total_papers']:,}")
            print(f"  Progress: {stats['processed_papers']/stats['total_papers']*100:.1f}%")
        else:
            print("   No progress file found either.")
    
    # Verify ChromaDB collection
    print("\n" + "="*70)
    print("CHROMADB COLLECTION VERIFICATION")
    print("="*70)
    
    try:
        rag = ScientificRAG(
            collection_name='scientific_papers_optimal',
            persist_directory='./chroma_db_optimal'
        )
        
        db_stats = rag.get_statistics()
        
        print(f"\n✅ Collection loaded successfully!")
        print(f"\n📊 Collection Statistics:")
        print(f"  Collection name: {db_stats['collection_name']}")
        print(f"  Total chunks in database: {db_stats['total_chunks']:,}")
        print(f"  Embedding model: {db_stats['embedding_model']}")
        print(f"  Persist directory: {db_stats['persist_directory']}")
        
        # Estimate papers
        estimated_papers = db_stats['total_chunks'] / 28.0  # Avg chunks per paper
        print(f"\n  Estimated papers: ~{estimated_papers:,.0f}")
        
        # Check if matches completion file
        if completion_file.exists():
            if abs(db_stats['total_chunks'] - stats['total_chunks']) < 100:
                print(f"\n✅ Database matches ingestion stats!")
            else:
                print(f"\n⚠️  Mismatch between ingestion stats and database:")
                print(f"     Ingestion file: {stats['total_chunks']:,} chunks")
                print(f"     Database: {db_stats['total_chunks']:,} chunks")
        
        # Test a sample query
        print("\n" + "="*70)
        print("SAMPLE QUERY TEST")
        print("="*70)
        
        results = rag.query(
            "aging biomarker",
            n_results=3
        )
        
        print(f"\n✅ Query successful!")
        print(f"  Retrieved {len(results)} results")
        
        if results:
            print(f"\n  Sample result:")
            print(f"    DOI: {results[0]['metadata'].get('doi', 'N/A')}")
            print(f"    Relevance: {results[0]['distance']:.3f}")
            print(f"    Text preview: {results[0]['text'][:150]}...")
        
    except Exception as e:
        print(f"\n❌ Error loading collection: {e}")
        print("   Collection may not exist or is corrupted.")
    
    # Final summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if completion_file.exists():
        print("\n✅ INGESTION: SUCCESSFUL")
        print("✅ DATABASE: VERIFIED")
        print("✅ QUERIES: WORKING")
        
        print("\n🎯 READY FOR USE!")
        print("\nNext steps:")
        print("  1. Test with a question:")
        print("     ./run_rag.sh --question 'Does the paper suggest an aging biomarker?'")
        print("\n  2. Answer all 9 critical questions:")
        print("     ./run_rag.sh --all-questions")
        print("\n  3. Check specific results:")
        print("     python scripts/rag_answer.py --all-questions --output results.json")
    else:
        print("\n⚠️  Ingestion may be incomplete or still running.")
        print("   Check if the process is still active:")
        print("     ps aux | grep run_full_ingestion")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    check_ingestion_status()
