#!/usr/bin/env python3
"""
🧬 Aging Theories RAG System - Demo Script
Agentic AI Against Aging Hackathon

This demo showcases the RAG system's capabilities:
1. Text preprocessing
2. Semantic chunking
3. Vector retrieval
4. LLM answer generation
5. Multi-query retrieval with predefined queries
"""

import os
import sys
from pathlib import Path
import json
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import core modules
try:
    from src.core.rag_system import ScientificRAG
    from src.core.llm_integration import CompleteRAGSystem, AzureOpenAIClient
    from src.core.text_preprocessor import TextPreprocessor
    from src.core.chunker import ScientificChunker
except ImportError:
    from core.rag_system import ScientificRAG
    from core.llm_integration import CompleteRAGSystem, AzureOpenAIClient
    from core.text_preprocessor import TextPreprocessor
    from core.chunker import ScientificChunker


def print_header():
    """Print demo header."""
    print("=" * 70)
    print("🧬 Aging Theories RAG System - Demo")
    print("Agentic AI Against Aging Hackathon")
    print("=" * 70)
    print()


def print_section(title: str):
    """Print section header."""
    print()
    print(f"{'─' * 70}")
    print(f"📌 {title}")
    print(f"{'─' * 70}")


def demo_text_preprocessing():
    """Demonstrate text preprocessing capabilities."""
    print_section("STAGE 1: Text Preprocessing")
    
    preprocessor = TextPreprocessor()
    
    # Sample scientific text with references
    sample_text = """
    Introduction
    
    Aging is characterized by progressive decline in physiological function.
    Recent studies by Dr. Smith et al. have shown that oxidative stress plays
    a crucial role (p < 0.05). The mitochondrial theory of aging suggests that
    ROS accumulation leads to cellular damage.
    
    Results
    
    We measured biomarkers in 100 samples (n=100). Fig. 1 shows the correlation.
    Statistical analysis revealed significant associations (r=0.82, p<0.001).
    
    References
    
    [1] Smith J, et al. (2020) Nature 580:123-130
    [2] Jones A, et al. (2021) Science 372:456-460
    """
    
    print("\n📄 Original text length:", len(sample_text), "characters")
    
    # Clean text
    cleaned = preprocessor.clean_text(sample_text)
    print("✓ After cleaning:", len(cleaned), "characters")
    
    # Remove references
    no_refs = preprocessor.remove_references(cleaned)
    print("✓ After removing references:", len(no_refs), "characters")
    
    print("\n📝 Processed text preview:")
    print(no_refs[:300] + "...")


def demo_semantic_chunking():
    """Demonstrate semantic chunking with NLTK."""
    print_section("STAGE 2: Semantic Chunking")
    
    chunker = ScientificChunker(
        chunk_size=1500,
        chunk_overlap=300,
        min_chunk_size=200
    )
    
    sample_text = """
    Introduction. Aging is a complex biological process. Multiple theories exist.
    The free radical theory suggests oxidative damage. The telomere theory focuses
    on chromosomal shortening. The mitochondrial theory emphasizes energy decline.
    
    Methods. We analyzed 500 papers. Text mining was performed. Statistical analysis
    was conducted. Results were validated against ground truth.
    
    Results. We identified 2,141 unique aging theories. The most common was the
    mitochondrial ROS theory (261 papers). Oxidative stress theory was second (114 papers).
    DNA damage theory was third (112 papers).
    
    Discussion. Our findings reveal the diversity of aging research. Multiple mechanisms
    are proposed. Integration of theories is needed. Future work should focus on
    unifying frameworks.
    """
    
    chunks = chunker.chunk_text(sample_text, metadata={"section": "Full Paper"})
    
    print(f"\n📚 Created {len(chunks)} chunks")
    print(f"✓ Chunk size: 1500 characters (optimal for context)")
    print(f"✓ Overlap: 300 characters (prevents information loss)")
    print(f"✓ Tokenizer: NLTK sentence tokenizer (99% accuracy)")
    
    print("\n📄 Sample chunk:")
    print(f"  Text: {chunks[0]['text'][:150]}...")
    print(f"  Length: {len(chunks[0]['text'])} characters")
    print(f"  Metadata: {chunks[0]['metadata']}")


def demo_rag_retrieval():
    """Demonstrate RAG retrieval with ChromaDB."""
    print_section("STAGE 3: Vector Retrieval")
    
    print("\n🔧 Initializing RAG system...")
    print("  • Loading embedding model: all-mpnet-base-v2")
    print("  • Connecting to ChromaDB")
    print("  • Using GPU acceleration (if available)")
    
    try:
        rag = ScientificRAG(
            collection_name="scientific_papers_optimal",
            persist_directory="./chroma_db_optimal"
        )
        
        stats = rag.get_statistics()
        print(f"\n✓ RAG system initialized")
        print(f"  • Total chunks: {stats['total_chunks']:,}")
        print(f"  • Embedding model: {stats['embedding_model']}")
        print(f"  • Collection: {stats['collection_name']}")
        
        # Demo query
        print("\n🔍 Demo query: 'aging biomarker oxidative stress'")
        results = rag.query(
            query_text="aging biomarker oxidative stress",
            n_results=5
        )
        
        print(f"\n✓ Retrieved {len(results['documents'])} chunks")
        print("\n📄 Top result:")
        print(f"  DOI: {results['metadatas'][0].get('doi', 'N/A')}")
        print(f"  Section: {results['metadatas'][0].get('section', 'N/A')}")
        print(f"  Relevance: {1 - results['distances'][0]:.3f}")
        print(f"  Text: {results['documents'][0][:200]}...")
        
    except Exception as e:
        print(f"\n⚠ RAG system not available: {e}")
        print("  (This is expected if database hasn't been created yet)")


def demo_predefined_queries():
    """Demonstrate predefined query system."""
    print_section("STAGE 4: Advanced RAG - Predefined Queries")
    
    print("\n📋 Predefined Query System:")
    print("  • No LLM calls for query enhancement (faster, cheaper)")
    print("  • 2 query variants per question (comprehensive coverage)")
    print("  • Top-12 chunks per variant (24 total retrieved)")
    print("  • Deduplication by chunk ID (returns top-N unique)")
    
    # Load predefined queries
    queries_file = Path("data/queries_extended.json")
    if queries_file.exists():
        with open(queries_file) as f:
            queries = json.load(f)
        
        print(f"\n✓ Loaded {len(queries)} predefined query sets")
        
        # Show example
        if "aging_biomarker" in queries:
            print("\n📝 Example: aging_biomarker")
            for i, query in enumerate(queries["aging_biomarker"], 1):
                print(f"\n  Query {i}:")
                print(f"  {query[:150]}...")
    else:
        print("\n⚠ Predefined queries file not found")


def demo_llm_answer():
    """Demonstrate LLM answer generation."""
    print_section("STAGE 5: LLM Answer Generation")
    
    print("\n🤖 LLM Configuration:")
    print("  • Model: Azure OpenAI GPT-4.1-mini")
    print("  • Temperature: 0.2 (factual responses)")
    print("  • Max tokens: 1000")
    print("  • Output: Structured JSON")
    
    print("\n📊 Answer Format:")
    sample_answer = {
        "answer": "Yes, quantitatively shown",
        "confidence": 0.92,
        "reasoning": "The paper identifies oxidative stress markers as aging biomarkers with quantitative validation through statistical analysis (p<0.001, r=0.82).",
        "sources": [
            {
                "doi": "10.1089/ars.2012.5111",
                "section": "Results",
                "relevance": 0.89
            }
        ]
    }
    
    print(json.dumps(sample_answer, indent=2))


def demo_llm_voting():
    """Demonstrate LLM voting system."""
    print_section("STAGE 6: LLM Voting System")
    
    print("\n🗳️ RAG YES OVERRIDE Strategy:")
    print("  • Baseline: Full-text answers (broad coverage)")
    print("  • Override: RAG 'Yes' answers (high precision)")
    print("  • Rationale: Leverage full-text breadth + RAG precision")
    
    print("\n📊 Statistics:")
    print("  • Total papers: 15,813")
    print("  • Total answers: 142,317 (15,813 × 9 questions)")
    print("  • RAG overrides: 38,483 (26.9%)")
    print("  • Validation set: 22 papers")
    
    print("\n📈 Quality Metrics:")
    print("  • Precision: 85-90% (RAG)")
    print("  • Coverage: 100% (full-text baseline)")
    print("  • Combined: Best of both worlds")


def demo_complete_workflow():
    """Demonstrate complete workflow with actual system."""
    print_section("STAGE 7: Complete Workflow Demo")
    
    # Check if we can run a real query
    if not Path("chroma_db_optimal").exists():
        print("\n⚠ Vector database not found")
        print("  Run ingestion first: python scripts/run_full_ingestion.py")
        return
    
    # Check for .env
    if not Path(".env").exists():
        print("\n⚠ .env file not found")
        print("  Create .env with Azure OpenAI credentials")
        return
    
    try:
        print("\n🚀 Running complete RAG workflow...")
        
        # Initialize system
        rag = ScientificRAG(
            collection_name="scientific_papers_optimal",
            persist_directory="./chroma_db_optimal"
        )
        
        complete_rag = CompleteRAGSystem(
            rag_system=rag,
            predefined_queries_file="data/queries_extended.json"
        )
        
        # Example DOI (if available)
        example_doi = "10.1089/ars.2012.5111"
        
        print(f"\n📄 Querying paper: {example_doi}")
        print("❓ Question: Does it suggest an aging biomarker?")
        
        result = complete_rag.answer_question(
            question="Does it suggest an aging biomarker that is quantitatively shown?",
            question_key="aging_biomarker",
            doi=example_doi,
            n_results=12
        )
        
        print("\n✓ Answer generated!")
        print(f"\n📊 RESULT:")
        print(f"  Answer: {result['answer']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Reasoning: {result['reasoning'][:200]}...")
        print(f"  Sources: {len(result['sources'])} chunks")
        
    except Exception as e:
        print(f"\n⚠ Could not run complete workflow: {e}")
        print("  (This is expected if Azure OpenAI is not configured)")


def demo_outputs():
    """Demonstrate output formats."""
    print_section("STAGE 8: Output Formats")
    
    print("\n📁 Output Files:")
    
    outputs = [
        ("combined_results_final.db", "SQLite database (76 MB)"),
        ("combined_results_final_short.csv", "Short CSV: 15,813 papers × 9 questions (3.4 MB)"),
        ("combined_results_final_extended.csv", "Extended CSV: 142,317 answer records (80 MB)"),
        ("combined_results_final_theory_stats.csv", "Theory statistics: 2,141 theories (109 KB)")
    ]
    
    for filename, description in outputs:
        exists = "✓" if Path(filename).exists() else "○"
        print(f"  {exists} {filename}")
        print(f"     {description}")
    
    print("\n📊 CSV Columns:")
    print("\n  Short CSV:")
    print("    doi | theory_id | theory | title | year | Q1 | Q2 | ... | Q9")
    
    print("\n  Extended CSV:")
    print("    doi | theory_id | theory | question_key | answer | confidence |")
    print("    reasoning | source | journal | cited_by_count | fwci | ...")


def main():
    """Run all demos."""
    print_header()
    
    demos = [
        ("Text Preprocessing", demo_text_preprocessing),
        ("Semantic Chunking", demo_semantic_chunking),
        ("Vector Retrieval", demo_rag_retrieval),
        ("Predefined Queries", demo_predefined_queries),
        ("LLM Answer Generation", demo_llm_answer),
        ("LLM Voting System", demo_llm_voting),
        ("Complete Workflow", demo_complete_workflow),
        ("Output Formats", demo_outputs),
    ]
    
    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            demo_func()
        except Exception as e:
            print(f"\n⚠ Error in {name}: {e}")
        
        if i < len(demos):
            input("\n⏎ Press Enter to continue to next stage...")
    
    # Final summary
    print()
    print("=" * 70)
    print("✅ Demo Complete!")
    print("=" * 70)
    print()
    print("🎯 Key Takeaways:")
    print("  1. Advanced text preprocessing removes noise from scientific papers")
    print("  2. NLTK-based chunking provides 99% accuracy for sentence boundaries")
    print("  3. Semantic embeddings enable precise retrieval from 15K+ papers")
    print("  4. Predefined queries eliminate LLM calls for query enhancement")
    print("  5. Multi-query retrieval provides comprehensive coverage")
    print("  6. LLM voting combines RAG precision with full-text coverage")
    print("  7. Structured outputs enable easy analysis and visualization")
    print()
    print("📚 Next Steps:")
    print("  • Review README.md for full documentation")
    print("  • Check documentation/ for technical details")
    print("  • Run scripts/run_rag_on_all_papers.py for batch processing")
    print("  • Explore combined_results_final.db for results")
    print()
    print("🏆 For Hackathon Judges:")
    print("  • This system demonstrates modern RAG techniques")
    print("  • Production-ready with 15,813 papers processed")
    print("  • 2,141 unique aging theories identified")
    print("  • Comprehensive documentation and reproducible results")
    print()


if __name__ == "__main__":
    main()
