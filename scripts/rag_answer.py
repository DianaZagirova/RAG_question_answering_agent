"""
Complete RAG system for answering aging research questions.
Combines vector retrieval with Azure OpenAI for high-quality answers.
"""
import sys
import os
import argparse
import json
from pathlib import Path

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
from src.core.rag_system import ScientificRAG
from src.core.llm_integration import AzureOpenAIClient, CompleteRAGSystem


def main():
    parser = argparse.ArgumentParser(
        description="Complete RAG system for aging research questions"
    )
    parser.add_argument(
        '--questions-file',
        type=str,
        default=str(Path(__file__).parent.parent / 'data/questions_part2.json'),
        help='Path to JSON file containing question -> options mapping'
    )
    parser.add_argument(
        '--doi',
        type=str,
        help='Restrict retrieval to a specific DOI'
    )
    parser.add_argument(
        '--question',
        type=str,
        help='Single question to answer'
    )
    parser.add_argument(
        '--all-questions',
        action='store_true',
        help='Answer all 9 critical aging questions'
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
        '--embedding-model',
        type=str,
        default=os.getenv('EMBEDDING_MODEL', 'allenai/specter2'),
        help='Embedding model'
    )
    parser.add_argument(
        '--backup-model',
        type=str,
        default=os.getenv('BACKUP_EMBEDDING_MODEL', 'sentence-transformers/all-mpnet-base-v2'),
        help='Backup embedding model'
    )
    parser.add_argument(
        '--n-results',
        type=int,
        default=10,
        help='Number of chunks to retrieve'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.3,
        help='LLM temperature (0-2, lower=more factual)'
    )
    parser.add_argument(
        '--llm-model',
        type=str,
        default=os.getenv('OPENAI_MODEL', 'gpt-4.1-mini'),
        choices=['gpt-4.1-mini', 'gpt-4.1'],
        help='Azure OpenAI chat model to use'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=1000,
        help='Maximum tokens in LLM response'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='rag_answers.json',
        help='Output file for results'
    )
    parser.add_argument(
        '--no-sources',
        action='store_true',
        help='Do not include sources in output (faster)'
    )
    parser.add_argument(
        '--save-enhanced-queries',
        type=str,
        help='Save enhanced queries to a JSON file'
    )
    parser.add_argument(
        '--use-multi-query',
        action='store_true',
        help='Use multi-query retrieval (LLM-enhanced + HyDE + expanded queries) for better coverage'
    )
    parser.add_argument(
        '--predefined-queries',
        type=str,
        default='data/queries_extended.json',
        help='Path to JSON file with predefined queries (default: data/queries_extended.json)'
    )
    
    args = parser.parse_args()
    
    # Initialize RAG system
    print("\n" + "="*70)
    print("Complete RAG System for Aging Research")
    print("="*70 + "\n")
    
    print("Initializing vector database...")
    rag = ScientificRAG(
        collection_name=args.collection_name,
        persist_directory=args.persist_dir,
        embedding_model=args.embedding_model,
        backup_embedding_model=args.backup_model
    )
    
    print("\nInitializing LLM...")
    llm_client = AzureOpenAIClient(model=args.llm_model)
    complete_rag = CompleteRAGSystem(
        rag_system=rag,
        llm_client=llm_client,
        default_n_results=args.n_results,
        use_multi_query=args.use_multi_query
    )
    
    print("\n" + "="*70 + "\n")
    
    # Answer questions
    if args.all_questions:
        questions_path = Path(args.questions_file)
        if not questions_path.exists():
            print(f"❌ Questions file not found: {questions_path}")
            sys.exit(1)

        with open(questions_path, 'r') as f:
            questions_data = json.load(f)

        questions = []
        # Handle new structured format: {"key": {"question": "...", "answers": "..."}}
        for idx, (question_key, question_obj) in enumerate(questions_data.items(), start=1):
            if isinstance(question_obj, dict):
                question_text = question_obj.get('question', '')
                options_str = question_obj.get('answers', '')
            else:
                # Fallback for old format
                question_text = question_key
                options_str = question_obj
            
            options = [opt.strip() for opt in options_str.split('/') if opt.strip()]
            questions.append((idx, question_text, options, question_key))

        results = complete_rag.answer_aging_questions(
            questions=questions,
            output_file=args.output,
            temperature=args.temperature,
            doi=args.doi,
            max_tokens=args.max_tokens
        )
        
        print("="*70)
        print(f"✓ All {len(questions)} questions answered!")
        print(f"✓ Results saved to: {args.output}")
        print("="*70 + "\n")
        
        # Save enhanced queries if requested
        if args.save_enhanced_queries and rag.query_preprocessor:
            enhanced_queries = rag.query_preprocessor.get_enhanced_queries()
            if enhanced_queries:
                with open(args.save_enhanced_queries, 'w') as f:
                    json.dump(enhanced_queries, f, indent=2)
                print(f"✓ Enhanced queries saved to: {args.save_enhanced_queries}\n")
        
    elif args.question:
        # Single question
        result = complete_rag.answer_question(
            question=args.question,
            n_results=args.n_results,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            include_sources=not args.no_sources,
            doi=args.doi
        )
        
        # Display result
        print("\n" + "="*70)
        print("QUESTION")
        print("="*70)
        print(result['question'])
        
        print("\n" + "="*70)
        print("ANSWER")
        print("="*70)
        if result['answer']:
            print(result['answer'])
        else:
            print(f"❌ Error: {result['error']}")
        
        if result['sources'] and not args.no_sources:
            print("\n" + "="*70)
            print(f"SOURCES ({result['n_sources']} papers)")
            print("="*70)
            for i, source in enumerate(result['sources'][:5], 1):
                print(f"\n[{i}] {source['title'][:70]}...")
                print(f"    DOI: {source['doi']}")
                print(f"    Section: {source['section']}")
                print(f"    Relevance: {source['relevance']:.3f}")
        
        if result['llm_metadata'].get('usage'):
            print("\n" + "="*70)
            print("USAGE")
            print("="*70)
            usage = result['llm_metadata']['usage']
            print(f"Model: {result['llm_metadata'].get('model')}")
            print(f"Prompt tokens: {usage['prompt_tokens']}")
            print(f"Completion tokens: {usage['completion_tokens']}")
            print(f"Total tokens: {usage['total_tokens']}")
        
        print("\n" + "="*70 + "\n")
        
        # Save to file
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"✓ Result saved to: {args.output}\n")
    
    else:
        print("Please specify --question or --all-questions")
        print("\nExample usage:")
        print("  # Single question:")
        print("  python rag_answer.py --question 'Does the paper suggest an aging biomarker?'")
        print("\n  # All 9 questions:")
        print("  python rag_answer.py --all-questions")
        print("\n  # With custom parameters:")
        print("  python rag_answer.py --question '...' --n-results 15 --temperature 0.5")


if __name__ == "__main__":
    main()
