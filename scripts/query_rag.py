"""
Query interface for the RAG system.
Allows users to ask questions and get answers from the scientific papers.
"""
import argparse
import json
from typing import Optional
from rag_system import ScientificRAG, create_context_for_llm


class RAGQueryInterface:
    """Interactive interface for querying the RAG system."""
    
    def __init__(
        self,
        collection_name: str = "scientific_papers",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "allenai/specter2",
        backup_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """Initialize the query interface."""
        print("Loading RAG system...")
        self.rag = ScientificRAG(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_model=embedding_model,
            backup_embedding_model=backup_model
        )
        
        # Get and display statistics
        stats = self.rag.get_statistics()
        print("\n" + "="*60)
        print("RAG System Ready")
        print("="*60)
        print(f"Collection: {stats['collection_name']}")
        print(f"Total chunks: {stats['total_chunks']:,}")
        print(f"Embedding model: {stats['embedding_model']}")
        print("="*60 + "\n")
    
    def query(
        self,
        question: str,
        n_results: int = 5,
        output_format: str = "detailed"
    ) -> dict:
        """
        Query the RAG system and return results.
        
        Args:
            question: The question to ask
            n_results: Number of context chunks to retrieve
            output_format: 'detailed', 'simple', or 'llm'
            
        Returns:
            Dictionary with query results
        """
        print(f"\n🔍 Question: {question}")
        print(f"   Retrieving top {n_results} relevant chunks...\n")
        
        # Get answer context
        response = self.rag.answer_question(
            question=question,
            n_context_chunks=n_results,
            include_metadata=True
        )
        
        if output_format == "detailed":
            return self._format_detailed(response)
        elif output_format == "simple":
            return self._format_simple(response)
        elif output_format == "llm":
            return self._format_for_llm(response)
        else:
            return response
    
    def _format_detailed(self, response: dict) -> dict:
        """Format response with full details."""
        print("📚 Retrieved Sources:")
        print("="*60)
        
        for i, source in enumerate(response['sources'], 1):
            print(f"\n[Source {i}]")
            print(f"  Title: {source['title']}")
            print(f"  DOI: {source['doi']}")
            print(f"  Section: {source['section']}")
            print(f"  Relevance: {source['relevance']:.3f}")
        
        print("\n" + "="*60)
        print("📝 Context:")
        print("="*60)
        print(response['context'])
        print("="*60 + "\n")
        
        return response
    
    def _format_simple(self, response: dict) -> dict:
        """Format response with simple output."""
        print(f"Found {response['n_sources']} relevant sources\n")
        
        print("Top sources:")
        for i, source in enumerate(response['sources'], 1):
            print(f"  {i}. {source['title'][:80]}... (relevance: {source['relevance']:.3f})")
        
        print("\n" + "="*60)
        print("Context preview:")
        print("="*60)
        print(response['context'][:500] + "...\n")
        
        return response
    
    def _format_for_llm(self, response: dict) -> dict:
        """Format response for LLM consumption."""
        llm_context = create_context_for_llm(response)
        
        print("System Prompt:")
        print("-"*60)
        print(llm_context['system_prompt'])
        print("\n" + "-"*60)
        print("User Prompt:")
        print("-"*60)
        print(llm_context['user_prompt'])
        print("-"*60 + "\n")
        
        return llm_context
    
    def interactive_mode(self):
        """Run in interactive mode where user can ask multiple questions."""
        print("\n🤖 Interactive RAG Query Mode")
        print("="*60)
        print("Ask questions about the scientific papers in the database.")
        print("Type 'quit' or 'exit' to stop.")
        print("Type 'help' for available commands.")
        print("="*60 + "\n")
        
        while True:
            try:
                question = input("❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if question.lower() == 'help':
                    self._print_help()
                    continue
                
                if question.lower() == 'stats':
                    self._print_stats()
                    continue
                
                if not question:
                    continue
                
                # Parse optional parameters from question
                n_results = 5
                output_format = "detailed"
                
                # Allow format specification like: "question [n=10] [format=simple]"
                if '[' in question and ']' in question:
                    import re
                    params = re.findall(r'\[([^\]]+)\]', question)
                    for param in params:
                        if '=' in param:
                            key, value = param.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            if key == 'n':
                                n_results = int(value)
                            elif key == 'format':
                                output_format = value
                    
                    # Remove parameters from question
                    question = re.sub(r'\[([^\]]+)\]', '', question).strip()
                
                # Query and display results
                self.query(question, n_results=n_results, output_format=output_format)
                
                print("\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
    
    def _print_help(self):
        """Print help information."""
        print("\n" + "="*60)
        print("Available Commands:")
        print("="*60)
        print("  <question>              - Ask a question")
        print("  <question> [n=10]       - Retrieve 10 results")
        print("  <question> [format=simple]  - Use simple format")
        print("  stats                   - Show database statistics")
        print("  help                    - Show this help message")
        print("  quit/exit/q             - Exit the program")
        print("\nAvailable formats:")
        print("  detailed (default)      - Full details with all metadata")
        print("  simple                  - Brief summary")
        print("  llm                     - Formatted for LLM consumption")
        print("="*60 + "\n")
    
    def _print_stats(self):
        """Print database statistics."""
        stats = self.rag.get_statistics()
        print("\n" + "="*60)
        print("Database Statistics:")
        print("="*60)
        print(f"  Collection: {stats['collection_name']}")
        print(f"  Total chunks: {stats['total_chunks']:,}")
        print(f"  Embedding model: {stats['embedding_model']}")
        print(f"  Storage location: {stats['persist_directory']}")
        print("="*60 + "\n")
    
    def batch_query(self, questions_file: str, output_file: str):
        """
        Process multiple questions from a file.
        
        Args:
            questions_file: Path to file with questions (one per line)
            output_file: Path to output JSON file
        """
        print(f"\n📂 Loading questions from: {questions_file}")
        
        with open(questions_file, 'r') as f:
            questions = [line.strip() for line in f if line.strip()]
        
        print(f"   Found {len(questions)} questions")
        print(f"   Processing...\n")
        
        results = []
        for i, question in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] {question}")
            response = self.rag.answer_question(question, n_context_chunks=5)
            results.append({
                'question': question,
                'sources': response['sources'],
                'context': response['context']
            })
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Query the RAG system for scientific papers"
    )
    parser.add_argument(
        '--collection-name',
        type=str,
        default='scientific_papers',
        help='ChromaDB collection name'
    )
    parser.add_argument(
        '--persist-dir',
        type=str,
        default='./chroma_db',
        help='Directory where ChromaDB is persisted'
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='allenai/specter2',
        help='Embedding model to use'
    )
    parser.add_argument(
        '--backup-model',
        type=str,
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='Backup embedding model'
    )
    parser.add_argument(
        '--question',
        type=str,
        help='Single question to ask (non-interactive mode)'
    )
    parser.add_argument(
        '--n-results',
        type=int,
        default=5,
        help='Number of results to retrieve'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['detailed', 'simple', 'llm'],
        default='detailed',
        help='Output format'
    )
    parser.add_argument(
        '--batch',
        type=str,
        help='File with questions (one per line) for batch processing'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='batch_results.json',
        help='Output file for batch processing'
    )
    
    args = parser.parse_args()
    
    # Initialize interface
    interface = RAGQueryInterface(
        collection_name=args.collection_name,
        persist_directory=args.persist_dir,
        embedding_model=args.embedding_model,
        backup_model=args.backup_model
    )
    
    # Run in appropriate mode
    if args.batch:
        interface.batch_query(args.batch, args.output)
    elif args.question:
        interface.query(args.question, n_results=args.n_results, output_format=args.format)
    else:
        interface.interactive_mode()


if __name__ == "__main__":
    main()
