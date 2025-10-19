"""
Specialized query interface optimized for aging research questions.
Implements query expansion, multi-query retrieval, and question-specific strategies.
"""
import argparse
import json
from typing import List, Dict, Tuple
from rag_system import ScientificRAG, create_context_for_llm


class AgingPapersQueryInterface:
    """
    Specialized interface for querying aging research papers.
    Optimized for the 9 critical questions about aging biomarkers,
    mechanisms, and species-specific phenomena.
    """
    
    def __init__(
        self,
        collection_name: str = "scientific_papers",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "allenai/specter2",
        backup_model: str = "sentence-transformers/all-mpnet-base-v2"
    ):
        """Initialize the specialized query interface."""
        print("Loading RAG system with optimal configuration...")
        self.rag = ScientificRAG(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_model=embedding_model,
            backup_embedding_model=backup_model
        )
        
        # Query expansion mappings
        self.query_expansions = {
            'biomarker': ['biomarker', 'marker', 'indicator', 'predictor', 'measure', 'measurement'],
            'mechanism': ['mechanism', 'pathway', 'process', 'molecular basis', 'cellular mechanism'],
            'aging': ['aging', 'senescence', 'age-related', 'longevity', 'lifespan', 'life span'],
            'intervention': ['intervention', 'treatment', 'therapy', 'drug', 'compound', 'supplementation'],
            'species': ['species', 'animal', 'organism', 'vertebrate', 'mammal', 'bird'],
        }
        
        # Question-specific configurations
        self.question_configs = {
            'biomarker': {'n_results': 12, 'keywords': ['biomarker', 'mortality', 'health', 'predictor']},
            'mechanism': {'n_results': 10, 'keywords': ['mechanism', 'molecular', 'cellular', 'pathway']},
            'intervention': {'n_results': 8, 'keywords': ['intervention', 'longevity', 'extend', 'increase']},
            'theory': {'n_results': 10, 'keywords': ['theory', 'hypothesis', 'model', 'framework']},
            'species': {'n_results': 15, 'keywords': []},  # Broad search for specific species
        }
    
    def expand_query(self, query: str, expansion_type: str = 'general') -> List[str]:
        """
        Expand query with synonyms and related terms.
        Returns list of query variations.
        """
        queries = [query]
        
        # Add variations based on expansion type
        if expansion_type == 'biomarker':
            queries.extend([
                query.replace('biomarker', 'marker'),
                query.replace('biomarker', 'predictor'),
                query.replace('suggest', 'identify'),
                query.replace('suggest', 'propose'),
            ])
        elif expansion_type == 'mechanism':
            queries.extend([
                query.replace('mechanism', 'pathway'),
                query.replace('mechanism', 'molecular basis'),
                query.replace('suggest', 'describe'),
                query.replace('suggest', 'explain'),
            ])
        elif expansion_type == 'species':
            # Keep original for specific species queries
            pass
        
        # Remove duplicates
        return list(set(q for q in queries if q != query))[:3]  # Max 3 variations
    
    def query_with_strategy(
        self,
        question: str,
        question_type: str = 'general',
        use_multi_query: bool = True
    ) -> Dict:
        """
        Query using optimized strategy for question type.
        
        Args:
            question: The question to ask
            question_type: Type of question (biomarker, mechanism, species, etc.)
            use_multi_query: Whether to use multi-query retrieval
            
        Returns:
            Enhanced results dictionary
        """
        config = self.question_configs.get(question_type, {'n_results': 10, 'keywords': []})
        
        all_results = []
        seen_ids = set()
        
        # Main query
        results = self.rag.query(
            query_text=question,
            n_results=config['n_results']
        )
        
        for i, (doc, meta, dist, id_) in enumerate(zip(
            results['documents'],
            results['metadatas'],
            results['distances'],
            results['ids']
        )):
            if id_ not in seen_ids:
                all_results.append({
                    'text': doc,
                    'metadata': meta,
                    'distance': dist,
                    'id': id_,
                    'relevance': 1 - dist
                })
                seen_ids.add(id_)
        
        # Multi-query retrieval (if enabled)
        if use_multi_query and question_type != 'species':
            expanded_queries = self.expand_query(question, question_type)
            
            for expanded_q in expanded_queries[:2]:  # Use top 2 expansions
                exp_results = self.rag.query(
                    query_text=expanded_q,
                    n_results=5  # Fewer results for expanded queries
                )
                
                for doc, meta, dist, id_ in zip(
                    exp_results['documents'],
                    exp_results['metadatas'],
                    exp_results['distances'],
                    exp_results['ids']
                ):
                    if id_ not in seen_ids:
                        all_results.append({
                            'text': doc,
                            'metadata': meta,
                            'distance': dist * 1.1,  # Slightly penalize expanded queries
                            'id': id_,
                            'relevance': (1 - dist) * 0.9
                        })
                        seen_ids.add(id_)
        
        # Sort by relevance
        all_results.sort(key=lambda x: x['relevance'], reverse=True)
        
        # Return top N
        top_results = all_results[:config['n_results']]
        
        return {
            'question': question,
            'question_type': question_type,
            'documents': [r['text'] for r in top_results],
            'metadatas': [r['metadata'] for r in top_results],
            'distances': [r['distance'] for r in top_results],
            'ids': [r['id'] for r in top_results],
            'n_sources': len(top_results),
            'used_multi_query': use_multi_query and len(expanded_queries) > 0
        }
    
    def answer_aging_question(
        self,
        question_number: int,
        question_text: str,
        doi: str = None
    ) -> Dict:
        """
        Answer one of the 9 critical aging questions.
        
        Args:
            question_number: Q1-Q9
            question_text: The full question text
            doi: Optional DOI to search within specific paper
            
        Returns:
            Enhanced answer with context and metadata
        """
        # Determine question type
        question_type_map = {
            1: 'biomarker',
            2: 'mechanism',
            3: 'intervention',
            4: 'theory',
            5: 'biomarker',
            6: 'species',
            7: 'species',
            8: 'species',
            9: 'species',
        }
        
        question_type = question_type_map.get(question_number, 'general')
        
        # Query with optimized strategy
        results = self.query_with_strategy(
            question=question_text,
            question_type=question_type,
            use_multi_query=True
        )
        
        # Format context
        context_parts = []
        sources = []
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'],
            results['metadatas'],
            results['distances']
        ), 1):
            relevance = 1 - distance
            
            # Format context with metadata
            header = f"[Source {i}]"
            if metadata.get('title'):
                header += f" {metadata['title'][:80]}"
            if metadata.get('doi'):
                header += f" | DOI: {metadata['doi']}"
            if metadata.get('section'):
                header += f" | Section: {metadata['section']}"
            header += f" | Relevance: {relevance:.3f}"
            
            context_parts.append(f"{header}\n{doc}\n")
            
            sources.append({
                'doi': metadata.get('doi', 'unknown'),
                'title': metadata.get('title', 'unknown'),
                'section': metadata.get('section', 'unknown'),
                'relevance': relevance
            })
        
        context = "\n---\n".join(context_parts)
        
        return {
            'question_number': question_number,
            'question': question_text,
            'question_type': question_type,
            'context': context,
            'sources': sources,
            'n_sources': len(sources),
            'strategy_used': 'multi-query' if results.get('used_multi_query') else 'single-query'
        }
    
    def batch_answer_all_questions(
        self,
        doi: str = None,
        output_file: str = "aging_questions_results.json"
    ) -> Dict:
        """
        Answer all 9 critical aging questions at once.
        
        Args:
            doi: Optional DOI to analyze specific paper
            output_file: Where to save results
            
        Returns:
            Dictionary with all answers
        """
        questions = [
            (1, "Does the paper suggest an aging biomarker that is quantitatively shown to reflect aging pace or health state?"),
            (2, "Does the paper suggest a molecular mechanism of aging?"),
            (3, "Does the paper suggest a longevity intervention to test?"),
            (4, "Does the paper claim that aging cannot be reversed?"),
            (5, "Does the paper suggest a biomarker that predicts maximal lifespan differences between species?"),
            (6, "Does the paper explain why the naked mole rat can live 40+ years despite its small size?"),
            (7, "Does the paper explain why birds live much longer than mammals on average?"),
            (8, "Does the paper explain why large animals live longer than small ones?"),
            (9, "Does the paper explain why calorie restriction increases the lifespan of vertebrates?"),
        ]
        
        print(f"\n{'='*60}")
        print("Answering All 9 Critical Aging Questions")
        print(f"{'='*60}\n")
        
        all_answers = {}
        
        for q_num, q_text in questions:
            print(f"Q{q_num}: {q_text[:60]}...")
            
            answer = self.answer_aging_question(q_num, q_text, doi)
            all_answers[f"Q{q_num}"] = answer
            
            print(f"  ✓ Retrieved {answer['n_sources']} sources ({answer['strategy_used']})")
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(all_answers, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
        print(f"{'='*60}\n")
        
        return all_answers


def main():
    parser = argparse.ArgumentParser(
        description="Query aging research papers with optimized strategies"
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
        help='ChromaDB persist directory'
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='allenai/specter2',
        help='Primary embedding model'
    )
    parser.add_argument(
        '--backup-model',
        type=str,
        default='sentence-transformers/all-mpnet-base-v2',
        help='Backup embedding model'
    )
    parser.add_argument(
        '--question',
        type=str,
        help='Single question to ask'
    )
    parser.add_argument(
        '--question-type',
        type=str,
        choices=['biomarker', 'mechanism', 'intervention', 'theory', 'species', 'general'],
        default='general',
        help='Type of question for optimized retrieval'
    )
    parser.add_argument(
        '--all-questions',
        action='store_true',
        help='Answer all 9 critical aging questions'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='aging_questions_results.json',
        help='Output file for results'
    )
    
    args = parser.parse_args()
    
    # Initialize interface
    interface = AgingPapersQueryInterface(
        collection_name=args.collection_name,
        persist_directory=args.persist_dir,
        embedding_model=args.embedding_model,
        backup_model=args.backup_model
    )
    
    if args.all_questions:
        # Answer all 9 questions
        interface.batch_answer_all_questions(output_file=args.output)
    elif args.question:
        # Answer single question
        results = interface.query_with_strategy(
            question=args.question,
            question_type=args.question_type,
            use_multi_query=True
        )
        
        print(f"\n{'='*60}")
        print(f"Question: {args.question}")
        print(f"Type: {args.question_type}")
        print(f"{'='*60}\n")
        
        print(f"Retrieved {results['n_sources']} sources\n")
        
        for i, (doc, meta, dist) in enumerate(zip(
            results['documents'][:5],
            results['metadatas'][:5],
            results['distances'][:5]
        ), 1):
            print(f"[Source {i}] {meta.get('title', 'Unknown')[:60]}...")
            print(f"  Section: {meta.get('section')}, Relevance: {1-dist:.3f}")
            print(f"  {doc[:150]}...\n")
    else:
        print("Please specify --question or --all-questions")
        print("\nExample usage:")
        print("  # Single question:")
        print("  python query_aging_papers.py --question 'Does the paper suggest an aging biomarker?' --question-type biomarker")
        print("\n  # All 9 questions:")
        print("  python query_aging_papers.py --all-questions")


if __name__ == "__main__":
    main()
