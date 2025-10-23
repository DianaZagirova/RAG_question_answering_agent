"""
RAG System for Scientific Papers using ChromaDB and scientific embeddings.
Optimized for high-quality retrieval and answering specific questions.
"""
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple
import numpy as np
from pathlib import Path
import json
import torch
from .query_preprocessor import QueryPreprocessor, preprocess_for_scientific_papers


class SentenceTransformerEmbeddingFunction:
    """Custom embedding function for ChromaDB using SentenceTransformers."""
    
    def __init__(self, model, batch_size=32):
        self.model = model
        self.batch_size = batch_size
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for input texts with GPU acceleration."""
        embeddings = self.model.encode(
            input,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.model.device  # Use model's device (GPU if available)
        )
        return embeddings.tolist()


class ScientificRAG:
    """
    Retrieval-Augmented Generation system for scientific papers.
    Uses ChromaDB for vector storage and retrieval.
    """
    
    def __init__(
        self,
        collection_name: str = "scientific_papers_optimal",
        persist_directory: str = "./chroma_db_optimal",
        embedding_model: str = "allenai/specter2",
        backup_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_query_preprocessing: bool = True
    ):
        """
        Initialize the RAG system.
        
        Args:
            collection_name: Name for the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_model: Model for creating embeddings (scientific papers optimized)
            backup_embedding_model: Fallback model if primary is unavailable
            use_query_preprocessing: Whether to use advanced query preprocessing
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        self.use_query_preprocessing = use_query_preprocessing
        # Note: LLM client will be set later via set_llm_client() for advanced preprocessing
        self.query_preprocessor = QueryPreprocessor(use_cache=True) if use_query_preprocessing else None
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Try to load scientific model, fallback to general model
        print(f"Loading embedding model: {embedding_model}")
        
        # Detect GPU availability
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        if device == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
        
        try:
            self.embedding_model = SentenceTransformer(embedding_model, device=device)
            self.model_name = embedding_model
            print(f"✓ Loaded {embedding_model}")
        except Exception as e:
            print(f"⚠ Could not load {embedding_model}, falling back to {backup_embedding_model}")
            print(f"  Error: {e}")
            self.embedding_model = SentenceTransformer(backup_embedding_model, device=device)
            self.model_name = backup_embedding_model
            print(f"✓ Loaded {backup_embedding_model}")
        
        # Create embedding function wrapper with batch processing
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            self.embedding_model,
            batch_size=64  # Larger batch size for GPU
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            print(f"✓ Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            print(f"✓ Created new collection: {collection_name}")
    
    def add_chunks(
        self,
        chunks: List[Dict],
        batch_size: int = 100
    ) -> int:
        """
        Add chunks to the vector database in batches.
        
        Args:
            chunks: List of chunk dictionaries with 'text', 'metadata', and 'chunk_id'
            batch_size: Number of chunks to process at once
            
        Returns:
            Number of chunks added
        """
        total_added = 0
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Prepare batch data
            ids = [f"{chunk['metadata'].get('doi', 'unknown')}_{chunk['chunk_id']}" 
                   for chunk in batch]
            documents = [chunk['text'] for chunk in batch]
            metadatas = [chunk['metadata'] for chunk in batch]
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            total_added += len(batch)
            
            if (i + batch_size) % 1000 == 0:
                print(f"  Added {total_added} chunks...")
        
        return total_added
    
    def query(
        self,
        query_text: str,
        n_results: int = 10,
        filter_dict: Optional[Dict] = None,
        use_preprocessing: Optional[bool] = None,
        use_multi_query: bool = False,
        predefined_queries: Optional[List[str]] = None
    ) -> Dict:
        """
        Query the RAG system for relevant chunks with optional preprocessing.
        
        Args:
            query_text: The question or query
            n_results: Number of results to return
            filter_dict: Optional metadata filters (e.g., {'section': 'Results'})
            use_preprocessing: Override default preprocessing setting
            use_multi_query: Use multiple query variants for better coverage
            predefined_queries: List of predefined query variants (bypasses LLM enhancement)
            
        Returns:
            Dictionary containing results with documents, metadata, and distances
        """
        # If predefined queries provided, use them directly (no LLM enhancement)
        if predefined_queries:
            return self._query_with_predefined_variants(predefined_queries, n_results, filter_dict)
        
        # Apply query preprocessing if enabled
        if use_preprocessing is None:
            use_preprocessing = self.use_query_preprocessing
        
        # Multi-query retrieval: use multiple query variants
        if use_multi_query and use_preprocessing and self.query_preprocessor and self.query_preprocessor.llm_client:
            return self._multi_query_retrieval(query_text, n_results, filter_dict)
        
        # Single query with preprocessing
        processed_query = query_text
        if use_preprocessing and self.query_preprocessor:
            # Use LLM enhancement if available (best for matching scientific text)
            if self.query_preprocessor.llm_client:
                preprocessed = self.query_preprocessor.preprocess_query(
                    query_text,
                    use_llm_enhancement=True,
                    use_expansion=False,
                    use_contextualization=False,
                    use_hyde=False
                )
                processed_query = preprocessed['processed_query']
            else:
                # Fallback to simple preprocessing
                processed_query = preprocess_for_scientific_papers(query_text)
        
        results = self.collection.query(
            query_texts=[processed_query],
            n_results=n_results,
            where=filter_dict,
            include=['documents', 'metadatas', 'distances']
        )
        
        return {
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0],
            'distances': results['distances'][0],
            'ids': results['ids'][0],
            'processed_query': processed_query,
            'query_variants_used': 1
        }
    
    def _query_with_predefined_variants(
        self,
        predefined_queries: List[str],
        n_results: int,
        filter_dict: Optional[Dict]
    ) -> Dict:
        """
        Query using predefined query variants (no LLM enhancement).
        Retrieves top chunks from each variant and returns unique results.
        
        Args:
            predefined_queries: List of predefined query strings
            n_results: Total number of unique results to return
            filter_dict: Optional metadata filters
            
        Returns:
            Dictionary with unique documents, metadata, and distances
        """
        all_results = {}  # id -> (doc, metadata, distance, query_index)
        
        # Retrieve from each predefined query variant
        per_query_results = 12  # Retrieve top 12 from each query
        
        for query_idx, query_variant in enumerate(predefined_queries):
            results = self.collection.query(
                query_texts=[query_variant],
                n_results=per_query_results,
                where=filter_dict,
                include=['documents', 'metadatas', 'distances']
            )
            
            for doc, meta, dist, doc_id in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0],
                results['ids'][0]
            ):
                # Keep best score for each unique document
                if doc_id not in all_results or dist < all_results[doc_id][2]:
                    all_results[doc_id] = (doc, meta, dist, query_idx)
        
        # Sort by distance and take top n_results unique chunks
        sorted_results = sorted(all_results.items(), key=lambda x: x[1][2])[:n_results]
        
        # Unpack results
        documents = [r[1][0] for r in sorted_results]
        metadatas = [r[1][1] for r in sorted_results]
        distances = [r[1][2] for r in sorted_results]
        ids = [r[0] for r in sorted_results]
        
        return {
            'documents': documents,
            'metadatas': metadatas,
            'distances': distances,
            'ids': ids,
            'processed_query': f"Predefined queries ({len(predefined_queries)} variants)",
            'query_variants_used': len(predefined_queries),
            'unique_chunks': len(documents)
        }
    
    def _multi_query_retrieval(
        self,
        query_text: str,
        n_results: int,
        filter_dict: Optional[Dict]
    ) -> Dict:
        """
        Multi-query retrieval: Run multiple query variants and merge results.
        This improves coverage by retrieving from different perspectives.
        
        Strategy:
        1. Original question
        2. LLM-enhanced (scientific style)
        3. HyDE (hypothetical answer)
        4. Expanded with synonyms
        """
        # Generate query variants
        preprocessed = self.query_preprocessor.preprocess_query(
            query_text,
            use_llm_enhancement=True,
            use_expansion=True,
            use_contextualization=True,
            use_hyde=True
        )
        
        query_variants = []
        
        # 1. Enhanced query (primary)
        if preprocessed.get('enhanced_query'):
            query_variants.append(('enhanced', preprocessed['enhanced_query']))
        
        # 2. HyDE document (hypothetical answer)
        if preprocessed.get('hyde_document'):
            query_variants.append(('hyde', preprocessed['hyde_document']))
        
        # 3. Original query
        query_variants.append(('original', query_text))
        
        # 4. Expanded query (with synonyms)
        if preprocessed.get('processed_query') != query_text:
            query_variants.append(('expanded', preprocessed['processed_query']))
        
        # Retrieve from each variant
        all_results = {}  # id -> (doc, metadata, distance, source_variant)
        
        # Retrieve more from each variant, then merge
        per_variant_results = max(n_results // len(query_variants), 5)
        
        for variant_name, variant_query in query_variants:
            results = self.collection.query(
                query_texts=[variant_query],
                n_results=per_variant_results * 2,  # Get more to ensure diversity
                where=filter_dict,
                include=['documents', 'metadatas', 'distances']
            )
            
            for doc, meta, dist, doc_id in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0],
                results['ids'][0]
            ):
                # Keep best score for each unique document
                if doc_id not in all_results or dist < all_results[doc_id][2]:
                    all_results[doc_id] = (doc, meta, dist, variant_name)
        
        # Sort by distance and take top n_results
        sorted_results = sorted(all_results.items(), key=lambda x: x[1][2])[:n_results]
        
        # Unpack results
        documents = [r[1][0] for r in sorted_results]
        metadatas = [r[1][1] for r in sorted_results]
        distances = [r[1][2] for r in sorted_results]
        ids = [r[0] for r in sorted_results]
        
        return {
            'documents': documents,
            'metadatas': metadatas,
            'distances': distances,
            'ids': ids,
            'processed_query': preprocessed.get('enhanced_query', query_text),
            'query_variants_used': len(query_variants),
            'query_variants': [v[0] for v in query_variants]
        }
    
    def query_with_reranking(
        self,
        query_text: str,
        n_results: int = 10,
        rerank_top_k: int = 20,
        filter_dict: Optional[Dict] = None
    ) -> Dict:
        """
        Query with two-stage retrieval: initial retrieval + reranking.
        
        Args:
            query_text: The question or query
            n_results: Final number of results to return
            rerank_top_k: Number of initial results to retrieve for reranking
            filter_dict: Optional metadata filters
            
        Returns:
            Dictionary containing reranked results
        """
        # First stage: retrieve more candidates
        initial_results = self.query(
            query_text=query_text,
            n_results=rerank_top_k,
            filter_dict=filter_dict
        )
        
        # Second stage: rerank using cross-encoder or better scoring
        # For now, we'll use the initial ranking (can be enhanced with cross-encoder)
        
        return {
            'documents': initial_results['documents'][:n_results],
            'metadatas': initial_results['metadatas'][:n_results],
            'distances': initial_results['distances'][:n_results],
            'ids': initial_results['ids'][:n_results]
        }
    
    def answer_question(
        self,
        question: str,
        n_context_chunks: int = 5,
        include_metadata: bool = True,
        metadata_filter: Optional[Dict] = None,
        use_multi_query: bool = False,
        predefined_queries: Optional[List[str]] = None
    ) -> Dict:
        """
        Answer a question using retrieved context.
        Returns formatted context that can be used with an LLM.
        
        Args:
            question: The question to answer
            n_context_chunks: Number of context chunks to retrieve
            include_metadata: Whether to include metadata in the context
            
        Returns:
            Dictionary with question, context, and metadata
        """
        # Retrieve relevant chunks
        results = self.query(
            query_text=question,
            n_results=n_context_chunks,
            filter_dict=metadata_filter,
            use_multi_query=use_multi_query,
            predefined_queries=predefined_queries
        )
        
        # Format context
        context_parts = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'],
            results['metadatas'],
            results['distances']
        )):
            if include_metadata:
                context_header = f"[Source {i+1}] "
                if 'title' in metadata:
                    context_header += f"Title: {metadata['title']} | "
                if 'doi' in metadata:
                    context_header += f"DOI: {metadata['doi']} | "
                if 'section' in metadata:
                    context_header += f"Section: {metadata['section']} | "
                context_header += f"Relevance: {1 - distance:.3f}"
                
                context_parts.append(f"{context_header}\n{doc}\n")
            else:
                context_parts.append(doc)
        
        context = "\n---\n".join(context_parts)
        
        return {
            'question': question,
            'context': context,
            'n_sources': len(results['documents']),
            'sources': [
                {
                    'doi': meta.get('doi', 'unknown'),
                    'title': meta.get('title', 'unknown'),
                    'section': meta.get('section', 'unknown'),
                    'relevance': 1 - dist
                }
                for meta, dist in zip(results['metadatas'], results['distances'])
            ]
        }
    
    def set_llm_client(self, llm_client):
        """
        Set LLM client for advanced query preprocessing.
        
        Args:
            llm_client: LLM client instance (e.g., AzureOpenAIClient)
        """
        if self.query_preprocessor:
            self.query_preprocessor.llm_client = llm_client
            print("✓ LLM client connected to query preprocessor for enhanced retrieval")
    
    def get_statistics(self) -> Dict:
        """Get statistics about the RAG database."""
        count = self.collection.count()
        
        stats = {
            'collection_name': self.collection_name,
            'total_chunks': count,
            'embedding_model': self.model_name,
            'persist_directory': str(self.persist_directory)
        }
        
        # Add cache stats if available
        if self.query_preprocessor:
            stats['query_cache'] = self.query_preprocessor.get_cache_stats()
        
        return stats
    
    def reset_collection(self):
        """Delete and recreate the collection (use with caution!)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✓ Collection {self.collection_name} reset")


def create_context_for_llm(
    rag_response: Dict,
    system_prompt: Optional[str] = None
) -> Dict:
    """
    Create a formatted prompt for an LLM using RAG results.
    
    Args:
        rag_response: Output from answer_question()
        system_prompt: Optional custom system prompt
        
    Returns:
        Dictionary with system and user prompts
    """
    if system_prompt is None:
        system_prompt = """You are a scientific research assistant specializing in analyzing biomedical literature.
Your task is to answer questions based on the provided scientific paper excerpts.

Guidelines:
1. Base your answers strictly on the provided context
2. Cite sources by their source numbers [Source N]
3. If the context doesn't contain enough information, say so
4. Use scientific terminology appropriately
5. Distinguish between findings, hypotheses, and speculation in the papers
6. If multiple papers are cited, synthesize information across them"""
    
    user_prompt = f"""Question: {rag_response['question']}

Context from scientific papers:

{rag_response['context']}

Please answer the question based on the provided context. If you reference specific information, cite the source number."""
    
    return {
        'system_prompt': system_prompt,
        'user_prompt': user_prompt,
        'sources': rag_response['sources']
    }


if __name__ == "__main__":
    # Test the RAG system
    print("Initializing RAG system...")
    rag = ScientificRAG(
        collection_name="test_papers",
        persist_directory="./test_chroma_db"
    )
    
    # Test with sample chunks
    test_chunks = [
        {
            'text': 'Aging is characterized by progressive decline in cellular function and increased susceptibility to disease.',
            'chunk_id': 0,
            'metadata': {
                'doi': '10.1234/test1',
                'title': 'The Biology of Aging',
                'section': 'Introduction'
            }
        },
        {
            'text': 'Several biomarkers have been proposed for measuring biological age, including telomere length and DNA methylation patterns.',
            'chunk_id': 1,
            'metadata': {
                'doi': '10.1234/test1',
                'title': 'The Biology of Aging',
                'section': 'Results'
            }
        }
    ]
    
    print("\nAdding test chunks...")
    rag.add_chunks(test_chunks)
    
    print("\nQuerying...")
    response = rag.answer_question("What biomarkers are used for aging?")
    
    print("\nResponse:")
    print(json.dumps(response, indent=2))
    
    print("\nStatistics:")
    print(json.dumps(rag.get_statistics(), indent=2))
