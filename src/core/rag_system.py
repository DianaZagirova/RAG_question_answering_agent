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
        backup_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize the RAG system.
        
        Args:
            collection_name: Name for the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_model: Model for creating embeddings (scientific papers optimized)
            backup_embedding_model: Fallback model if primary is unavailable
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
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
        filter_dict: Optional[Dict] = None
    ) -> Dict:
        """
        Query the RAG system for relevant chunks.
        
        Args:
            query_text: The question or query
            n_results: Number of results to return
            filter_dict: Optional metadata filters (e.g., {'section': 'Results'})
            
        Returns:
            Dictionary containing results with documents, metadata, and distances
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=filter_dict,
            include=['documents', 'metadatas', 'distances']
        )
        
        return {
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0],
            'distances': results['distances'][0],
            'ids': results['ids'][0]
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
        metadata_filter: Optional[Dict] = None
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
        results = self.query_with_reranking(
            query_text=question,
            n_results=n_context_chunks,
            rerank_top_k=n_context_chunks * 2,
            filter_dict=metadata_filter
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
    
    def get_statistics(self) -> Dict:
        """Get statistics about the RAG database."""
        count = self.collection.count()
        
        return {
            'collection_name': self.collection_name,
            'total_chunks': count,
            'embedding_model': self.model_name,
            'persist_directory': str(self.persist_directory)
        }
    
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
