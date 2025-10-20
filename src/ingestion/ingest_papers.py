"""
Main ingestion pipeline for processing papers from SQLite database into RAG system.
"""
import sqlite3
import json
import sys
from pathlib import Path
from typing import Optional, Generator, Dict, List, Set
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.text_preprocessor import TextPreprocessor
from core.chunker import ScientificChunker, Chunk
from core.rag_system import ScientificRAG


class PaperIngestionPipeline:
    """Pipeline for ingesting papers from database into RAG system."""
    
    def __init__(
        self,
        db_path: str,
        rag_collection_name: str = "scientific_papers",
        rag_persist_dir: str = "./chroma_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "allenai/specter2",
        backup_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        allowed_dois: Optional[List[str]] = None,
        allowed_pmids: Optional[List[str]] = None
    ):
        """
        Initialize the ingestion pipeline.
        
        Args:
            db_path: Path to the papers.db SQLite database
            rag_collection_name: Name for the ChromaDB collection
            rag_persist_dir: Directory to persist ChromaDB
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            embedding_model: Primary embedding model (scientific papers optimized)
            backup_model: Fallback embedding model
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        # Initialize components
        print("Initializing pipeline components...")
        self.preprocessor = TextPreprocessor()
        print("✓ Text preprocessor ready")
        
        self.chunker = ScientificChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        print(f"✓ Chunker ready (size={chunk_size}, overlap={chunk_overlap})")
        
        self.rag = ScientificRAG(
            collection_name=rag_collection_name,
            persist_directory=rag_persist_dir,
            embedding_model=embedding_model,
            backup_embedding_model=backup_model
        )
        print("✓ RAG system ready\n")
        
        self.stats = {
            'total_papers': 0,
            'processed_papers': 0,
            'skipped_papers': 0,
            'total_chunks': 0,
            'errors': []
        }
        self.allowed_dois = set(allowed_dois) if allowed_dois else None
        self.allowed_pmids = set(allowed_pmids) if allowed_pmids else None
        
        # Cache for already-ingested DOIs (batch check at start)
        self._ingested_dois_cache: Optional[Set[str]] = None
    
    def fetch_papers(
        self,
        batch_size: int = 100,
        limit: Optional[int] = None
    ) -> Generator[Dict, None, None]:
        """
        Fetch papers from database in batches.
        
        Args:
            batch_size: Number of papers to fetch at once
            limit: Maximum number of papers to process (None for all)
            
        Yields:
            Paper dictionaries with relevant fields
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable column access by name
        cursor = conn.cursor()
        
        # Get total count
        if self.allowed_dois or self.allowed_pmids:
            cursor.execute("CREATE TEMP TABLE IF NOT EXISTS validated_ids (doi TEXT, pmid TEXT)")
            if self.allowed_dois:
                cursor.executemany("INSERT INTO validated_ids (doi, pmid) VALUES (?, NULL)", [(d,) for d in self.allowed_dois])
            if self.allowed_pmids:
                cursor.executemany("INSERT INTO validated_ids (doi, pmid) VALUES (NULL, ?)", [(p,) for p in self.allowed_pmids])
            cursor.execute(
                """
                SELECT COUNT(*) FROM papers 
                WHERE ((full_text_sections IS NOT NULL AND full_text_sections != '') 
                   OR (full_text IS NOT NULL AND full_text != ''))
                  AND EXISTS (
                    SELECT 1 FROM validated_ids v 
                    WHERE (v.doi IS NOT NULL AND v.doi = papers.doi)
                       OR (v.pmid IS NOT NULL AND v.pmid = papers.pmid)
                  )
                """
            )
            total = cursor.fetchone()[0]
        else:
            cursor.execute("SELECT COUNT(*) FROM papers WHERE (full_text_sections IS NOT NULL AND full_text_sections != '') OR (full_text IS NOT NULL AND full_text != '')")
            total = cursor.fetchone()[0]
        self.stats['total_papers'] = min(total, limit) if limit else total
        
        # Fetch papers in batches
        if self.allowed_dois or self.allowed_pmids:
            query = """
            SELECT doi, pmid, title, abstract, full_text, full_text_sections, 
                   authors, year, journal, topic_name, topic_field
            FROM papers 
            WHERE ((full_text_sections IS NOT NULL AND full_text_sections != '') 
               OR (full_text IS NOT NULL AND full_text != ''))
              AND EXISTS (
                SELECT 1 FROM validated_ids v 
                WHERE (v.doi IS NOT NULL AND v.doi = papers.doi)
                   OR (v.pmid IS NOT NULL AND v.pmid = papers.pmid)
              )
            """
        else:
            query = """
            SELECT doi, pmid, title, abstract, full_text, full_text_sections, 
                   authors, year, journal, topic_name, topic_field
            FROM papers 
            WHERE (full_text_sections IS NOT NULL AND full_text_sections != '') 
               OR (full_text IS NOT NULL AND full_text != '')
            """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            
            for row in rows:
                yield dict(row)
        
        conn.close()
    
    def _build_ingested_dois_cache(self):
        """Build a cache of all DOIs already in ChromaDB for fast lookup."""
        if self._ingested_dois_cache is not None:
            return
        
        print("Building cache of ingested DOIs...")
        try:
            # Get all DOIs from ChromaDB in one query
            all_data = self.rag.collection.get(include=['metadatas'])
            self._ingested_dois_cache = {
                meta.get('doi', 'Unknown') 
                for meta in all_data['metadatas']
                if meta.get('doi') not in ['Unknown', 'unknown', '#N/A', None]
            }
            print(f"✓ Cached {len(self._ingested_dois_cache):,} ingested DOIs")
        except Exception as e:
            print(f"⚠ Could not build cache: {e}")
            self._ingested_dois_cache = set()
    
    def process_paper(self, paper: Dict) -> Optional[List[Dict]]:
        """
        Process a single paper: preprocess, chunk, and prepare for RAG.
        
        Args:
            paper: Paper dictionary from database
            
        Returns:
            List of chunk dictionaries or None if paper should be skipped
        """
        doi = paper.get('doi') or paper.get('pmid') or 'unknown'
        
        # Fast cache-based duplicate check
        if self._ingested_dois_cache is not None and doi in self._ingested_dois_cache:
            self.stats['skipped_papers'] += 1
            return None
        
        try:
            # Preprocess text
            processed_text = self.preprocessor.preprocess(
                full_text=paper.get('full_text'),
                full_text_sections=paper.get('full_text_sections')
            )
            
            if not processed_text:
                self.stats['skipped_papers'] += 1
                return None
            
            # Create metadata for chunks (filter out None values for ChromaDB compatibility)
            metadata = {
                'doi': doi if doi else 'Unknown',
                'title': paper.get('title') or 'Unknown',
                'abstract': (paper.get('abstract') or '')[:500],  # Truncate abstract
                'authors': paper.get('authors') or 'Unknown',
                'year': str(paper.get('year') or 'Unknown'),  # Convert to string
                'journal': paper.get('journal') or 'Unknown',
                'topic': paper.get('topic_name') or 'Unknown',
                'field': paper.get('topic_field') or 'Unknown'
            }
            
            # Chunk the document
            chunks = self.chunker.chunk_document(processed_text, metadata)
            
            if not chunks:
                self.stats['skipped_papers'] += 1
                return None
            
            # Convert chunks to dictionaries, ensuring no None values in metadata
            chunk_dicts = []
            for chunk in chunks:
                # Filter out None values from chunk metadata
                clean_metadata = {k: v for k, v in chunk.metadata.items() if v is not None}
                # Ensure all values are valid types (str, int, float, bool)
                for key, value in clean_metadata.items():
                    if not isinstance(value, (str, int, float, bool)):
                        clean_metadata[key] = str(value)
                
                chunk_dicts.append({
                    'text': chunk.text,
                    'chunk_id': chunk.chunk_id,
                    'metadata': {
                        **clean_metadata,
                        'chunk_length': len(chunk.text)
                    }
                })
            
            
            self.stats['processed_papers'] += 1
            self.stats['total_chunks'] += len(chunks)
            
            return chunk_dicts
            
        except Exception as e:
            self.stats['errors'].append({
                'doi': doi,
                'error': str(e)
            })
            self.stats['skipped_papers'] += 1
            return None
    
    def run(
        self,
        batch_size: int = 100,
        limit: Optional[int] = None,
        save_progress_every: int = 1000
    ):
        """
        Run the full ingestion pipeline.
        
        Args:
            batch_size: Number of papers to fetch at once
            limit: Maximum number of papers to process
            save_progress_every: Save progress stats every N papers
        """
        print("="*60)
        print("Starting Paper Ingestion Pipeline")
        print("="*60)
        print(f"Database: {self.db_path}")
        print(f"Chunk size: {self.chunker.chunk_size}, Overlap: {self.chunker.chunk_overlap}")
        print(f"Embedding model: {self.rag.model_name}")
        print("="*60 + "\n")
        
        # Build cache of already-ingested DOIs for fast duplicate checking
        self._build_ingested_dois_cache()
        
        # Fetch and process papers
        paper_generator = self.fetch_papers(batch_size=batch_size, limit=limit)
        
        chunks_buffer = []
        buffer_size = 500  # Add to RAG in batches of 500 chunks
        
        with tqdm(total=self.stats['total_papers'], desc="Processing papers") as pbar:
            for paper in paper_generator:
                # Process paper
                chunk_dicts = self.process_paper(paper)
                
                if chunk_dicts:
                    chunks_buffer.extend(chunk_dicts)
                
                # Add to RAG when buffer is full
                if len(chunks_buffer) >= buffer_size:
                    self.rag.add_chunks(chunks_buffer)
                    chunks_buffer = []
                
                pbar.update(1)
                
                # Update progress display
                pbar.set_postfix({
                    'processed': self.stats['processed_papers'],
                    'skipped': self.stats['skipped_papers'],
                    'chunks': self.stats['total_chunks']
                })
                
                # Save progress periodically
                if (self.stats['processed_papers'] + self.stats['skipped_papers']) % save_progress_every == 0:
                    self._save_progress()
        
        # Add remaining chunks
        if chunks_buffer:
            self.rag.add_chunks(chunks_buffer)
        
        # Final statistics
        self._print_final_stats()
        self._save_progress(final=True)
    
    def _save_progress(self, final: bool = False):
        """Save progress statistics to file."""
        filename = "ingestion_complete.json" if final else "ingestion_progress.json"
        
        with open(filename, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        if final:
            print(f"\n✓ Final statistics saved to {filename}")
    
    def _print_final_stats(self):
        """Print final statistics."""
        print("\n" + "="*60)
        print("Ingestion Complete!")
        print("="*60)
        print(f"Total papers in database: {self.stats['total_papers']}")
        print(f"Successfully processed: {self.stats['processed_papers']}")
        print(f"Skipped (no valid text): {self.stats['skipped_papers']}")
        print(f"Total chunks created: {self.stats['total_chunks']}")
        print(f"Average chunks per paper: {self.stats['total_chunks'] / max(self.stats['processed_papers'], 1):.1f}")
        
        if self.stats['errors']:
            print(f"\nErrors encountered: {len(self.stats['errors'])}")
            print("First 5 errors:")
            for error in self.stats['errors'][:5]:
                print(f"  - DOI: {error['doi']}, Error: {error['error']}")
        
        # RAG statistics
        rag_stats = self.rag.get_statistics()
        print(f"\nRAG Database Statistics:")
        print(f"  Collection: {rag_stats['collection_name']}")
        print(f"  Total chunks in DB: {rag_stats['total_chunks']}")
        print(f"  Embedding model: {rag_stats['embedding_model']}")
        print(f"  Persist directory: {rag_stats['persist_directory']}")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest scientific papers into RAG system"
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
        default='scientific_papers',
        help='ChromaDB collection name'
    )
    parser.add_argument(
        '--persist-dir',
        type=str,
        default='./chroma_db',
        help='Directory to persist ChromaDB'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help='Target chunk size in characters'
    )
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=200,
        help='Overlap between chunks'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of papers to process (for testing)'
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
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='Backup embedding model'
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset existing collection before ingesting'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = PaperIngestionPipeline(
        db_path=args.db_path,
        rag_collection_name=args.collection_name,
        rag_persist_dir=args.persist_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
        backup_model=args.backup_model
    )
    
    # Reset if requested
    if args.reset:
        print("⚠ Resetting existing collection...")
        pipeline.rag.reset_collection()
        print()
    
    # Run ingestion
    pipeline.run(limit=args.limit)


if __name__ == "__main__":
    main()
