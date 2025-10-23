#!/usr/bin/env python3
"""
Run RAG system on all validated papers from database.

Pipeline:
1. Load validated papers from evaluations.db
2. For each paper, retrieve full text and abstract from papers.db
3. Load questions from questions JSON file
4. Use RAG (vector search + LLM) to answer each question
5. Store results in database with incremental saves

Features:
- Incremental processing (resume from where it left off)
- Abstract included in LLM context
- Results stored in SQLite database
- Progress tracking and error handling
- Rate limiting for API calls
"""
import sys
import os
from pathlib import Path
import sqlite3
import json
import re
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import argparse
from tqdm import tqdm
from datetime import datetime
import time

from dotenv import load_dotenv
load_dotenv()

# Set CUDA device before importing torch-dependent modules
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    cuda_device = os.getenv('CUDA_DEVICE', '3')
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
    print(f"🔧 Setting CUDA_VISIBLE_DEVICES={cuda_device} (from .env)")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.rag_system import ScientificRAG
from src.core.llm_integration import AzureOpenAIClient, CompleteRAGSystem, OpenAI


@dataclass
class PaperAnswers:
    """Results for one paper."""
    doi: str
    pmid: Optional[str]
    title: str
    abstract: Optional[str]
    validation_result: str
    confidence_score: int
    used_full_text: bool
    n_chunks_retrieved: int
    answers: Dict[str, Dict]  # question_key -> {answer, confidence, reasoning}
    timestamp: str


class RAGPaperProcessor:
    """Process papers through RAG system and store results."""
    
    def __init__(
        self,
        rag_system: ScientificRAG,
        llm_client: AzureOpenAIClient,
        questions_file: str,
        predefined_queries_file: Optional[str] = None
    ):
        self.rag = rag_system
        self.llm = llm_client
        
        # Load questions
        with open(questions_file, 'r') as f:
            self.questions = json.load(f)
        
        # Initialize complete RAG system
        self.complete_rag = CompleteRAGSystem(
            rag_system=self.rag,
            llm_client=self.llm,
            default_n_results=10,
            use_multi_query=False,
            predefined_queries_file=predefined_queries_file
        )
        
        print(f"✓ Loaded {len(self.questions)} questions")
    
    def get_validated_papers(
        self,
        evaluations_db: str,
        papers_db: str,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Get validated papers with full text and abstract.
        
        Criteria: valid OR doubted OR (not_valid AND confidence_score <= 7)
        """
        print("\n" + "="*70)
        print("LOADING VALIDATED PAPERS")
        print("="*70)
        
        # Get validated DOIs from evaluations.db
        eval_conn = sqlite3.connect(evaluations_db)
        eval_cursor = eval_conn.cursor()
        
        query = """
            SELECT doi, pmid, title, result, confidence_score
            FROM paper_evaluations
            WHERE result = 'valid'
               OR result = 'doubted'
               OR (result = 'not_valid' AND confidence_score <= 7)
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        eval_cursor.execute(query)
        validated_papers = eval_cursor.fetchall()
        eval_conn.close()
        
        print(f"✓ Found {len(validated_papers)} validated papers")
        
        # Get full text and abstract from papers.db
        papers_conn = sqlite3.connect(papers_db)
        papers_cursor = papers_conn.cursor()
        
        papers_with_text = []
        
        print("\n📖 Loading full text and abstracts...")
        for doi, pmid, title, result, conf_score in tqdm(validated_papers):
            papers_cursor.execute(
                """
                SELECT doi, pmid, title, abstract, full_text, full_text_sections
                FROM papers
                WHERE doi = ?
                """,
                (doi,),
            )
            
            row = papers_cursor.fetchone()
            if row:
                papers_with_text.append({
                    'doi': row[0],
                    'pmid': row[1],
                    'title': row[2],
                    'abstract': row[3],
                    'full_text': row[4],
                    'full_text_sections': row[5],
                    'validation_result': result,
                    'confidence_score': conf_score
                })
        
        papers_conn.close()
        
        print(f"✓ Loaded {len(papers_with_text)} papers with text")
        return papers_with_text
    
    def get_already_processed(self, results_db: str) -> set:
        """Get set of DOIs already processed."""
        if not Path(results_db).exists():
            return set()
        
        conn = sqlite3.connect(results_db)
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT doi FROM paper_metadata")
        processed = {row[0] for row in cur.fetchall()}
        conn.close()
        
        return processed
    
    def process_paper_batch(
        self,
        paper: Dict,
        temperature: float = 0.2,
        max_tokens: int = 2000
    ) -> PaperAnswers:
        """
        Process one paper through RAG system - BATCH MODE.
        Retrieves chunks for all questions and sends one LLM call.
        
        Args:
            paper: Paper dict with doi, abstract, full_text, etc.
            temperature: LLM temperature
            max_tokens: Max tokens for entire batch response
            
        Returns:
            PaperAnswers with all question answers
        """
        doi = paper['doi']
        abstract = paper.get('abstract', '')
        
        # Retrieve chunks for all questions
        all_contexts = {}
        for q_num, (q_key, q_obj) in enumerate(self.questions.items(), 1):
            # Determine n_results based on question type
            if q_num == 1 or q_num == 5:
                n_results = 12
            elif q_num == 2:
                n_results = 10
            elif q_num == 3:
                n_results = 8
            else:
                n_results = 15
            
            try:
                # Get predefined queries
                predefined_queries = None
                if q_key in self.complete_rag.predefined_queries:
                    predefined_queries = self.complete_rag.predefined_queries[q_key]
                
                # Retrieve context
                metadata_filter = {'doi': doi} if doi else None
                rag_response = self.rag.answer_question(
                    question=q_obj['question'],
                    n_context_chunks=n_results,
                    include_metadata=True,
                    metadata_filter=metadata_filter,
                    use_multi_query=False,
                    predefined_queries=predefined_queries
                )
                
                all_contexts[q_key] = {
                    'context': rag_response['context'],
                    'n_sources': len(rag_response['documents'])
                }
            except Exception as e:
                print(f"  ⚠ Error retrieving for {q_key}: {str(e)[:100]}")
                all_contexts[q_key] = {'context': '', 'n_sources': 0}
        
        # Build batch prompt with all questions
        batch_prompt = self._build_batch_prompt(abstract, all_contexts)
        
        # Single LLM call for all questions
        try:
            response = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=[
                    {"role": "system", "content": "You are a scientific research assistant analyzing aging research papers."},
                    {"role": "user", "content": batch_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            raw_response = response.choices[0].message.content
            answers = self._parse_batch_response(raw_response)
            
        except Exception as e:
            print(f"  ❌ LLM error: {str(e)[:100]}")
            answers = {q_key: {
                'answer': None,
                'confidence': 0.0,
                'reasoning': f'LLM Error: {str(e)}',
                'parse_error': True,
                'n_sources': all_contexts.get(q_key, {}).get('n_sources', 0)
            } for q_key in self.questions.keys()}
        
        # Add n_sources to answers
        for q_key in answers:
            if 'n_sources' not in answers[q_key]:
                answers[q_key]['n_sources'] = all_contexts.get(q_key, {}).get('n_sources', 0)
        
        total_chunks = sum(all_contexts.get(q_key, {}).get('n_sources', 0) for q_key in self.questions.keys())
        
        return PaperAnswers(
            doi=doi,
            pmid=paper.get('pmid'),
            title=paper.get('title', ''),
            abstract=abstract,
            validation_result=paper.get('validation_result', ''),
            confidence_score=paper.get('confidence_score', 0),
            used_full_text=bool(paper.get('full_text')),
            n_chunks_retrieved=total_chunks,
            answers=answers,
            timestamp=datetime.now().isoformat()
        )
    
    def _build_batch_prompt(self, abstract: str, all_contexts: Dict) -> str:
        """Build prompt for batch question answering."""
        prompt_parts = []
        
        # Add abstract
        if abstract:
            prompt_parts.append("="*70)
            prompt_parts.append("PAPER ABSTRACT")
            prompt_parts.append("="*70)
            prompt_parts.append(abstract)
            prompt_parts.append("="*70)
            prompt_parts.append("")
        
        # Add instruction
        prompt_parts.append("Answer ALL of the following questions based on the paper abstract and retrieved context chunks.")
        prompt_parts.append("For each question, provide your answer in JSON format with: answer, confidence (0-1), and reasoning.")
        prompt_parts.append("")
        
        # Add each question with its context
        for q_num, (q_key, q_obj) in enumerate(self.questions.items(), 1):
            q_text = q_obj['question']
            q_options = q_obj['answers']
            context = all_contexts.get(q_key, {}).get('context', '')
            
            prompt_parts.append(f"\n{'='*70}")
            prompt_parts.append(f"QUESTION {q_num}: {q_key}")
            prompt_parts.append(f"{'='*70}")
            prompt_parts.append(f"Question: {q_text}")
            prompt_parts.append(f"Options: {q_options}")
            prompt_parts.append(f"\nRetrieved Context:")
            prompt_parts.append(context if context else "[No context retrieved]")
        
        # Add output format instruction
        prompt_parts.append(f"\n{'='*70}")
        prompt_parts.append("OUTPUT FORMAT")
        prompt_parts.append(f"{'='*70}")
        prompt_parts.append("Provide answers as a JSON object with question keys:")
        prompt_parts.append('```json')
        prompt_parts.append('{')
        for q_key in self.questions.keys():
            prompt_parts.append(f'  "{q_key}": {{"answer": "...", "confidence": 0.0, "reasoning": "..."}},')
        prompt_parts.append('}')
        prompt_parts.append('```')
        
        return "\n".join(prompt_parts)
    
    def _parse_batch_response(self, raw_response: str) -> Dict:
        """Parse batch LLM response."""
        try:
            # Extract JSON from response
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without code blocks
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("No JSON found in response")
            
            parsed = json.loads(json_str)
            
            # Validate and format answers
            answers = {}
            for q_key in self.questions.keys():
                if q_key in parsed:
                    answers[q_key] = {
                        'answer': parsed[q_key].get('answer'),
                        'confidence': float(parsed[q_key].get('confidence', 0.0)),
                        'reasoning': parsed[q_key].get('reasoning', ''),
                        'parse_error': False
                    }
                else:
                    answers[q_key] = {
                        'answer': None,
                        'confidence': 0.0,
                        'reasoning': 'Missing in batch response',
                        'parse_error': True
                    }
            
            return answers
            
        except Exception as e:
            print(f"  ⚠ Parse error: {str(e)[:100]}")
            return {q_key: {
                'answer': None,
                'confidence': 0.0,
                'reasoning': f'Parse error: {str(e)}',
                'parse_error': True
            } for q_key in self.questions.keys()}
    
    def process_paper(
        self,
        paper: Dict,
        temperature: float = 0.2,
        max_tokens: int = 300
    ) -> PaperAnswers:
        """
        Process one paper through RAG system.
        
        Args:
            paper: Paper dict with doi, abstract, full_text, etc.
            temperature: LLM temperature
            max_tokens: Max tokens per answer
            
        Returns:
            PaperAnswers with all question answers
        """
        doi = paper['doi']
        abstract = paper.get('abstract', '')
        
        answers = {}
        total_chunks = 0
        
        for q_num, (q_key, q_obj) in enumerate(self.questions.items(), 1):
            q_text = q_obj['question']
            q_options_str = q_obj['answers']
            q_options = [opt.strip() for opt in q_options_str.split('/') if opt.strip()]
            
            # Determine n_results based on question type
            if q_num == 1 or q_num == 5:
                n_results = 12  # Biomarker questions
            elif q_num == 2:
                n_results = 10  # Mechanism questions
            elif q_num == 3:
                n_results = 8   # Intervention questions
            else:
                n_results = 15  # Species-specific questions
            
            try:
                # Answer question using RAG
                result = self.complete_rag.answer_question(
                    question=q_text,
                    n_results=n_results,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    answer_options=q_options,
                    doi=doi,
                    question_key=q_key,
                    include_abstract=True,  # Include abstract in context
                    abstract=abstract
                )
                
                answers[q_key] = {
                    'answer': result.get('answer'),
                    'confidence': result.get('confidence', 0.0),
                    'reasoning': result.get('reasoning', ''),
                    'parse_error': result.get('parse_error', False),
                    'n_sources': result.get('n_sources', 0)
                }
                
                total_chunks += result.get('n_sources', 0)
                
            except Exception as e:
                print(f"  ⚠ Error on question {q_key}: {str(e)[:100]}")
                answers[q_key] = {
                    'answer': None,
                    'confidence': 0.0,
                    'reasoning': f'Error: {str(e)}',
                    'parse_error': True,
                    'n_sources': 0
                }
        
        return PaperAnswers(
            doi=doi,
            pmid=paper.get('pmid'),
            title=paper.get('title', ''),
            abstract=abstract,
            validation_result=paper.get('validation_result', ''),
            confidence_score=paper.get('confidence_score', 0),
            used_full_text=bool(paper.get('full_text')),
            n_chunks_retrieved=total_chunks,
            answers=answers,
            timestamp=datetime.now().isoformat()
        )
    
    def init_results_db(self, results_db: str):
        """Initialize results database with tables."""
        conn = sqlite3.connect(results_db)
        cur = conn.cursor()
        
        # Paper metadata table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_metadata (
                doi TEXT PRIMARY KEY,
                pmid TEXT,
                title TEXT,
                abstract TEXT,
                validation_result TEXT,
                confidence_score INTEGER,
                used_full_text BOOLEAN,
                n_chunks_retrieved INTEGER,
                timestamp TEXT
            )
            """
        )
        
        # Answers table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doi TEXT,
                question_key TEXT,
                question_text TEXT,
                answer TEXT,
                confidence REAL,
                reasoning TEXT,
                parse_error BOOLEAN,
                n_sources INTEGER,
                UNIQUE(doi, question_key),
                FOREIGN KEY(doi) REFERENCES paper_metadata(doi)
            )
            """
        )
        
        # Processing log table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS processing_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doi TEXT,
                status TEXT,
                error_message TEXT,
                timestamp TEXT
            )
            """
        )
        
        conn.commit()
        conn.close()
        print(f"✓ Initialized database: {results_db}")
    
    def save_paper_result(self, results_db: str, result: PaperAnswers):
        """Save paper result to database."""
        conn = sqlite3.connect(results_db, timeout=30)
        cur = conn.cursor()
        
        # Insert/update paper metadata
        cur.execute(
            """
            INSERT INTO paper_metadata 
            (doi, pmid, title, abstract, validation_result, confidence_score, 
             used_full_text, n_chunks_retrieved, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(doi) DO UPDATE SET
                pmid=excluded.pmid,
                title=excluded.title,
                abstract=excluded.abstract,
                validation_result=excluded.validation_result,
                confidence_score=excluded.confidence_score,
                used_full_text=excluded.used_full_text,
                n_chunks_retrieved=excluded.n_chunks_retrieved,
                timestamp=excluded.timestamp
            """,
            (
                result.doi,
                result.pmid,
                result.title,
                result.abstract,
                result.validation_result,
                result.confidence_score,
                result.used_full_text,
                result.n_chunks_retrieved,
                result.timestamp
            )
        )
        
        # Delete old answers for this DOI
        cur.execute("DELETE FROM paper_answers WHERE doi = ?", (result.doi,))
        
        # Insert new answers
        for q_key, answer_data in result.answers.items():
            q_text = self.questions[q_key]['question']
            
            cur.execute(
                """
                INSERT INTO paper_answers 
                (doi, question_key, question_text, answer, confidence, reasoning, parse_error, n_sources)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.doi,
                    q_key,
                    q_text,
                    answer_data.get('answer'),
                    answer_data.get('confidence', 0.0),
                    answer_data.get('reasoning', ''),
                    answer_data.get('parse_error', False),
                    answer_data.get('n_sources', 0)
                )
            )
        
        # Log successful processing
        cur.execute(
            """
            INSERT INTO processing_log (doi, status, error_message, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (result.doi, 'success', None, datetime.now().isoformat())
        )
        
        conn.commit()
        conn.close()
    
    def process_all_papers(
        self,
        evaluations_db: str,
        papers_db: str,
        results_db: str,
        limit: Optional[int] = None,
        temperature: float = 0.2,
        max_tokens: int = 300
    ):
        """
        Process all validated papers through RAG system.
        
        Args:
            evaluations_db: Path to evaluations.db
            papers_db: Path to papers.db
            results_db: Path to results database (will be created)
            limit: Optional limit on number of papers
            temperature: LLM temperature
            max_tokens: Max tokens per answer
        """
        # Initialize results database
        self.init_results_db(results_db)
        
        # Get validated papers
        papers = self.get_validated_papers(evaluations_db, papers_db, limit)
        
        # Get already processed papers
        processed = self.get_already_processed(results_db)
        papers_to_process = [p for p in papers if p['doi'] not in processed]
        
        print(f"\n📊 Processing Status:")
        print(f"  Total validated papers: {len(papers)}")
        print(f"  Already processed: {len(processed)}")
        print(f"  To process: {len(papers_to_process)}")
        
        if not papers_to_process:
            print("\n✓ All papers already processed!")
            return
        
        print(f"\n{'='*70}")
        print("PROCESSING PAPERS")
        print(f"{'='*70}\n")
        
        # Process each paper
        for i, paper in enumerate(tqdm(papers_to_process, desc="Processing papers"), 1):
            doi = paper['doi']
            
            try:
                print(f"\n[{i}/{len(papers_to_process)}] {doi}")
                
                # Process paper through RAG (batch mode - all questions in one LLM call)
                result = self.process_paper_batch(paper, temperature, max_tokens)
                
                # Save result to database
                self.save_paper_result(results_db, result)
                
                # Show summary
                correct = sum(1 for a in result.answers.values() if a.get('answer') and not a.get('parse_error'))
                print(f"  ✓ Answered {correct}/{len(result.answers)} questions")
                
                # Rate limiting (simple delay)
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  ❌ Error processing {doi}: {str(e)}")
                
                # Log error
                conn = sqlite3.connect(results_db, timeout=30)
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO processing_log (doi, status, error_message, timestamp)
                    VALUES (?, ?, ?, ?)
                    """,
                    (doi, 'error', str(e), datetime.now().isoformat())
                )
                conn.commit()
                conn.close()
                
                continue
        
        print(f"\n{'='*70}")
        print("✓ PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Results saved to: {results_db}")


def main():
    parser = argparse.ArgumentParser(
        description="Run RAG system on all validated papers"
    )
    parser.add_argument(
        '--evaluations-db',
        type=str,
        default='/home/diana.z/hack/llm_judge/data/evaluations.db',
        help='Path to evaluations.db'
    )
    parser.add_argument(
        '--papers-db',
        type=str,
        default='/home/diana.z/hack/download_papers_pubmed/paper_collection/data/papers.db',
        help='Path to papers.db'
    )
    parser.add_argument(
        '--results-db',
        type=str,
        default='rag_results/rag_results.db',
        help='Path to results database (default: rag_results.db)'
    )
    parser.add_argument(
        '--questions-file',
        type=str,
        default='data/questions_part2.json',
        help='Path to questions JSON file'
    )
    parser.add_argument(
        '--predefined-queries',
        type=str,
        default='data/queries_extended.json',
        help='Path to predefined queries JSON file'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of papers to process (for testing)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.2,
        help='LLM temperature (default: 0.2)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=2000,
        help='Max tokens for batch response (default: 2000)'
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
        help='ChromaDB persistence directory'
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default=os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-mpnet-base-v2'),
        help='Embedding model name'
    )
    parser.add_argument(
        '--backup-model',
        type=str,
        default=os.getenv('BACKUP_EMBEDDING_MODEL', 'sentence-transformers/all-mpnet-base-v2'),
        help='Backup embedding model'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("RAG System - Batch Processing on Validated Papers")
    print("="*70)
    
    # Initialize RAG system
    print("\nInitializing RAG system...")
    rag = ScientificRAG(
        collection_name=args.collection_name,
        persist_directory=args.persist_dir,
        embedding_model=args.embedding_model,
        backup_embedding_model=args.backup_model
    )
    
    # Initialize LLM
    print("Initializing LLM...")
    llm = AzureOpenAIClient(model='gpt-4.1')
    
    # Initialize processor
    processor = RAGPaperProcessor(
        rag_system=rag,
        llm_client=llm,
        questions_file=args.questions_file,
        predefined_queries_file=args.predefined_queries
    )
    
    # Process all papers
    processor.process_all_papers(
        evaluations_db=args.evaluations_db,
        papers_db=args.papers_db,
        results_db=args.results_db,
        limit=args.limit,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )


if __name__ == "__main__":
    main()
