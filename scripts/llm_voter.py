"""
LLM Voter: Combine RAG and Full-Text evaluation results.

Strategy: RAG YES OVERRIDE
- Use full-text answers as baseline
- When RAG says "Yes" for a question, trust RAG's answer
- Rationale: Leverages full-text's broad coverage while trusting RAG's precision for positive findings
"""
import sqlite3
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import csv
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()


class LLMVoter:
    """Combines RAG and full-text evaluation results using RAG YES OVERRIDE strategy."""
    
    def __init__(
        self,
        fulltext_db_path: str,
        rag_db_path: str,
        questions_json_path: str,
        theory_mapping_path: str,
        validation_set_dir: str,
        papers_db_path: str,
        original_questions_path: str,
        output_db_path: str = None
    ):
        """
        Initialize the voter.
        
        Args:
            fulltext_db_path: Path to full-text evaluations DB (external)
            rag_db_path: Path to RAG evaluations DB (internal)
            questions_json_path: Path to questions_part2.json
            output_db_path: Path for output database (default: combined_results.db)
        """
        self.fulltext_db_path = Path(fulltext_db_path)
        self.rag_db_path = Path(rag_db_path)
        self.questions_json_path = Path(questions_json_path)
        self.theory_mapping_path = Path(theory_mapping_path)
        self.validation_set_dir = Path(validation_set_dir)
        self.papers_db_path = Path(papers_db_path)
        self.original_questions_path = Path(original_questions_path)
        self.output_db_path = Path(output_db_path) if output_db_path else Path("combined_results.db")
        
        # Validate paths
        if not self.fulltext_db_path.exists():
            raise FileNotFoundError(f"Full-text DB not found: {fulltext_db_path}")
        if not self.rag_db_path.exists():
            raise FileNotFoundError(f"RAG DB not found: {rag_db_path}")
        if not self.questions_json_path.exists():
            raise FileNotFoundError(f"Questions JSON not found: {questions_json_path}")
        if not self.theory_mapping_path.exists():
            raise FileNotFoundError(f"Theory mapping not found: {theory_mapping_path}")
        if not self.validation_set_dir.exists():
            raise FileNotFoundError(f"Validation set directory not found: {validation_set_dir}")
        if not self.papers_db_path.exists():
            raise FileNotFoundError(f"Papers database not found: {papers_db_path}")
        if not self.original_questions_path.exists():
            raise FileNotFoundError(f"Original questions mapping not found: {original_questions_path}")
        
        # Load questions mapping
        with open(self.questions_json_path, 'r') as f:
            self.questions_config = json.load(f)
        
        # Map question keys to full question text and valid answers
        self.question_mapping = {
            key: {
                'question': config['question'],
                'valid_answers': [a.strip() for a in config['answers'].split('/')]
            }
            for key, config in self.questions_config.items()
        }
        
        # Load original questions mapping
        with open(self.original_questions_path, 'r') as f:
            self.original_questions = json.load(f)
        
        # Load theory mapping
        with open(self.theory_mapping_path, 'r') as f:
            theory_data = json.load(f)
        
        # Create DOI to theory mapping and theory ID mapping
        self.doi_to_theory = {}
        self.theory_to_dois = {}
        self.theory_to_id = {}
        self.id_to_theory = {}
        
        for idx, theory in enumerate(theory_data.get('theories', []), start=1):
            theory_name = theory['name']
            theory_id = f"T{idx:04d}"
            dois = theory.get('dois', [])
            
            self.theory_to_dois[theory_name] = dois
            self.theory_to_id[theory_name] = theory_id
            self.id_to_theory[theory_id] = theory_name
            
            for doi in dois:
                self.doi_to_theory[doi] = theory_name
        
        # Get list of valid DOIs from theory mapping
        self.valid_dois = set(self.doi_to_theory.keys())
        
        print(f"✓ Loaded {len(self.question_mapping)} question mappings")
        print(f"✓ Loaded {len(self.original_questions)} original question mappings")
        print(f"✓ Loaded {len(self.theory_to_dois)} theories with {len(self.valid_dois)} DOIs")
        print(f"✓ Full-text DB: {self.fulltext_db_path}")
        print(f"✓ RAG DB: {self.rag_db_path}")
        print(f"✓ Validation set: {self.validation_set_dir}")
        print(f"✓ Papers DB: {self.papers_db_path}")
        print(f"✓ Output DB: {self.output_db_path}")
    
    def load_paper_metadata(self, doi: str) -> Dict[str, str]:
        """
        Load paper metadata from papers database.
        
        Args:
            doi: DOI to look up
            
        Returns:
            Dict with title, year, journal, date_published, cited_by_count, fwci, topic_name, keywords
        """
        conn = sqlite3.connect(self.papers_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT title, year, journal, date_published, cited_by_count, fwci, topic_name, keywords
            FROM papers
            WHERE doi = ?
        """, (doi,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'title': result[0] or '',
                'year': result[1] or '',
                'journal': result[2] or '',
                'date_published': result[3] or '',
                'cited_by_count': result[4] if result[4] is not None else '',
                'fwci': result[5] if result[5] is not None else '',
                'topic_name': result[6] or '',
                'keywords': result[7] or ''
            }
        else:
            # Return empty values if not found
            return {
                'title': '',
                'year': '',
                'journal': '',
                'date_published': '',
                'cited_by_count': '',
                'fwci': '',
                'topic_name': '',
                'keywords': ''
            }
    
    def _is_yes_answer(self, answer: str) -> bool:
        """
        Check if an answer is a "Yes" variant.
        
        Args:
            answer: The answer string
            
        Returns:
            True if answer contains "Yes" (case-insensitive)
        """
        if not answer:
            return False
        return 'yes' in answer.lower()
    
    def _validate_answer(self, question_key: str, answer: str) -> str:
        """
        Validate that answer matches allowed options for the question.
        
        Args:
            question_key: Question identifier
            answer: Answer to validate
            
        Returns:
            Validated answer or original if validation fails
        """
        if question_key not in self.question_mapping:
            return answer
        
        valid_answers = self.question_mapping[question_key]['valid_answers']
        
        # Check if answer matches any valid option
        for valid in valid_answers:
            if answer.strip().lower() == valid.strip().lower():
                return valid  # Return the canonical form
        
        # If no exact match, return original (will be flagged in output)
        return answer
    
    def load_validation_set_answer(self, doi: str) -> Optional[Dict[str, Dict]]:
        """
        Load answers from validation set for a specific DOI.
        
        Args:
            doi: DOI to load (will be converted to filename format)
            
        Returns:
            Dict[question_key] = {answer, confidence, reasoning, source} or None if not found
        """
        # Convert DOI to filename format (replace / with _)
        filename = doi.replace('/', '_') + '.json'
        filepath = self.validation_set_dir / filename
        
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Convert to standard format
            validation_answers = {}
            for question_key, answer_data in data.items():
                validation_answers[question_key] = {
                    'answer': answer_data.get('answer', 'N/A'),
                    'confidence': answer_data.get('confidence', 0.0),
                    'reasoning': answer_data.get('reasoning', ''),
                    'source': 'validation_set'
                }
            
            return validation_answers
        except Exception as e:
            print(f"⚠️  Error loading validation set for {doi}: {e}")
            return None
    
    def load_fulltext_answers(self) -> Dict[str, Dict[str, Dict]]:
        """
        Load all full-text answers.
        
        Returns:
            Dict[doi][question_key] = {answer, confidence, reasoning}
        """
        conn = sqlite3.connect(self.fulltext_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT doi, question_name, answer, confidence_score, reasoning
            FROM paper_answers
        """)
        
        fulltext_data = {}
        for doi, question_name, answer, confidence, reasoning in cursor.fetchall():
            if doi not in fulltext_data:
                fulltext_data[doi] = {}
            
            fulltext_data[doi][question_name] = {
                'answer': answer,
                'confidence': confidence,
                'reasoning': reasoning,
                'source': 'fulltext'
            }
        
        conn.close()
        print(f"✓ Loaded full-text answers for {len(fulltext_data)} DOIs")
        return fulltext_data
    
    def load_rag_answers(self) -> Dict[str, Dict[str, Dict]]:
        """
        Load all RAG answers.
        
        Returns:
            Dict[doi][question_key] = {answer, confidence, reasoning}
        """
        conn = sqlite3.connect(self.rag_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT doi, question_key, answer, confidence, reasoning
            FROM paper_answers
        """)
        
        rag_data = {}
        for doi, question_key, answer, confidence, reasoning in cursor.fetchall():
            if doi not in rag_data:
                rag_data[doi] = {}
            
            rag_data[doi][question_key] = {
                'answer': answer,
                'confidence': confidence,
                'reasoning': reasoning,
                'source': 'rag'
            }
        
        conn.close()
        print(f"✓ Loaded RAG answers for {len(rag_data)} DOIs")
        return rag_data
    
    def combine_answers(
        self,
        fulltext_data: Dict[str, Dict[str, Dict]],
        rag_data: Dict[str, Dict[str, Dict]]
    ) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, Dict]]]:
        """
        Combine answers using RAG YES OVERRIDE strategy, filtered by theory mapping.
        
        Strategy:
        1. Only process DOIs from theory mapping
        2. Start with full-text answers as baseline
        3. If RAG says "Yes" for a question, override with RAG answer
        4. Use validation set as fallback if DOI missing from both sources
        
        Args:
            fulltext_data: Full-text answers
            rag_data: RAG answers
            
        Returns:
            Tuple of (short_results, extended_results)
            - short_results: Dict[doi][question_key] = answer
            - extended_results: Dict[doi][question_key] = {answer, confidence, reasoning, source}
        """
        short_results = {}
        extended_results = {}
        
        stats = {
            'total_theory_dois': len(self.valid_dois),
            'processed_dois': 0,
            'fulltext_only': 0,
            'rag_only': 0,
            'both_sources': 0,
            'validation_set_used': 0,
            'missing_dois': 0,
            'rag_overrides': 0,
            'validation_warnings': [],
            'missing_dois_list': []
        }
        
        # Process only DOIs from theory mapping
        for doi in self.valid_dois:
            short_results[doi] = {}
            extended_results[doi] = {}
            
            fulltext_answers = fulltext_data.get(doi, {})
            rag_answers = rag_data.get(doi, {})
            
            # If DOI missing from both sources, try validation set
            if not fulltext_answers and not rag_answers:
                validation_answers = self.load_validation_set_answer(doi)
                if validation_answers:
                    fulltext_answers = validation_answers  # Use as baseline
                    stats['validation_set_used'] += 1
                else:
                    # DOI completely missing
                    stats['missing_dois'] += 1
                    stats['missing_dois_list'].append(doi)
                    print(f"⚠️  WARNING: DOI {doi} missing from all sources (fulltext, RAG, validation set)")
                    continue
            
            # Track source availability
            if fulltext_answers and not rag_answers:
                stats['fulltext_only'] += 1
            elif rag_answers and not fulltext_answers:
                stats['rag_only'] += 1
            elif fulltext_answers and rag_answers:
                stats['both_sources'] += 1
            
            stats['processed_dois'] += 1
            
            # Get all question keys from both sources
            all_questions = set(fulltext_answers.keys()) | set(rag_answers.keys())
            
            for question_key in all_questions:
                fulltext_answer = fulltext_answers.get(question_key, {})
                rag_answer = rag_answers.get(question_key, {})
                
                # RAG YES OVERRIDE logic
                if rag_answer and self._is_yes_answer(rag_answer.get('answer', '')):
                    # RAG says Yes -> use RAG answer
                    final_answer = rag_answer
                    stats['rag_overrides'] += 1
                elif fulltext_answer:
                    # Use full-text as baseline
                    final_answer = fulltext_answer
                elif rag_answer:
                    # Only RAG available (no full-text)
                    final_answer = rag_answer
                else:
                    # No answer available (shouldn't happen)
                    continue
                
                # Validate answer
                validated_answer = self._validate_answer(question_key, final_answer['answer'])
                if validated_answer != final_answer['answer']:
                    stats['validation_warnings'].append({
                        'doi': doi,
                        'question': question_key,
                        'original': final_answer['answer'],
                        'validated': validated_answer
                    })
                
                # Store short result
                short_results[doi][question_key] = validated_answer
                
                # Store extended result
                extended_results[doi][question_key] = {
                    'answer': validated_answer,
                    'confidence': final_answer.get('confidence', 0.0),
                    'reasoning': final_answer.get('reasoning', ''),
                    'source': final_answer.get('source', 'unknown')
                }
        
        print(f"\n📊 Combination Statistics:")
        print(f"  Total theory DOIs: {stats['total_theory_dois']}")
        print(f"  Processed DOIs: {stats['processed_dois']}")
        print(f"  Full-text only: {stats['fulltext_only']}")
        print(f"  RAG only: {stats['rag_only']}")
        print(f"  Both sources: {stats['both_sources']}")
        print(f"  Validation set used: {stats['validation_set_used']}")
        print(f"  Missing DOIs: {stats['missing_dois']}")
        print(f"  RAG overrides applied: {stats['rag_overrides']}")
        
        if stats['validation_warnings']:
            print(f"\n⚠️  Validation warnings: {len(stats['validation_warnings'])}")
            for warning in stats['validation_warnings'][:5]:
                print(f"    {warning['doi'][:20]}... | {warning['question']}: '{warning['original']}' -> '{warning['validated']}'")
            if len(stats['validation_warnings']) > 5:
                print(f"    ... and {len(stats['validation_warnings']) - 5} more")
        
        return short_results, extended_results
    
    def create_output_database(
        self,
        short_results: Dict[str, Dict[str, str]],
        extended_results: Dict[str, Dict[str, Dict]]
    ):
        """
        Create output database with two tables: short and extended results.
        
        Args:
            short_results: Dict[doi][question_key] = answer
            extended_results: Dict[doi][question_key] = {answer, confidence, reasoning, source}
        """
        # Remove existing database
        if self.output_db_path.exists():
            self.output_db_path.unlink()
            print(f"✓ Removed existing database: {self.output_db_path}")
        
        conn = sqlite3.connect(self.output_db_path)
        cursor = conn.cursor()
        
        # Create short results table (one row per DOI)
        question_keys = list(self.question_mapping.keys())
        columns_def = ', '.join([f"{key} TEXT" for key in question_keys])
        
        cursor.execute(f"""
            CREATE TABLE combined_answers_short (
                doi TEXT PRIMARY KEY,
                {columns_def},
                timestamp TEXT
            )
        """)
        
        # Create extended results table (one row per DOI-question pair)
        cursor.execute("""
            CREATE TABLE combined_answers_extended (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doi TEXT,
                question_key TEXT,
                question_text TEXT,
                answer TEXT,
                confidence REAL,
                reasoning TEXT,
                source TEXT,
                timestamp TEXT,
                UNIQUE(doi, question_key)
            )
        """)
        
        # Insert short results
        timestamp = datetime.now().isoformat()
        for doi, answers in short_results.items():
            values = [doi]
            for key in question_keys:
                values.append(answers.get(key, 'N/A'))
            values.append(timestamp)
            
            placeholders = ', '.join(['?'] * len(values))
            cursor.execute(f"""
                INSERT INTO combined_answers_short VALUES ({placeholders})
            """, values)
        
        # Insert extended results
        for doi, answers in extended_results.items():
            for question_key, data in answers.items():
                question_text = self.question_mapping[question_key]['question']
                cursor.execute("""
                    INSERT INTO combined_answers_extended 
                    (doi, question_key, question_text, answer, confidence, reasoning, source, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    doi,
                    question_key,
                    question_text,
                    data['answer'],
                    data['confidence'],
                    data['reasoning'],
                    data['source'],
                    timestamp
                ))
        
        conn.commit()
        conn.close()
        
        print(f"\n✓ Created output database: {self.output_db_path}")
        print(f"  - combined_answers_short: {len(short_results)} rows")
        print(f"  - combined_answers_extended: {sum(len(a) for a in extended_results.values())} rows")
    
    def export_to_csv(
        self,
        short_results: Dict[str, Dict[str, str]],
        extended_results: Dict[str, Dict[str, Dict]]
    ):
        """
        Export results to CSV files.
        
        Args:
            short_results: Dict[doi][question_key] = answer
            extended_results: Dict[doi][question_key] = {answer, confidence, reasoning, source}
        """
        # Define output paths
        short_csv_path = self.output_db_path.parent / f"{self.output_db_path.stem}_short.csv"
        extended_csv_path = self.output_db_path.parent / f"{self.output_db_path.stem}_extended.csv"
        
        # Export short results (one row per DOI)
        question_keys = list(self.question_mapping.keys())
        
        print("  Loading paper metadata for short CSV...")
        with open(short_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header - add theory_id, theory, paper metadata, and original question names
            header = ['doi', 'theory_id', 'theory', 'title', 'year']
            # Add original question names as headers
            for key in question_keys:
                header.append(self.original_questions.get(key, key))
            writer.writerow(header)
            
            # Data rows
            for doi in sorted(short_results.keys()):
                theory = self.doi_to_theory.get(doi, 'Unknown')
                theory_id = self.theory_to_id.get(theory, '')
                
                # Load paper metadata
                paper_meta = self.load_paper_metadata(doi)
                
                row = [
                    doi,
                    theory_id,
                    theory,
                    paper_meta['title'],
                    paper_meta['year']
                ]
                
                # Add answers for each question
                for key in question_keys:
                    row.append(short_results[doi].get(key, 'N/A'))
                
                writer.writerow(row)
        
        print(f"\n✓ Created short CSV: {short_csv_path}")
        print(f"  - {len(short_results)} rows")
        
        # Export extended results (one row per DOI-question pair)
        print("  Loading paper metadata for extended CSV...")
        
        # Pre-load all paper metadata to avoid repeated queries
        paper_metadata_cache = {}
        for doi in extended_results.keys():
            paper_metadata_cache[doi] = self.load_paper_metadata(doi)
        
        with open(extended_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header - add theory_id, theory, paper metadata, and use original question name
            header = [
                'doi', 'theory_id', 'theory', 'title', 'year',
                'question_key', 'question_text', 'answer', 'confidence', 'reasoning', 'source',
                'journal', 'date_published', 'cited_by_count', 'fwci', 'openalex_topic_name', 'keywords'
            ]
            writer.writerow(header)
            
            # Data rows - sort by DOI then question_key for consistency
            for doi in sorted(extended_results.keys()):
                theory = self.doi_to_theory.get(doi, 'Unknown')
                theory_id = self.theory_to_id.get(theory, '')
                paper_meta = paper_metadata_cache[doi]
                
                for question_key in sorted(extended_results[doi].keys()):
                    data = extended_results[doi][question_key]
                    # Use original question text
                    original_question = self.original_questions.get(question_key, question_key)
                    
                    row = [
                        doi,
                        theory_id,
                        theory,
                        paper_meta['title'],
                        paper_meta['year'],
                        question_key,
                        original_question,
                        data['answer'],
                        data['confidence'],
                        data['reasoning'],
                        data['source'],
                        paper_meta['journal'],
                        paper_meta['date_published'],
                        paper_meta['cited_by_count'],
                        paper_meta['fwci'],
                        paper_meta['topic_name'],
                        paper_meta['keywords']
                    ]
                    writer.writerow(row)
        
        print(f"✓ Created extended CSV: {extended_csv_path}")
        print(f"  - {sum(len(a) for a in extended_results.values())} rows")
        
        # Export theory statistics
        theory_csv_path = self.output_db_path.parent / f"{self.output_db_path.stem}_theory_stats.csv"
        with open(theory_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['theory_name', 'doi_count']
            writer.writerow(header)
            
            # Data rows - sort by DOI count descending
            theory_stats = [(theory, len(dois)) for theory, dois in self.theory_to_dois.items()]
            theory_stats.sort(key=lambda x: x[1], reverse=True)
            
            for theory_name, doi_count in theory_stats:
                writer.writerow([theory_name, doi_count])
        
        print(f"✓ Created theory statistics CSV: {theory_csv_path}")
        print(f"  - {len(theory_stats)} theories")
    
    def create_theory_stats_table(self):
        """Create theory statistics table in database."""
        conn = sqlite3.connect(self.output_db_path)
        cursor = conn.cursor()
        
        # Create theory statistics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS theory_statistics (
                theory_name TEXT PRIMARY KEY,
                doi_count INTEGER
            )
        """)
        
        # Insert data
        for theory_name, dois in self.theory_to_dois.items():
            cursor.execute("""
                INSERT INTO theory_statistics (theory_name, doi_count)
                VALUES (?, ?)
            """, (theory_name, len(dois)))
        
        conn.commit()
        conn.close()
        
        print(f"✓ Created theory_statistics table with {len(self.theory_to_dois)} theories")
    
    def run(self):
        """Execute the full voting process."""
        print("\n" + "="*70)
        print("LLM VOTER - Combining RAG and Full-Text Results")
        print("="*70)
        print(f"Strategy: RAG YES OVERRIDE")
        print(f"  → Full-text answers as baseline")
        print(f"  → RAG 'Yes' answers override full-text")
        print("="*70 + "\n")
        
        # Load data
        print("📥 Loading data...")
        fulltext_data = self.load_fulltext_answers()
        rag_data = self.load_rag_answers()
        
        # Combine answers
        print("\n🔄 Combining answers...")
        short_results, extended_results = self.combine_answers(fulltext_data, rag_data)
        
        # Create output database
        print("\n💾 Creating output database...")
        self.create_output_database(short_results, extended_results)
        self.create_theory_stats_table()
        
        # Export to CSV
        print("\n📄 Exporting to CSV...")
        self.export_to_csv(short_results, extended_results)
        
        print("\n" + "="*70)
        print("✅ COMPLETE!")
        print("="*70)
        print(f"\nOutput files:")
        print(f"  Database: {self.output_db_path}")
        print(f"  Short CSV: {self.output_db_path.stem}_short.csv")
        print(f"  Extended CSV: {self.output_db_path.stem}_extended.csv")
        print(f"  Theory Stats CSV: {self.output_db_path.stem}_theory_stats.csv")
        print("\nDatabase tables:")
        print("  1. combined_answers_short - One row per DOI with all answers")
        print("  2. combined_answers_extended - Detailed view with confidence and reasoning")
        print("  3. theory_statistics - Theory names and DOI counts")
        print("\nCSV files:")
        print("  1. *_short.csv - One row per DOI with theory and answers")
        print("  2. *_extended.csv - One row per DOI-question with theory, confidence, reasoning, source")
        print("  3. *_theory_stats.csv - Theory names and DOI counts")
        print("\nExample queries:")
        print(f"  sqlite3 {self.output_db_path} 'SELECT * FROM combined_answers_short LIMIT 5'")
        print(f"  sqlite3 {self.output_db_path} 'SELECT * FROM theory_statistics ORDER BY doi_count DESC LIMIT 10'")
        print("="*70 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Combine RAG and full-text evaluation results using RAG YES OVERRIDE strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategy: RAG YES OVERRIDE
  → Use full-text answers as baseline
  → When RAG says "Yes" for a question, trust RAG's answer
  → Rationale: Leverages full-text's broad coverage while trusting RAG's precision

Example:
  python llm_voter.py --output combined_results.db
  
  # Or with custom paths:
  python llm_voter.py \\
    --fulltext-db /path/to/qa_results.db \\
    --rag-db /path/to/rag_results_fast.db \\
    --questions /path/to/questions_part2.json \\
    --output combined_results.db
        """
    )
    
    parser.add_argument(
        '--fulltext-db',
        type=str,
        default=os.getenv('FULLTEXT_EVAL_DB', '/home/diana.z/hack/theories_extraction_agent/qa_results/qa_results.db'),
        help='Path to full-text evaluations database (default: from .env FULLTEXT_EVAL_DB)'
    )
    parser.add_argument(
        '--rag-db',
        type=str,
        default=os.getenv('RAG_EVAL_DB', 'rag_results_fast.db'),
        help='Path to RAG evaluations database (default: from .env RAG_EVAL_DB)'
    )
    parser.add_argument(
        '--questions',
        type=str,
        default='data/questions_part2.json',
        help='Path to questions JSON file (default: data/questions_part2.json)'
    )
    parser.add_argument(
        '--theory-mapping',
        type=str,
        default=os.getenv('THEORY_MAPPING', '/home/diana.z/hack/theories_extraction_agent/output/final_output/final_theory_to_dois_mapping.json'),
        help='Path to theory-to-DOIs mapping JSON (default: from .env THEORY_MAPPING)'
    )
    parser.add_argument(
        '--validation-set',
        type=str,
        default='data/validation_set_qa',
        help='Path to validation set directory (default: data/validation_set_qa)'
    )
    parser.add_argument(
        '--papers-db',
        type=str,
        default=os.getenv('PAPERS_DB', '/home/diana.z/hack/download_papers_pubmed/paper_collection/data/papers.db'),
        help='Path to papers database (default: from .env PAPERS_DB)'
    )
    parser.add_argument(
        '--original-questions',
        type=str,
        default='data/original_questions_mapping.json',
        help='Path to original questions mapping JSON (default: data/original_questions_mapping.json)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='combined_results.db',
        help='Output database path (default: combined_results.db)'
    )
    
    args = parser.parse_args()
    
    # Create and run voter
    voter = LLMVoter(
        fulltext_db_path=args.fulltext_db,
        rag_db_path=args.rag_db,
        questions_json_path=args.questions,
        theory_mapping_path=args.theory_mapping,
        validation_set_dir=args.validation_set,
        papers_db_path=args.papers_db,
        original_questions_path=args.original_questions,
        output_db_path=args.output
    )
    
    voter.run()


if __name__ == "__main__":
    main()
