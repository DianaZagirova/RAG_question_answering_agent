#!/usr/bin/env python3
"""
Export sample results from rag_results_fast.db to JSON format.
Creates a JSON file with 50 example DOIs and their complete RAG results.
"""
import sqlite3
import json
from pathlib import Path

def export_sample_results(db_path: str, output_file: str, sample_size: int = 50):
    """Export sample results to JSON."""
    print(f"Exporting {sample_size} sample results from {db_path}...")
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # Get sample DOIs
    cur.execute(f"""
        SELECT doi FROM paper_metadata 
        ORDER BY RANDOM() 
        LIMIT {sample_size}
    """)
    sample_dois = [row['doi'] for row in cur.fetchall()]
    
    print(f"Selected {len(sample_dois)} random papers")
    
    # Export complete data for these DOIs
    results = []
    
    for doi in sample_dois:
        # Get metadata
        cur.execute("""
            SELECT * FROM paper_metadata WHERE doi = ?
        """, (doi,))
        metadata = dict(cur.fetchone())
        
        # Get answers
        cur.execute("""
            SELECT question_key, question_text, answer, confidence, reasoning, n_sources
            FROM paper_answers 
            WHERE doi = ?
        """, (doi,))
        
        answers = {}
        for row in cur.fetchall():
            answers[row['question_key']] = {
                'question_text': row['question_text'],
                'answer': row['answer'],
                'confidence': row['confidence'],
                'reasoning': row['reasoning'],
                'n_sources': row['n_sources']
            }
        
        results.append({
            'doi': metadata['doi'],
            'pmid': metadata['pmid'],
            'title': metadata['title'],
            'abstract': metadata['abstract'],
            'validation_result': metadata['validation_result'],
            'confidence_score': metadata['confidence_score'],
            'used_full_text': bool(metadata['used_full_text']),
            'n_chunks_retrieved': metadata['n_chunks_retrieved'],
            'timestamp': metadata['timestamp'],
            'answers': answers
        })
    
    conn.close()
    
    # Write to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Exported {len(results)} papers to {output_file}")
    
    # Print statistics
    file_size = Path(output_file).stat().st_size / 1024
    print(f"✓ File size: {file_size:.1f} KB")
    print(f"✓ Average answers per paper: {sum(len(r['answers']) for r in results) / len(results):.1f}")

if __name__ == "__main__":
    export_sample_results(
        db_path='rag_results_fast.db',
        output_file='data/sample_rag_results.json',
        sample_size=50
    )
