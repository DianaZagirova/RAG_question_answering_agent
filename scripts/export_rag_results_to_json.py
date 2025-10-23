#!/usr/bin/env python3
"""
Export RAG results from SQLite database to JSON format.
"""

import sqlite3
import json
import argparse
from pathlib import Path
from typing import Dict, List


def export_to_json(db_path: str, output_json: str):
    """Export RAG results from database to JSON."""
    
    print(f"\n{'='*70}")
    print("EXPORTING RAG RESULTS TO JSON")
    print(f"{'='*70}\n")
    
    print(f"Reading from: {db_path}")
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Access columns by name
    cur = conn.cursor()
    
    # Get all papers with metadata
    cur.execute("""
        SELECT doi, pmid, title, abstract, validation_result, 
               confidence_score, used_full_text, n_chunks_retrieved, timestamp
        FROM paper_metadata
        ORDER BY timestamp
    """)
    
    papers = cur.fetchall()
    print(f"✓ Found {len(papers)} papers with results")
    
    # Build results structure
    results = []
    
    for paper in papers:
        doi = paper['doi']
        
        # Get all answers for this paper
        cur.execute("""
            SELECT question_key, question_text, answer, confidence, 
                   reasoning, parse_error, n_sources
            FROM paper_answers
            WHERE doi = ?
            ORDER BY question_key
        """, (doi,))
        
        answers_rows = cur.fetchall()
        
        # Format answers
        answers = {}
        for ans in answers_rows:
            answers[ans['question_key']] = {
                'question': ans['question_text'],
                'answer': ans['answer'],
                'confidence': ans['confidence'],
                'reasoning': ans['reasoning'],
                'parse_error': bool(ans['parse_error']),
                'n_sources': ans['n_sources']
            }
        
        # Build paper result
        paper_result = {
            'doi': paper['doi'],
            'pmid': paper['pmid'],
            'title': paper['title'],
            'abstract': paper['abstract'],
            'validation_result': paper['validation_result'],
            'confidence_score': paper['confidence_score'],
            'used_full_text': bool(paper['used_full_text']),
            'n_chunks_retrieved': paper['n_chunks_retrieved'],
            'timestamp': paper['timestamp'],
            'answers': answers
        }
        
        results.append(paper_result)
    
    conn.close()
    
    # Write to JSON file
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Exported {len(results)} papers to: {output_json}")
    
    # Print summary statistics
    total_answers = sum(len(p['answers']) for p in results)
    valid_answers = sum(
        sum(1 for a in p['answers'].values() if a['answer'] and not a['parse_error'])
        for p in results
    )
    
    print(f"\n{'='*70}")
    print("EXPORT SUMMARY")
    print(f"{'='*70}")
    print(f"Total papers: {len(results)}")
    print(f"Total answers: {total_answers}")
    print(f"Valid answers: {valid_answers}")
    print(f"Papers with full text: {sum(1 for p in results if p['used_full_text'])}")
    print(f"Average chunks per paper: {sum(p['n_chunks_retrieved'] for p in results) / len(results):.1f}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Export RAG results to JSON")
    parser.add_argument(
        '--db', 
        type=str, 
        default='rag_results_fast.db',
        help='Path to SQLite database'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='rag_results/rag_results.json',
        help='Output JSON file path'
    )
    
    args = parser.parse_args()
    
    # Check if database exists
    if not Path(args.db).exists():
        print(f"❌ Database not found: {args.db}")
        return
    
    export_to_json(args.db, args.output)


if __name__ == '__main__':
    main()
