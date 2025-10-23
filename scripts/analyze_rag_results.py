#!/usr/bin/env python3
"""
Analyze RAG results from database.

Shows statistics, accuracy, and exports results to JSON/CSV.
"""
import sqlite3
import json
import argparse
from pathlib import Path
from collections import Counter
import pandas as pd


def analyze_results(results_db: str):
    """Analyze RAG results from database."""
    conn = sqlite3.connect(results_db)
    
    print("\n" + "="*70)
    print("RAG RESULTS ANALYSIS")
    print("="*70)
    
    # Overall statistics
    cur = conn.cursor()
    
    # Total papers
    cur.execute("SELECT COUNT(*) FROM paper_metadata")
    total_papers = cur.fetchone()[0]
    
    # Papers with full text
    cur.execute("SELECT COUNT(*) FROM paper_metadata WHERE used_full_text = 1")
    with_full_text = cur.fetchone()[0]
    
    # Total answers
    cur.execute("SELECT COUNT(*) FROM paper_answers")
    total_answers = cur.fetchone()[0]
    
    # Successful answers (not null, no parse error)
    cur.execute("SELECT COUNT(*) FROM paper_answers WHERE answer IS NOT NULL AND parse_error = 0")
    successful_answers = cur.fetchone()[0]
    
    # Parse errors
    cur.execute("SELECT COUNT(*) FROM paper_answers WHERE parse_error = 1")
    parse_errors = cur.fetchone()[0]
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Total papers processed: {total_papers}")
    print(f"  Papers with full text: {with_full_text} ({with_full_text/total_papers*100:.1f}%)")
    print(f"  Total answers: {total_answers}")
    print(f"  Successful answers: {successful_answers} ({successful_answers/total_answers*100:.1f}%)")
    print(f"  Parse errors: {parse_errors} ({parse_errors/total_answers*100:.1f}%)")
    
    # Answer distribution by question
    print(f"\n📋 Answer Distribution by Question:")
    cur.execute("""
        SELECT question_key, answer, COUNT(*) as count
        FROM paper_answers
        WHERE answer IS NOT NULL AND parse_error = 0
        GROUP BY question_key, answer
        ORDER BY question_key, count DESC
    """)
    
    current_question = None
    for q_key, answer, count in cur.fetchall():
        if q_key != current_question:
            print(f"\n  {q_key}:")
            current_question = q_key
        print(f"    {answer}: {count}")
    
    # Average confidence by question
    print(f"\n🎯 Average Confidence by Question:")
    cur.execute("""
        SELECT question_key, AVG(confidence) as avg_conf, COUNT(*) as count
        FROM paper_answers
        WHERE answer IS NOT NULL AND parse_error = 0
        GROUP BY question_key
        ORDER BY avg_conf DESC
    """)
    
    for q_key, avg_conf, count in cur.fetchall():
        print(f"  {q_key}: {avg_conf:.2f} (n={count})")
    
    # Processing errors
    cur.execute("SELECT COUNT(*) FROM processing_log WHERE status = 'error'")
    processing_errors = cur.fetchone()[0]
    
    if processing_errors > 0:
        print(f"\n⚠️  Processing Errors: {processing_errors}")
        cur.execute("""
            SELECT doi, error_message, timestamp
            FROM processing_log
            WHERE status = 'error'
            ORDER BY timestamp DESC
            LIMIT 10
        """)
        print("  Recent errors:")
        for doi, error, timestamp in cur.fetchall():
            print(f"    {doi}: {error[:80]}...")
    
    conn.close()


def export_to_json(results_db: str, output_file: str):
    """Export results to JSON file."""
    conn = sqlite3.connect(results_db)
    
    # Get all papers with answers
    query = """
        SELECT 
            pm.doi,
            pm.pmid,
            pm.title,
            pm.abstract,
            pm.validation_result,
            pm.confidence_score,
            pm.used_full_text,
            pm.n_chunks_retrieved,
            pm.timestamp
        FROM paper_metadata pm
        ORDER BY pm.doi
    """
    
    cur = conn.cursor()
    cur.execute(query)
    papers = cur.fetchall()
    
    results = []
    for paper in papers:
        doi = paper[0]
        
        # Get answers for this paper
        cur.execute("""
            SELECT question_key, answer, confidence, reasoning, parse_error, n_sources
            FROM paper_answers
            WHERE doi = ?
        """, (doi,))
        
        answers = {}
        for q_key, answer, conf, reasoning, parse_error, n_sources in cur.fetchall():
            answers[q_key] = {
                'answer': answer,
                'confidence': conf,
                'reasoning': reasoning,
                'parse_error': bool(parse_error),
                'n_sources': n_sources
            }
        
        results.append({
            'doi': paper[0],
            'pmid': paper[1],
            'title': paper[2],
            'abstract': paper[3],
            'validation_result': paper[4],
            'confidence_score': paper[5],
            'used_full_text': bool(paper[6]),
            'n_chunks_retrieved': paper[7],
            'timestamp': paper[8],
            'answers': answers
        })
    
    conn.close()
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Exported {len(results)} papers to {output_file}")


def export_to_csv(results_db: str, output_file: str):
    """Export results to CSV file (flattened format)."""
    conn = sqlite3.connect(results_db)
    
    query = """
        SELECT 
            pm.doi,
            pm.pmid,
            pm.title,
            pm.validation_result,
            pm.confidence_score,
            pm.used_full_text,
            pm.n_chunks_retrieved,
            pa.question_key,
            pa.answer,
            pa.confidence,
            pa.parse_error,
            pa.n_sources
        FROM paper_metadata pm
        LEFT JOIN paper_answers pa ON pm.doi = pa.doi
        ORDER BY pm.doi, pa.question_key
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df.to_csv(output_file, index=False)
    print(f"✓ Exported {len(df)} rows to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze RAG results from database"
    )
    parser.add_argument(
        '--results-db',
        type=str,
        required=True,
        help='Path to results database'
    )
    parser.add_argument(
        '--export-json',
        type=str,
        help='Export results to JSON file'
    )
    parser.add_argument(
        '--export-csv',
        type=str,
        help='Export results to CSV file'
    )
    
    args = parser.parse_args()
    
    if not Path(args.results_db).exists():
        print(f"❌ Database not found: {args.results_db}")
        return
    
    # Analyze results
    analyze_results(args.results_db)
    
    # Export if requested
    if args.export_json:
        print(f"\n📤 Exporting to JSON...")
        export_to_json(args.results_db, args.export_json)
    
    if args.export_csv:
        print(f"\n📤 Exporting to CSV...")
        export_to_csv(args.results_db, args.export_csv)
    
    print("\n" + "="*70)
    print("✓ Analysis complete")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
