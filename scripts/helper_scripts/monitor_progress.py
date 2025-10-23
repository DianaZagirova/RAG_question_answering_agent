#!/usr/bin/env python3
"""
Simple script to monitor RAG processing progress
"""
import sqlite3
import time
import sys
from pathlib import Path

def check_progress(results_db):
    if not Path(results_db).exists():
        return 0, 0, "Database not found"
    
    conn = sqlite3.connect(results_db)
    cur = conn.cursor()
    
    # Get total processed
    cur.execute("SELECT COUNT(*) FROM paper_metadata")
    processed = cur.fetchone()[0]
    
    # Get recent processing rate
    cur.execute("""
        SELECT COUNT(*) FROM processing_log 
        WHERE timestamp > datetime('now', '-1 hour')
        AND status = 'success'
    """)
    recent = cur.fetchone()[0]
    
    conn.close()
    
    return processed, recent, "OK"

if __name__ == "__main__":
    results_db = sys.argv[1] if len(sys.argv) > 1 else "rag_results_fast.db"
    
    try:
        while True:
            processed, recent, status = check_progress(results_db)
            print(f"\r📊 Processed: {processed} papers | Last hour: {recent} papers | Status: {status}", end="", flush=True)
            time.sleep(30)
    except KeyboardInterrupt:
        print(f"\n\nFinal count: {processed} papers processed")
