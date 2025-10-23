#!/usr/bin/env python3
"""
View enhanced queries from a saved JSON file or from the RAG system cache.
"""
import json
import argparse
from pathlib import Path


def view_enhanced_queries(file_path: str):
    """Load and display enhanced queries from a JSON file."""
    with open(file_path, 'r') as f:
        queries = json.load(f)
    
    print("\n" + "="*80)
    print("Enhanced Queries")
    print("="*80)
    
    for i, (original, enhanced) in enumerate(queries.items(), 1):
        print(f"\n[{i}] ORIGINAL:")
        # Wrap long lines
        if len(original) > 75:
            words = original.split()
            line = ""
            for word in words:
                if len(line) + len(word) + 1 <= 75:
                    line += word + " "
                else:
                    print(f"    {line.strip()}")
                    line = word + " "
            if line:
                print(f"    {line.strip()}")
        else:
            print(f"    {original}")
        
        print(f"\n    ENHANCED:")
        # Wrap enhanced query
        if len(enhanced) > 75:
            words = enhanced.split()
            line = ""
            for word in words:
                if len(line) + len(word) + 1 <= 75:
                    line += word + " "
                else:
                    print(f"    {line.strip()}")
                    line = word + " "
            if line:
                print(f"    {line.strip()}")
        else:
            print(f"    {enhanced}")
        
        print("-" * 80)
    
    print(f"\nTotal: {len(queries)} enhanced queries")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="View enhanced queries from JSON file"
    )
    parser.add_argument(
        'file',
        type=str,
        nargs='?',
        default='enhanced_queries.json',
        help='Path to enhanced queries JSON file (default: enhanced_queries.json)'
    )
    
    args = parser.parse_args()
    
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        print("\nTo generate enhanced queries, run:")
        print("  python scripts/rag_answer.py --all-questions --save-enhanced-queries enhanced_queries.json")
        return
    
    view_enhanced_queries(str(file_path))


if __name__ == "__main__":
    main()
