"""
Inspect database to understand full_text_sections formats.
This helps ensure we handle all paper formats correctly.
"""
import sqlite3
import json
from collections import defaultdict

DB_PATH = '/home/diana.z/hack/download_papers_pubmed/paper_collection/data/papers.db'

def inspect_database():
    """Inspect the database to understand data formats."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("="*70)
    print("DATABASE INSPECTION")
    print("="*70)
    
    # 1. Total papers
    cursor.execute("SELECT COUNT(*) FROM papers")
    total = cursor.fetchone()[0]
    print(f"\n📊 Total papers in database: {total:,}")
    
    # 2. Papers with full_text_sections
    cursor.execute("""
        SELECT COUNT(*) 
        FROM papers 
        WHERE full_text_sections IS NOT NULL 
        AND full_text_sections != ''
        AND full_text_sections != 'null'
    """)
    with_sections = cursor.fetchone()[0]
    print(f"📄 Papers with full_text_sections: {with_sections:,} ({with_sections/total*100:.1f}%)")
    
    # 3. Papers with full_text
    cursor.execute("""
        SELECT COUNT(*) 
        FROM papers 
        WHERE full_text IS NOT NULL 
        AND full_text != ''
        AND full_text != 'null'
    """)
    with_full_text = cursor.fetchone()[0]
    print(f"📄 Papers with full_text: {with_full_text:,} ({with_full_text/total*100:.1f}%)")
    
    # 4. Papers with either
    cursor.execute("""
        SELECT COUNT(*) 
        FROM papers 
        WHERE (full_text_sections IS NOT NULL AND full_text_sections != '' AND full_text_sections != 'null')
        OR (full_text IS NOT NULL AND full_text != '' AND full_text != 'null')
    """)
    with_either = cursor.fetchone()[0]
    print(f"✅ Papers with EITHER (usable): {with_either:,} ({with_either/total*100:.1f}%)")
    
    # 5. Papers with both
    cursor.execute("""
        SELECT COUNT(*) 
        FROM papers 
        WHERE (full_text_sections IS NOT NULL AND full_text_sections != '' AND full_text_sections != 'null')
        AND (full_text IS NOT NULL AND full_text != '' AND full_text != 'null')
    """)
    with_both = cursor.fetchone()[0]
    print(f"📋 Papers with BOTH: {with_both:,} ({with_both/total*100:.1f}%)")
    
    # 6. Papers with neither
    papers_without = total - with_either
    print(f"❌ Papers with NEITHER (unusable): {papers_without:,} ({papers_without/total*100:.1f}%)")
    
    print("\n" + "="*70)
    print("EXAMINING full_text_sections FORMATS")
    print("="*70)
    
    # Get sample papers with full_text_sections
    cursor.execute("""
        SELECT doi, full_text_sections, full_text
        FROM papers 
        WHERE full_text_sections IS NOT NULL 
        AND full_text_sections != ''
        AND full_text_sections != 'null'
        LIMIT 10
    """)
    
    format_types = defaultdict(int)
    examples = {}
    
    for i, (doi, sections, full_text) in enumerate(cursor.fetchall(), 1):
        print(f"\n--- Sample {i}: {doi} ---")
        
        # Try to parse as JSON
        try:
            parsed = json.loads(sections)
            if isinstance(parsed, dict):
                format_types['dict'] += 1
                if 'dict' not in examples:
                    examples['dict'] = {
                        'doi': doi,
                        'sections': sections[:200],
                        'keys': list(parsed.keys())[:5],
                        'sample_key': list(parsed.keys())[0] if parsed else None,
                        'sample_value': parsed[list(parsed.keys())[0]][:100] if parsed else None
                    }
                print(f"  Format: JSON Dictionary")
                print(f"  Keys ({len(parsed)}): {list(parsed.keys())[:5]}")
                if parsed:
                    first_key = list(parsed.keys())[0]
                    print(f"  Sample ({first_key}): {parsed[first_key][:100]}...")
            elif isinstance(parsed, list):
                format_types['list'] += 1
                if 'list' not in examples:
                    examples['list'] = {
                        'doi': doi,
                        'sections': sections[:200],
                        'length': len(parsed)
                    }
                print(f"  Format: JSON List (length: {len(parsed)})")
                print(f"  First item: {str(parsed[0])[:100]}...")
            else:
                format_types['other_json'] += 1
                print(f"  Format: JSON (type: {type(parsed)})")
        except json.JSONDecodeError:
            format_types['string'] += 1
            if 'string' not in examples:
                examples['string'] = {
                    'doi': doi,
                    'content': sections[:200]
                }
            print(f"  Format: Plain String")
            print(f"  Content: {sections[:100]}...")
        except Exception as e:
            format_types['error'] += 1
            print(f"  Format: Error - {e}")
        
        # Check if has full_text too
        has_ft = full_text and full_text.strip() and full_text != 'null'
        print(f"  Also has full_text: {has_ft}")
    
    print("\n" + "="*70)
    print("FORMAT SUMMARY")
    print("="*70)
    for fmt, count in sorted(format_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {fmt}: {count} samples")
    
    # Check more samples to get better statistics
    print("\n" + "="*70)
    print("ANALYZING 1000 SAMPLES FOR FORMAT DISTRIBUTION")
    print("="*70)
    
    cursor.execute("""
        SELECT full_text_sections
        FROM papers 
        WHERE full_text_sections IS NOT NULL 
        AND full_text_sections != ''
        AND full_text_sections != 'null'
        LIMIT 1000
    """)
    
    large_sample_formats = defaultdict(int)
    for (sections,) in cursor.fetchall():
        try:
            parsed = json.loads(sections)
            if isinstance(parsed, dict):
                large_sample_formats['JSON Dictionary'] += 1
            elif isinstance(parsed, list):
                large_sample_formats['JSON List'] += 1
            else:
                large_sample_formats['JSON Other'] += 1
        except:
            large_sample_formats['Plain String'] += 1
    
    print("\nFormat distribution (1000 samples):")
    total_samples = sum(large_sample_formats.values())
    for fmt, count in sorted(large_sample_formats.items(), key=lambda x: x[1], reverse=True):
        pct = count/total_samples*100 if total_samples > 0 else 0
        print(f"  {fmt}: {count:,} ({pct:.1f}%)")
    
    # Check section names in dictionaries
    print("\n" + "="*70)
    print("COMMON SECTION NAMES (from dict format)")
    print("="*70)
    
    cursor.execute("""
        SELECT full_text_sections
        FROM papers 
        WHERE full_text_sections IS NOT NULL 
        AND full_text_sections != ''
        AND full_text_sections != 'null'
        LIMIT 500
    """)
    
    all_keys = defaultdict(int)
    for (sections,) in cursor.fetchall():
        try:
            parsed = json.loads(sections)
            if isinstance(parsed, dict):
                for key in parsed.keys():
                    all_keys[key] += 1
        except:
            pass
    
    print("\nTop 20 section names:")
    for key, count in sorted(all_keys.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {key}: {count}")
    
    conn.close()
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print("""
1. ✅ Prioritize full_text_sections over full_text (as already implemented)
2. ✅ Handle JSON dictionary format (most common)
3. ⚠ Check if list format needs special handling
4. ✅ Fall back to full_text when sections unavailable
5. ✅ Skip papers with neither field populated
    """)
    
    print("\n" + "="*70)
    print(f"INGESTION TARGET: ~{with_either:,} papers")
    print("="*70)


if __name__ == "__main__":
    inspect_database()
