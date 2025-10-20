"""
Text preprocessing module for scientific papers.
Handles special characters, formatting, and removes references section.
"""
import re
import os
import unicodedata
from typing import Dict, Optional


class TextPreprocessor:
    """Preprocessor for scientific paper text."""
    
    def __init__(self):
        # Common reference section headers (case insensitive)
        self.reference_headers = [
            r'\n\s*references?\s*\n',
            r'\n\s*bibliography\s*\n',
            r'\n\s*works?\s+cited\s*\n',
            r'\n\s*literature\s+cited\s*\n',
            r'\n\s*reference\s+list\s*\n',
        ]
        
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters, normalizing whitespace,
        and handling common formatting issues in scientific papers.
        """
        if not text:
            return ""
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove non-ASCII characters that might cause issues
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        # Remove multiple consecutive spaces
        text = re.sub(r' +', ' ', text)
        
        # Normalize line breaks (remove excessive newlines)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove page numbers and common artifacts
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # Remove figure/table references like "Fig. 1" or "Table 2" standalone lines
        text = re.sub(r'\n\s*(Fig\.|Figure|Table|Supplementary)\s+\d+[A-Za-z]?\s*[:\.]?\s*\n', '\n', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove DOI patterns
        text = re.sub(r'doi:\s*\S+', '', text, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        text = text.strip()
        
        return text
    
    def remove_references(self, text: str) -> str:
        """
        Remove references section from the text.
        Tries multiple patterns to detect reference sections.
        """
        if not text:
            return ""
        
        # Try each reference header pattern
        for pattern in self.reference_headers:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                # Cut text at the reference section
                text = text[:match.start()]
                break
        
        # Additional heuristic: if we see numbered references like [1], [2]... 
        # in high density at the end, cut it
        lines = text.split('\n')
        ref_density_threshold = 0.5  # 50% of lines have reference markers
        
        for i in range(len(lines) - 1, max(0, len(lines) - 50), -1):
            # Check last 50 lines for reference pattern density
            window = lines[i:]
            ref_count = sum(1 for line in window if re.search(r'^\s*\[\d+\]|\d+\.\s+\w+.*\(\d{4}\)', line))
            
            if len(window) > 5 and ref_count / len(window) > ref_density_threshold:
                text = '\n'.join(lines[:i])
                break
        
        return text.strip()
    
    def extract_sections_from_dict(self, sections_json: str) -> str:
        """
        Extract and concatenate text from full_text_sections JSON.
        Handles both dict and list formats.
        Prioritizes meaningful sections and excludes references.
        """
        import json
        
        if not sections_json:
            return ""
        
        try:
            sections = json.loads(sections_json)
        except json.JSONDecodeError:
            # If JSON parsing fails, return empty (will fall back to full_text)
            return ""
        
        # Handle list format (0.5% of cases)
        if isinstance(sections, list):
            # List format: just concatenate all text items
            text_parts = []
            for item in sections:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict):
                    # If list contains dicts, extract text from dict values
                    for value in item.values():
                        if isinstance(value, str):
                            text_parts.append(value)
            return '\n\n'.join(text_parts)
        
        # Handle dict format (99.5% of cases)
        if not isinstance(sections, dict):
            return ""
        # If sections dict seems to be a fallback single blob, just return a budgeted slice
        keys = list(sections.keys())
        if len(keys) <= 2:
            key0 = keys[0].lower() if keys else ''
            if (not keys or
                'full text' in key0 or 'fallback' in key0 or 'extraction' in key0 or
                len(keys[0]) > 40):
                blob = sections.get(keys[0], '') if keys else ''
                blob = blob if isinstance(blob, str) else ''
                BUDGET = int(os.getenv('PREPROCESS_BUDGET_CHARS', '40000'))
                return (blob[:BUDGET] + '\n') if blob else ''
        
        # Scoring-based prioritization and budgeted assembly
        priority_scores = {
            'abstract': 100,
            'introduction': 90,
            'background': 70,
            'results': 85,
            'discussion': 88,
            'conclusion': 82,
            'conclusions': 82
            
        }
        # Exclusions
        exclude_sections = [
            'references', 'bibliography', 'acknowledgments', 'acknowledgements',
            'funding', 'competing interests', 'author contributions', 'data availability', 
            'methods', 'materials', 'materials and methods', 'supplementary', 'appendix', 
            'conflict of interest', 'acknowledgement', 'statistical analysis', 
            'experimental setup', 'datasets'
        ]

        # Compute scores per available section
        scored = []
        for key, value in sections.items():
            if not value:
                continue
            key_l = key.lower()
            if any(excl in key_l for excl in exclude_sections):
                continue
            base = priority_scores.get(key_l, 50)
            # Boost sections that explicitly mention theory/model/hypothesis
            if re.search(r"theor|model|hypoth", key_l):
                base += 15
            if re.search(r"result|analysis", key_l):
                base += 15
            # Penalize methods
            if re.search(r"method|materials", key_l):
                base -= 25
            scored.append((base, key, value))

        # Sort by score desc, then by a small tie-breaker on key
        scored.sort(key=lambda x: (-x[0], x[1].lower()))

        # Assemble under a character budget with adaptive allocation
        BUDGET = int(os.getenv('PREPROCESS_BUDGET_CHARS', '20000'))
        
        # Adaptive per-section caps based on priority
        def get_section_cap(key_lower):
            if key_lower in ['abstract']:
                return 3000  # Abstract: compact but complete
            elif key_lower in ['introduction', 'background']:
                return 5000  # Intro: more context needed
            elif key_lower in ['discussion', 'conclusions', 'conclusion']:
                return 6000  # Discussion: most theory-rich
            elif key_lower in ['results']:
                return 4000  # Results: less theory, more data
            else:
                return 3000  # Other sections: minimal
        
        # Smart sampling: take beginning + end for long sections
        def smart_sample(text, max_chars):
            if len(text) <= max_chars:
                return text
            # Take first 60% and last 40% to capture intro and conclusions
            first_part = int(max_chars * 0.6)
            last_part = max_chars - first_part
            return text[:first_part] + "\n[...]\n" + text[-last_part:]
        
        assembled = []
        used = 0
        for _, key, value in scored:
            if used >= BUDGET:
                break
            
            section_cap = get_section_cap(key.lower())
            part = smart_sample(value, section_cap)
            chunk = f"{key}\n{part}\n\n"
            
            if used + len(chunk) > BUDGET:
                # trim final chunk to fit
                remaining = max(0, BUDGET - used)
                chunk = chunk[:remaining]
            assembled.append(chunk)
            used += len(chunk)

        return ''.join(assembled)
    
    def preprocess(self, full_text: Optional[str], full_text_sections: Optional[str], abstract: Optional[str] = None) -> Optional[str]:
        """
        Main preprocessing function.
        Prioritizes full_text_sections, falls back to full_text.
        Returns None if no text is available.
        """
        text = None
        used_sections = False
        
        # Priority 1: full_text_sections
        if full_text_sections:
            section_text = self.extract_sections_from_dict(full_text_sections)
            if section_text:
                text = section_text
                used_sections = True
        
        # Priority 2: full_text
        if not text and full_text:
            text = full_text
        
        # If no text available, return None
        if not text:
            return None

        # If we used sectioned text and there's no explicit Abstract section, inject the DB abstract (if available)
        if used_sections and abstract:
            # Look for an 'Abstract' header case-insensitively at line starts
            has_abstract_header = bool(re.search(r'^\s*abstract\s*$', text, flags=re.IGNORECASE | re.MULTILINE))
            if not has_abstract_header:
                abstract_block = f"Abstract\n{abstract}\n\n"
                text = abstract_block + text
        
        # Apply preprocessing
        text = self.remove_references(text)
        text = self.clean_text(text)
        
        # Enforce final budget cap if specified (keeps final prompt size under control)
        try:
            final_budget = int(os.getenv('PREPROCESS_BUDGET_CHARS', '40000'))
            if final_budget > 0 and len(text) > final_budget:
                text = text[:final_budget]
        except Exception:
            pass

        # Final check - ensure we have substantial text
        if len(text.strip()) < 100:  # Minimum 100 characters
            return None
        
        return text


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = TextPreprocessor()
    
    test_text = """
    Introduction
    
    This is a test paper about aging biomarkers.
    
    Results
    
    We found several important markers. Fig. 1 shows the results.
    
    References
    
    [1] Smith et al. (2020) Nature
    [2] Jones et al. (2021) Science
    """
    
    processed = preprocessor.clean_text(test_text)
    processed = preprocessor.remove_references(processed)
    
    print("Original length:", len(test_text))
    print("Processed length:", len(processed))
    print("\nProcessed text:\n", processed)
