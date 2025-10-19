"""
Text preprocessing module for scientific papers.
Handles special characters, formatting, and removes references section.
"""
import re
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
        
        # Define section order preference for scientific papers
        section_order = [
            'Abstract',
            'Introduction', 
            'Background',
            'Methods',
            'Materials and Methods',
            'Methodology',
            'Results',
            'Discussion',
            'Conclusion',
            'Conclusions',
            'Supplementary',
        ]
        
        # Sections to exclude
        exclude_sections = [
            'References',
            'Bibliography',
            'Acknowledgments',
            'Acknowledgements',
            'Funding',
            'Competing Interests',
            'Author Contributions',
            'Data Availability',
        ]
        
        text_parts = []
        
        # First, add sections in preferred order
        for section_name in section_order:
            for key, value in sections.items():
                if key.lower() == section_name.lower() and value:
                    text_parts.append(f"{key}\n{value}\n")
        
        # Then add any remaining sections not in the list (except excluded ones)
        added_keys = set(s.lower() for s in section_order)
        for key, value in sections.items():
            if (key.lower() not in added_keys and 
                key not in exclude_sections and
                not any(excl.lower() in key.lower() for excl in exclude_sections) and
                value):
                text_parts.append(f"{key}\n{value}\n")
        
        return '\n'.join(text_parts)
    
    def preprocess(self, full_text: Optional[str], full_text_sections: Optional[str]) -> Optional[str]:
        """
        Main preprocessing function.
        Prioritizes full_text_sections, falls back to full_text.
        Returns None if no text is available.
        """
        text = None
        
        # Priority 1: full_text_sections
        if full_text_sections:
            text = self.extract_sections_from_dict(full_text_sections)
        
        # Priority 2: full_text
        if not text and full_text:
            text = full_text
        
        # If no text available, return None
        if not text:
            return None
        
        # Apply preprocessing
        text = self.remove_references(text)
        text = self.clean_text(text)
        
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
