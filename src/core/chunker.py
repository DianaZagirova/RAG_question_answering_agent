"""
Advanced chunking module for scientific papers.
Uses NLTK sentence tokenizer with semantic-aware chunking and section preservation.
"""
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass

# Initialize NLTK sentence tokenizer
try:
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    print("⚠ NLTK not available, falling back to regex-based sentence splitting")
    NLTK_AVAILABLE = False
    sent_tokenize = None


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    chunk_id: int
    start_char: int
    end_char: int
    section: str = ""
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ScientificChunker:
    """
    Semantic chunker optimized for scientific papers.
    Preserves section boundaries and creates overlapping chunks for context.
    """
    
    def __init__(
        self, 
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1500
    ):
        """
        Initialize chunker with parameters.
        
        Args:
            chunk_size: Target size for chunks (in characters)
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum chunk size to keep
            max_chunk_size: Maximum chunk size before forcing split
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # Section header patterns
        self.section_pattern = re.compile(
            r'^(Abstract|Introduction|Background|Methods?|Materials?\s+and\s+Methods?|'
            r'Methodology|Results?|Discussion|Conclusions?|Summary|Supplementary|'
            r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*$',
            re.MULTILINE
        )
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using NLTK's sentence tokenizer.
        NLTK handles abbreviations and scientific text patterns better than regex.
        Falls back to regex if NLTK is unavailable.
        """
        if NLTK_AVAILABLE and sent_tokenize:
            # Use NLTK's punkt tokenizer - trained on scientific text patterns
            sentences = sent_tokenize(text)
            return [s.strip() for s in sentences if s.strip()]
        else:
            # Fallback: simple regex-based splitting
            # Patterns that should NOT be treated as sentence boundaries
            abbreviations = r'(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|etc|i\.e|e\.g|et al|Fig|Tab|vol|no|pp)'
            
            # Replace abbreviation periods temporarily
            text = re.sub(rf'({abbreviations})\.', r'\1<PERIOD>', text, flags=re.IGNORECASE)
            
            # Split on sentence boundaries (., !, ?) followed by space and capital letter
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
            
            # Restore periods
            sentences = [s.replace('<PERIOD>', '.') for s in sentences]
            
            return [s.strip() for s in sentences if s.strip()]
    
    def _detect_sections(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Detect section boundaries in the text.
        Returns list of (section_name, start_pos, end_pos) tuples.
        """
        sections = []
        matches = list(self.section_pattern.finditer(text))
        
        for i, match in enumerate(matches):
            section_name = match.group(1)
            start_pos = match.end()
            
            # End position is the start of next section or end of text
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            
            sections.append((section_name, start_pos, end_pos))
        
        # If no sections found, treat entire text as one section
        if not sections:
            sections = [("Full Text", 0, len(text))]
        
        return sections
    
    def _chunk_text_by_sentences(
        self, 
        text: str, 
        section_name: str = ""
    ) -> List[Chunk]:
        """
        Chunk text by sentences with overlap, respecting semantic boundaries.
        """
        sentences = self._split_by_sentences(text)
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_id = 0
        
        for i, sentence in enumerate(sentences):
            sentence_size = len(sentence)
            
            # If adding this sentence exceeds max_chunk_size, save current chunk
            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    start_char=0,  # Will be adjusted later
                    end_char=0,
                    section=section_name
                ))
                chunk_id += 1
                
                # Create overlap: keep last few sentences
                overlap_sentences = []
                overlap_size = 0
                for sent in reversed(current_chunk):
                    if overlap_size + len(sent) <= self.chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_size += len(sent)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_size = overlap_size
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_size += sentence_size
            
            # If we've reached target chunk size, save it
            if current_size >= self.chunk_size and i < len(sentences) - 1:
                chunk_text = ' '.join(current_chunk)
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    start_char=0,
                    end_char=0,
                    section=section_name
                ))
                chunk_id += 1
                
                # Create overlap
                overlap_sentences = []
                overlap_size = 0
                for sent in reversed(current_chunk):
                    if overlap_size + len(sent) <= self.chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_size += len(sent)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_size = overlap_size
        
        # Add remaining sentences as final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    start_char=0,
                    end_char=0,
                    section=section_name
                ))
        
        return chunks
    
    def chunk_document(
        self, 
        text: str, 
        metadata: Dict = None
    ) -> List[Chunk]:
        """
        Main chunking function that creates section-aware chunks with overlap.
        
        Args:
            text: The preprocessed paper text
            metadata: Additional metadata to attach to chunks (e.g., DOI, title)
            
        Returns:
            List of Chunk objects
        """
        if not text:
            return []
        
        if metadata is None:
            metadata = {}
        
        # Detect sections
        sections = self._detect_sections(text)
        
        all_chunks = []
        global_chunk_id = 0
        
        # Process each section
        for section_name, start_pos, end_pos in sections:
            section_text = text[start_pos:end_pos].strip()
            
            if len(section_text) < self.min_chunk_size:
                continue
            
            # Chunk this section
            section_chunks = self._chunk_text_by_sentences(section_text, section_name)
            
            # Update chunk IDs and metadata
            for chunk in section_chunks:
                chunk.chunk_id = global_chunk_id
                chunk.start_char = start_pos
                chunk.end_char = end_pos
                chunk.metadata = {**metadata, 'section': section_name}
                global_chunk_id += 1
                all_chunks.append(chunk)
        
        return all_chunks
    
    def get_chunk_statistics(self, chunks: List[Chunk]) -> Dict:
        """Get statistics about the chunks."""
        if not chunks:
            return {}
        
        chunk_sizes = [len(chunk.text) for chunk in chunks]
        sections = [chunk.section for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(chunk_sizes) / len(chunks),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'sections': list(set(sections)),
            'chunks_per_section': {
                section: sections.count(section) 
                for section in set(sections)
            }
        }


if __name__ == "__main__":
    # Test the chunker
    test_text = """
    Introduction
    
    This is a test paper about aging biomarkers. We investigate several important markers that have been shown to correlate with biological age. The study of aging has been a fundamental question in biology for decades. Recent advances in molecular biology have enabled us to identify specific biomarkers.
    
    Methods
    
    We collected samples from 100 participants aged 20-80 years. Each sample was analyzed using mass spectrometry. We measured levels of various proteins and metabolites. Statistical analysis was performed using R software.
    
    Results
    
    We identified 15 significant biomarkers associated with chronological age. These markers showed strong correlation coefficients ranging from 0.7 to 0.9. The most significant marker was protein XYZ, which showed a correlation of 0.89.
    
    Discussion
    
    Our findings suggest that these biomarkers could be useful for assessing biological age. This has important implications for aging research and personalized medicine. Future studies should validate these findings in larger cohorts.
    """
    
    chunker = ScientificChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_document(test_text, metadata={'doi': 'test/123'})
    
    print(f"Created {len(chunks)} chunks\n")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i} (Section: {chunk.section}):")
        print(f"  Length: {len(chunk.text)}")
        print(f"  Text preview: {chunk.text[:100]}...")
        print()
    
    stats = chunker.get_chunk_statistics(chunks)
    print("Statistics:", stats)
