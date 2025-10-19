"""
Advanced chunking module using LangChain's RecursiveCharacterTextSplitter.
Optimized for scientific papers with section preservation.
"""
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter


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


class ScientificChunkerAdvanced:
    """
    Advanced semantic chunker using LangChain's RecursiveCharacterTextSplitter.
    Preserves section boundaries and creates overlapping chunks for context.
    """
    
    def __init__(
        self, 
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        use_section_aware: bool = True
    ):
        """
        Initialize chunker with LangChain splitter.
        
        Args:
            chunk_size: Target size for chunks (in characters)
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum chunk size to keep
            use_section_aware: If True, process each section separately
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.use_section_aware = use_section_aware
        
        # Initialize LangChain's RecursiveCharacterTextSplitter
        # It tries to split on: "\n\n" → "\n" → " " → ""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",  # Paragraph breaks (highest priority)
                "\n",    # Line breaks
                ". ",    # Sentence ends
                "! ",
                "? ",
                "; ",
                ": ",
                ", ",
                " ",     # Word boundaries
                ""       # Character level (last resort)
            ],
            keep_separator=True,
            is_separator_regex=False
        )
        
        # Section header patterns
        self.section_pattern = re.compile(
            r'^(Abstract|Introduction|Background|Methods?|Materials?\s+and\s+Methods?|'
            r'Methodology|Results?|Discussion|Conclusions?|Summary|Supplementary|'
            r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*$',
            re.MULTILINE
        )
    
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
    
    def chunk_document(
        self, 
        text: str, 
        metadata: Dict = None
    ) -> List[Chunk]:
        """
        Main chunking function using LangChain splitter.
        
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
        
        all_chunks = []
        global_chunk_id = 0
        
        if self.use_section_aware:
            # Detect sections and process each separately
            sections = self._detect_sections(text)
            
            for section_name, start_pos, end_pos in sections:
                section_text = text[start_pos:end_pos].strip()
                
                if len(section_text) < self.min_chunk_size:
                    continue
                
                # Use LangChain splitter on this section
                section_chunks = self.text_splitter.split_text(section_text)
                
                # Convert to Chunk objects
                for chunk_text in section_chunks:
                    if len(chunk_text) >= self.min_chunk_size:
                        chunk = Chunk(
                            text=chunk_text,
                            chunk_id=global_chunk_id,
                            start_char=start_pos,
                            end_char=end_pos,
                            section=section_name,
                            metadata={**metadata, 'section': section_name}
                        )
                        all_chunks.append(chunk)
                        global_chunk_id += 1
        else:
            # Process entire document without section awareness
            text_chunks = self.text_splitter.split_text(text)
            
            for chunk_text in text_chunks:
                if len(chunk_text) >= self.min_chunk_size:
                    chunk = Chunk(
                        text=chunk_text,
                        chunk_id=global_chunk_id,
                        start_char=0,
                        end_char=len(text),
                        section="Full Text",
                        metadata={**metadata, 'section': "Full Text"}
                    )
                    all_chunks.append(chunk)
                    global_chunk_id += 1
        
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


class ScientificChunkerNLTK:
    """
    Alternative chunker using NLTK for sentence tokenization.
    More sophisticated sentence boundary detection for scientific text.
    """
    
    def __init__(
        self, 
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100
    ):
        """Initialize NLTK-based chunker."""
        try:
            import nltk
            # Try to use punkt tokenizer
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                print("Downloading NLTK punkt tokenizer...")
                nltk.download('punkt', quiet=True)
            
            self.sent_tokenize = nltk.sent_tokenize
            self.has_nltk = True
        except ImportError:
            print("⚠ NLTK not available, falling back to simple splitting")
            self.has_nltk = False
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # Section header patterns
        self.section_pattern = re.compile(
            r'^(Abstract|Introduction|Background|Methods?|Materials?\s+and\s+Methods?|'
            r'Methodology|Results?|Discussion|Conclusions?|Summary|Supplementary|'
            r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*$',
            re.MULTILINE
        )
    
    def _tokenize_sentences(self, text: str) -> List[str]:
        """Tokenize text into sentences using NLTK or fallback."""
        if self.has_nltk:
            return self.sent_tokenize(text)
        else:
            # Simple fallback
            return re.split(r'(?<=[.!?])\s+', text)
    
    def _detect_sections(self, text: str) -> List[Tuple[str, int, int]]:
        """Detect section boundaries."""
        sections = []
        matches = list(self.section_pattern.finditer(text))
        
        for i, match in enumerate(matches):
            section_name = match.group(1)
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections.append((section_name, start_pos, end_pos))
        
        if not sections:
            sections = [("Full Text", 0, len(text))]
        
        return sections
    
    def chunk_document(self, text: str, metadata: Dict = None) -> List[Chunk]:
        """Chunk document using NLTK sentence tokenization."""
        if not text:
            return []
        
        if metadata is None:
            metadata = {}
        
        all_chunks = []
        global_chunk_id = 0
        
        sections = self._detect_sections(text)
        
        for section_name, start_pos, end_pos in sections:
            section_text = text[start_pos:end_pos].strip()
            
            if len(section_text) < self.min_chunk_size:
                continue
            
            # Tokenize into sentences
            sentences = self._tokenize_sentences(section_text)
            
            # Group sentences into chunks
            current_chunk = []
            current_size = 0
            
            for sentence in sentences:
                sentence_size = len(sentence)
                
                # If adding this sentence exceeds chunk size, save current chunk
                if current_size + sentence_size > self.chunk_size and current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    if len(chunk_text) >= self.min_chunk_size:
                        chunk = Chunk(
                            text=chunk_text,
                            chunk_id=global_chunk_id,
                            start_char=start_pos,
                            end_char=end_pos,
                            section=section_name,
                            metadata={**metadata, 'section': section_name}
                        )
                        all_chunks.append(chunk)
                        global_chunk_id += 1
                    
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
                
                current_chunk.append(sentence)
                current_size += sentence_size
            
            # Add remaining sentences
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunk = Chunk(
                        text=chunk_text,
                        chunk_id=global_chunk_id,
                        start_char=start_pos,
                        end_char=end_pos,
                        section=section_name,
                        metadata={**metadata, 'section': section_name}
                    )
                    all_chunks.append(chunk)
                    global_chunk_id += 1
        
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
    # Test both chunkers
    test_text = """
    Introduction
    
    Aging is a complex biological process characterized by progressive decline in cellular function. 
    Multiple theories have been proposed to explain the mechanisms of aging. The free radical theory 
    suggests that oxidative damage accumulates over time. Telomere shortening is another well-established 
    hallmark of aging.
    
    Methods
    
    We collected samples from 100 participants aged 20-80 years. Blood samples were analyzed using 
    mass spectrometry. We measured levels of various proteins and metabolites. Statistical analysis 
    was performed using R software version 4.0.
    
    Results
    
    We identified 15 significant biomarkers associated with chronological age. These markers showed 
    strong correlation coefficients ranging from 0.7 to 0.9. The most significant marker was protein 
    XYZ, which showed a correlation of 0.89. Pathway analysis revealed enrichment in oxidative stress 
    and inflammatory pathways.
    """
    
    print("="*60)
    print("Testing LangChain-based Chunker")
    print("="*60)
    chunker1 = ScientificChunkerAdvanced(chunk_size=300, chunk_overlap=50)
    chunks1 = chunker1.chunk_document(test_text, metadata={'doi': 'test/12345'})
    print(f"Created {len(chunks1)} chunks\n")
    for i, chunk in enumerate(chunks1[:3]):
        print(f"Chunk {i} (Section: {chunk.section}):")
        print(f"  Length: {len(chunk.text)}")
        print(f"  Text: {chunk.text[:100]}...\n")
    
    print("\n" + "="*60)
    print("Testing NLTK-based Chunker")
    print("="*60)
    chunker2 = ScientificChunkerNLTK(chunk_size=300, chunk_overlap=50)
    chunks2 = chunker2.chunk_document(test_text, metadata={'doi': 'test/12345'})
    print(f"Created {len(chunks2)} chunks\n")
    for i, chunk in enumerate(chunks2[:3]):
        print(f"Chunk {i} (Section: {chunk.section}):")
        print(f"  Length: {len(chunk.text)}")
        print(f"  Text: {chunk.text[:100]}...\n")
