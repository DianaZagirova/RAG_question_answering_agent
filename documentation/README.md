# Scientific Papers RAG System

A high-quality Retrieval-Augmented Generation (RAG) system designed specifically for scientific papers, optimized for answering precise questions about biomedical research.

## Features

### 🔬 Scientific Paper Optimized
- **Specialized embeddings**: Uses SPECTER2 model trained on scientific papers
- **Section-aware chunking**: Preserves paper structure (Introduction, Methods, Results, etc.)
- **Smart preprocessing**: Removes references, handles special characters, normalizes text
- **Semantic chunking**: Creates overlapping chunks while maintaining context

### 🎯 High-Quality Retrieval
- **Vector database**: ChromaDB with cosine similarity
- **Two-stage retrieval**: Initial retrieval + reranking capability
- **Metadata filtering**: Search by section, year, journal, topic
- **Relevance scoring**: Returns similarity scores for transparency

### 📊 Robust Processing
- **Text prioritization**: `full_text_sections` → `full_text` → skip
- **Reference removal**: Multiple strategies to exclude bibliography
- **Quality filtering**: Skips papers without substantial text
- **Error handling**: Tracks and reports processing issues

## Architecture

```
Database (papers.db) 
    ↓
Text Preprocessor (clean, remove references)
    ↓
Semantic Chunker (section-aware, overlapping)
    ↓
Embeddings (SPECTER2 / all-MiniLM-L6-v2)
    ↓
Vector DB (ChromaDB)
    ↓
Query Interface (RAG retrieval)
```

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: The system will automatically download the embedding model on first run (~500MB for SPECTER2).

### 2. Verify Database

Ensure your `papers.db` is accessible:
```bash
python check_db.py
```

## Usage

### Step 1: Ingest Papers

Process papers from the database into the RAG system:

```bash
# Full ingestion (all papers)
python ingest_papers.py

# Test with limited papers
python ingest_papers.py --limit 100

# Custom parameters
python ingest_papers.py \
    --db-path /path/to/papers.db \
    --collection-name my_papers \
    --chunk-size 1200 \
    --chunk-overlap 250
```

**Parameters:**
- `--db-path`: Path to papers.db (default: auto-detected)
- `--collection-name`: ChromaDB collection name (default: "scientific_papers")
- `--persist-dir`: Where to store ChromaDB (default: "./chroma_db")
- `--chunk-size`: Target chunk size in characters (default: 1000)
- `--chunk-overlap`: Overlap between chunks (default: 200)
- `--limit`: Process only N papers (for testing)
- `--reset`: Reset existing collection before ingesting
- `--embedding-model`: Primary embedding model (default: "allenai/specter2")
- `--backup-model`: Fallback model (default: "sentence-transformers/all-MiniLM-L6-v2")

**Expected output:**
```
Processing papers: 100%|████████████| 42735/42735
Ingestion Complete!
Total papers: 42735
Successfully processed: 42500
Total chunks: 425000
```

### Step 2: Query the System

#### Interactive Mode

```bash
python query_rag.py
```

Example interaction:
```
❓ Your question: What biomarkers are suggested for aging?
🔍 Retrieving top 5 relevant chunks...

📚 Retrieved Sources:
[Source 1]
  Title: Telomere Length and Biological Aging
  DOI: 10.1234/aging.2020
  Section: Results
  Relevance: 0.892

[Context with relevant excerpts...]
```

#### Single Query Mode

```bash
python query_rag.py --question "Does the paper suggest a biomarker for aging?"
```

#### Batch Processing

Create a file `questions.txt`:
```
What biomarkers are suggested for aging?
How does telomere length correlate with age?
What are the mechanisms of cellular senescence?
```

Process all questions:
```bash
python query_rag.py --batch questions.txt --output results.json
```

#### Advanced Options

```bash
# Retrieve more results
python query_rag.py --question "..." --n-results 10

# Get LLM-formatted output
python query_rag.py --question "..." --format llm

# Simple output format
python query_rag.py --question "..." --format simple
```

**Interactive commands:**
- `<question>` - Ask a question
- `<question> [n=10]` - Retrieve 10 results
- `<question> [format=simple]` - Use simple format
- `stats` - Show database statistics
- `help` - Show help
- `quit` - Exit

## Components

### 1. Text Preprocessor (`text_preprocessor.py`)

Handles text cleaning and normalization:
- Removes special characters and artifacts
- Normalizes whitespace
- Extracts and concatenates sections from JSON
- Removes references section (multiple strategies)
- Filters out figures, tables, URLs, emails

### 2. Chunker (`chunker.py`)

Creates semantic chunks optimized for scientific papers:
- Section-aware splitting
- Sentence-based chunking (respects abbreviations)
- Overlapping chunks for context preservation
- Configurable chunk size and overlap
- Maintains section metadata

### 3. RAG System (`rag_system.py`)

Core retrieval system:
- ChromaDB for vector storage
- SPECTER2 embeddings (scientific papers specialized)
- Cosine similarity search
- Metadata filtering support
- Two-stage retrieval with reranking
- Context formatting for LLMs

### 4. Ingestion Pipeline (`ingest_papers.py`)

End-to-end processing:
- Batch fetching from SQLite
- Progress tracking with statistics
- Error handling and logging
- Configurable processing parameters

### 5. Query Interface (`query_rag.py`)

User-facing query system:
- Interactive mode
- Batch processing
- Multiple output formats
- Relevance scoring
- Source attribution

## Configuration

### Embedding Models

**Primary: SPECTER2** (`allenai/specter2`)
- Trained on 146M+ scientific papers
- Optimized for biomedical literature
- Best quality for scientific queries
- Size: ~440MB

**Backup: all-MiniLM-L6-v2** (`sentence-transformers/all-MiniLM-L6-v2`)
- General-purpose model
- Faster and smaller
- Good fallback option
- Size: ~80MB

### Chunking Parameters

**Recommended settings:**
- Chunk size: 1000-1500 characters
  - Smaller: More precise retrieval
  - Larger: More context per chunk
- Chunk overlap: 150-250 characters
  - Ensures continuity across boundaries
  - Prevents information loss

### Retrieval Parameters

**n_results**: Number of chunks to retrieve
- 3-5: Focused, specific questions
- 5-10: Broader questions requiring synthesis
- 10+: Comprehensive coverage

## Data Flow

### Paper Processing
1. **Fetch** from database (prioritize `full_text_sections`)
2. **Parse** JSON sections if available
3. **Remove** references section
4. **Clean** special characters and artifacts
5. **Split** into section-aware chunks
6. **Overlap** chunks for context
7. **Embed** using SPECTER2
8. **Store** in ChromaDB

### Query Processing
1. **User question** input
2. **Embed** question using same model
3. **Search** vector database (cosine similarity)
4. **Rank** results by relevance
5. **Format** context with metadata
6. **Return** top-k results with sources

## Statistics

After ingestion, you'll see:
```python
{
  "total_papers": 42735,
  "processed_papers": 42500,
  "skipped_papers": 235,
  "total_chunks": 425000,
  "avg_chunks_per_paper": 10.0
}
```

Database statistics:
```python
{
  "collection_name": "scientific_papers",
  "total_chunks": 425000,
  "embedding_model": "allenai/specter2",
  "persist_directory": "./chroma_db"
}
```

## Example Queries

### Specific Biomarker Questions
```
"What biomarkers are suggested for aging in this paper?"
"Does this research propose any novel aging biomarkers?"
"Which proteins are identified as aging markers?"
```

### Mechanism Questions
```
"What are the mechanisms of cellular senescence discussed?"
"How does oxidative stress contribute to aging?"
"What role do telomeres play in aging?"
```

### Method Questions
```
"What techniques were used to measure biological age?"
"How were the biomarkers validated?"
"What statistical methods were employed?"
```

## Troubleshooting

### Issue: Model download fails
**Solution**: Download manually or use backup model:
```bash
python query_rag.py --embedding-model sentence-transformers/all-MiniLM-L6-v2
```

### Issue: Out of memory
**Solution**: Process in smaller batches:
```bash
python ingest_papers.py --limit 10000
# Wait for completion, then continue with next batch
```

### Issue: Slow processing
**Solution**: 
- Reduce chunk size: `--chunk-size 800`
- Use backup model (faster)
- Process on GPU if available

### Issue: Low quality results
**Solutions**:
- Increase n_results: `--n-results 10`
- Adjust chunk size (try 1200-1500)
- Increase chunk overlap: `--chunk-overlap 300`
- Check if SPECTER2 is being used (not backup model)

## File Structure

```
rag_agent/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── check_db.py               # Database inspection tool
├── text_preprocessor.py      # Text cleaning and preprocessing
├── chunker.py                # Semantic chunking
├── rag_system.py             # Core RAG system
├── ingest_papers.py          # Ingestion pipeline
├── query_rag.py              # Query interface
├── chroma_db/                # Vector database (created on first run)
├── ingestion_progress.json   # Processing statistics
└── ingestion_complete.json   # Final statistics
```

## Performance

**Ingestion speed**: ~50-100 papers/second (depending on hardware)
**Query latency**: ~100-300ms per query
**Storage**: ~5-10 GB for 42,000 papers (depends on chunks)

## Advanced Usage

### Custom Preprocessing

Modify `text_preprocessor.py` to adjust:
- Reference detection patterns
- Special character handling
- Section prioritization

### Custom Chunking

Modify `chunker.py` to adjust:
- Section detection patterns
- Sentence splitting rules
- Chunk size strategy

### Integration with LLMs

```python
from rag_system import ScientificRAG, create_context_for_llm

rag = ScientificRAG()
response = rag.answer_question("What biomarkers for aging?")
llm_prompt = create_context_for_llm(response)

# Use llm_prompt with OpenAI, Anthropic, etc.
```

## Citation

If you use this system in your research, please cite the underlying models:

**SPECTER2**:
```
@inproceedings{singh2022scirepeval,
  title={SciRepEval: A Multi-Format Benchmark for Scientific Document Representations},
  author={Singh, Amanpreet et al.},
  booktitle={EMNLP},
  year={2022}
}
```

## License

MIT License - See LICENSE file for details

## Support

For issues or questions:
1. Check troubleshooting section
2. Review error logs in `ingestion_progress.json`
3. Run with `--limit 10` to test on small dataset
4. Check that papers.db has full text data
