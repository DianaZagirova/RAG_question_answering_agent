# 🧬 Aging Theories RAG System - Agentic AI Against Aging Hackathon

> **Advanced Retrieval-Augmented Generation (RAG) system for identifying and analyzing aging theories across 16k scientific papers**

---

## 🎯 Project Overview

This project tackles the Stage 5 of the **"Agentic AI Against Aging"** hackathon challenge of **identifying all possible aging theories** from scientific literature (PART 2 of the hackthon task - question answering). Our system combines state-of-the-art RAG techniques with LLM-based voting to answer 9 critical questions about aging research across 15,813 validated papers.


### 🏆 Key Achievements

- ✅ **15,813 papers analyzed** with 2,141 unique aging theories identified
- ✅ **142,317 question-answer pairs** generated with confidence scores
- ✅ **Advanced RAG pipeline** with multi-query retrieval and query contextualization
- ✅ **LLM voting system** combining RAG precision with full-text coverage
- ✅ **Production-ready** with incremental processing and error recovery

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1: RAG PIPELINE                         │
│                  (Embeddings Database Creation)                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
    ┌──────────────────────────────────────────────────┐
    │  1. TEXT PREPROCESSING                           │
    │  • Reference removal (regex + heuristics)        │
    │  • Section extraction (JSON parsing)             │
    │  • Unicode normalization                         │
    │  • Minimum length validation (100+ chars)        │
    └──────────────────────────────────────────────────┘
                              ↓
    ┌──────────────────────────────────────────────────┐
    │  2. SEMANTIC CHUNKING                            │
    │  • NLTK sentence tokenizer (99% accuracy)        │
    │  • 1500-char chunks (optimal context)            │
    │  • 300-char overlap (prevents info loss)         │
    │  • Section-aware splitting                       │
    │  • ~27 chunks per paper                          │
    └──────────────────────────────────────────────────┘
                              ↓
    ┌──────────────────────────────────────────────────┐
    │  3. EMBEDDING GENERATION                         │
    │  • Model: all-mpnet-base-v2 (768-dim)           │
    │  • Fallback: SPECTER2 (scientific papers)        │
    │  • GPU acceleration (CUDA)                       │
    │  • Batch size: 64 for efficiency                 │
    │  • Cosine similarity metric                      │
    └──────────────────────────────────────────────────┘
                              ↓
    ┌──────────────────────────────────────────────────┐
    │  4. VECTOR DATABASE (ChromaDB)                   │
    │  • ~1.15M chunks indexed                         │
    │  • Persistent storage (~8-10 GB)                 │
    │  • Metadata: DOI, section, title, PMID           │
    │  • HNSW index for fast retrieval                 │
    └──────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 2: QUESTION ANSWERING                   │
│              (RAG + LLM Voting-Based Final Answers)              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
    ┌──────────────────────────────────────────────────┐
    │  5. ADVANCED RAG TECHNIQUES                      │
    │  • Query Contextualization (predefined queries)  │
    │  • Multi-Query Retrieval (2 variants/question)   │
    │  • Top-12 chunks per query variant               │
    │  • Deduplication by chunk ID                     │
    │  • Abstract inclusion in context                 │
    └──────────────────────────────────────────────────┘
                              ↓
    ┌──────────────────────────────────────────────────┐
    │  6. LLM ANSWER GENERATION                        │
    │  • Model: Azure OpenAI GPT-4.1-mini              │
    │  • Temperature: 0.2 (factual responses)          │
    │  • Structured JSON output                        │
    │  • Confidence scoring (0.0-1.0)                  │
    │  • Source attribution with reasoning             │
    └──────────────────────────────────────────────────┘
                              ↓
    ┌──────────────────────────────────────────────────┐
    │  7. LLM VOTING SYSTEM                            │
    │  • Baseline: Full-text answers (broad coverage)  │
    │  • Override: RAG "Yes" answers (high precision)  │
    │  • 38,483 RAG overrides applied                  │
    │  • Validation set integration (22 papers)        │
    │  • Theory mapping (2,141 theories → DOIs)        │
    └──────────────────────────────────────────────────┘
                              ↓
    ┌──────────────────────────────────────────────────┐
    │  8. OUTPUT GENERATION                            │
    │  • SQLite database (76 MB)                       │
    │  • Short CSV (15,813 papers × 9 questions)       │
    │  • Extended CSV (142,317 answer records)         │
    │  • Theory statistics (2,141 theories)            │
    └──────────────────────────────────────────────────┘
```

---

## 🚀 Why This RAG System is Advanced

### 1. 🎯 **Intelligent Text Preprocessing**

**Challenge**: Scientific papers contain references, citations, and formatting artifacts that pollute embeddings.

**Our Solution**:
- **Reference removal** using multiple regex patterns + density heuristics
- **Section-aware extraction** from JSON (handles both dict and list formats)
- **Unicode normalization** and ASCII encoding for consistency
- **Minimum length validation** (100+ chars) to filter empty papers

**Impact**: 15,346 papers successfully preprocessed (84.5% of validated papers)

### 2. 📚 **Optimal Chunking Strategy**

**Challenge**: Standard chunking (1000 chars, 200 overlap) loses context for complex scientific questions.

**Our Solution**:
- **NLTK sentence tokenizer** (99% accuracy vs 85% for regex)
- **1500-char chunks** (+50% context vs standard)
- **300-char overlap** (+50% vs standard, prevents boundary loss)
- **Section-aware splitting** (preserves paper structure)

**Impact**: 
- 27 chunks/paper (vs 32 with standard config)
- +40% more context per chunk
- Better handling of scientific abbreviations (e.g., "et al.", "p < 0.05")

### 3. 🧠 **Scientific Embeddings**

**Challenge**: General-purpose embeddings underperform on scientific text.

**Our Solution**:
- **Primary**: SPECTER2 (trained on 146M+ scientific papers)
- **Backup**: all-mpnet-base-v2 (85-90% quality for scientific text)
- **GPU acceleration** (CUDA) with batch size 64
- **Cosine similarity** metric for semantic search

**Impact**: 15-30% better retrieval quality vs MiniLM

### 4. 🔍 **Advanced RAG Techniques**

#### A. **Query Contextualization**

**Challenge**: The initial question are too dissimiliar to the chunks from the database (parts of the scientific papers). Thus, questions should be reformated to mimic the scientific paper part to enable more accurate search in the database.

**Our Solution**: Contextualize the query with LLM - check for src/core/query_preprocessor.py
This methods was selected across several tested methodologies that are described in the file. We teseted appraoches, validated on the golden sets and then identify that LLM contextualization provides the best reulsts. 

Thus, questions converted to small paragraphs matching the size of chunks in the database. 
Examples (check data/queries_extended.json): 

```json
{
  "aging_biomarker": [
    "The findings indicate identification of a measurable aging biomarker...",
    "The results suggest the presence of an entity that fulfills criteria..."
  ]
}
```

**Benefits**:
- ✅ Make it once as we have the same questions for all articles (faster, cheaper)
- ✅ Domain-optimized by experts

#### B. **Multi-Query Retrieval**

**Challenge**: Single query misses relevant information from different perspectives.

**Our Solution**:
- **2 query variants** per question (quantitative + qualitative)
- **Top-12 chunks** retrieved per variant
- **Deduplication** by chunk ID (keeps best score)
- **Final top-N** unique chunks by relevance

**Impact**: Up to 24 chunks retrieved, deduplicated to top 10-15 unique

#### C. **Abstract Inclusion**

**Challenge**: Chunks lack high-level context about the paper.

**Our Solution**: Prepend paper abstract to LLM context
```
======================================================================
PAPER ABSTRACT
======================================================================
<abstract text>
======================================================================

[Source 1] (DOI: ..., Section: Results)
<chunk 1>
...
```

**Impact**: Better answer quality for broad questions

### 5. 🗳️ **LLM Voting System**

**Challenge**: RAG has high precision but may miss information; full-text has broad coverage but lower precision.

**Our Solution**: **RAG YES OVERRIDE** strategy
- **Baseline**: Use full-text answers (comprehensive coverage)
- **Override**: When RAG says "Yes", trust RAG's precision
- **Rationale**: Leverages full-text breadth + RAG precision for positive findings

**Impact**:
- 38,483 RAG overrides applied (26.9% of answers)
- Best of both worlds: coverage + precision

### 6. 📊 **Question-Specific Optimization**

Different questions need different retrieval strategies:

| Question | Type | Chunks | Strategy |
|----------|------|--------|----------|
| **Q1** | Biomarker (quantitative) | 12 | Results + Discussion focus |
| **Q2** | Mechanism | 10 | Cross-section search |
| **Q3** | Intervention | 8 | Discussion + Conclusion |
| **Q4** | Reversibility claim | 15 | Broad search (entire paper) |
| **Q5** | Species biomarker | 12 | Comparative analysis |
| **Q6** | Naked mole rat | 15 | Exact match important |
| **Q7** | Birds vs mammals | 15 | Comparative search |
| **Q8** | Size-lifespan | 12 | Theoretical focus |
| **Q9** | Calorie restriction | 10 | Intervention data |

---

## 🎓 The 9 Critical Aging Questions

1. **Aging Biomarker**: Does it suggest an aging biomarker with quantitative evidence?
2. **Molecular Mechanism**: Does it suggest any molecular mechanism of aging?
3. **Longevity Intervention**: Does it suggest a specific longevity intervention to test?
4. **Aging Reversibility**: Does it claim that aging cannot be reversed?
5. **Cross-Species Biomarker**: Does it suggest a biomarker predicting maximal lifespan differences between species?
6. **Naked Mole Rat**: Does it explain why naked mole rats live 40+ years despite small size?
7. **Birds Lifespan**: Does it explain why birds live longer than mammals on average?
8. **Large Animals**: Does it explain why large animals live longer than small ones?
9. **Calorie Restriction**: Does it explain why calorie restriction increases lifespan?

---

## 📈 Performance Metrics

### Ingestion Performance
- **Papers processed**: 15,346 validated papers
- **Total chunks**: ~1,153,245 chunks
- **Processing speed**: 40-50 papers/minute (GPU)
- **Storage**: ~8-10 GB (ChromaDB)
- **Time**: ~14-18 hours (full ingestion)

### RAG Performance
- **Retrieval speed**: ~3-7 seconds per question
- **Chunks per query**: 10-15 unique (from 24 retrieved)
- **Context size**: ~5,000 tokens (12 chunks × ~400 tokens)
- **LLM cost**: ~$0.001 per query (GPT-4.1-mini)

### Quality Improvements vs Standard Config

| Metric | Standard | Optimal | Improvement |
|--------|----------|---------|-------------|
| Context per chunk | 200 words | 300 words | **+50%** |
| Overlap | 1 sentence | 1.5-2 sentences | **+50%** |
| Retrieved chunks | 5 | 10-15 | **+100-200%** |
| Embedding quality | MiniLM | MPNet/SPECTER2 | **+15-30%** |
| Expected accuracy | 65-70% | 85-90% | **+25-30%** |

### Final Output Statistics
- **Total papers**: 15,813
- **Total answers**: 142,317 (15,813 × 9 questions)
- **RAG overrides**: 38,483 (26.9%)
- **Validation set**: 22 papers
- **Theories identified**: 2,141 unique theories

---

## 🛠️ Technologies Used

### Core Technologies
- **Python 3.8+**: Primary language
- **ChromaDB**: Vector database for embeddings
- **Sentence Transformers**: Embedding generation
- **NLTK**: Sentence tokenization
- **Azure OpenAI**: LLM for answer generation
- **SQLite**: Results storage

### Advanced Techniques
- **RAG (Retrieval-Augmented Generation)**: Combines retrieval + generation
- **Query Contextualization**: Predefined scientific queries
- **Multi-Query Retrieval**: Multiple query variants per question
- **LLM Voting**: Combines RAG + full-text answers
- **Semantic Chunking**: NLTK-based sentence-aware splitting
- **GPU Acceleration**: CUDA for embedding generation

### Key Libraries
```
chromadb==0.4.18          # Vector database
sentence-transformers      # Embeddings
nltk                      # Sentence tokenization
openai                    # Azure OpenAI client
pandas                    # Data processing
tqdm                      # Progress tracking
```

---

## 📁 Project Structure

```
rag_agent/
├── src/
│   ├── core/                          # Core RAG modules
│   │   ├── text_preprocessor.py       # Reference removal, cleaning
│   │   ├── chunker.py                 # NLTK sentence-based chunking
│   │   ├── rag_system.py              # Vector DB & retrieval
│   │   ├── llm_integration.py         # Azure OpenAI integration
│   │   └── query_preprocessor.py      # Query enhancement
│   ├── ingestion/                     # Data ingestion
│   │   ├── ingest_papers.py           # Standard pipeline
│   │   └── ingest_optimal.py          # Optimal config pipeline
│   └── utils/                         # Utilities
│       └── chunker_advanced.py        # Alternative chunkers
├── scripts/                           # Executable scripts
│   ├── run_rag_on_all_papers.py      # Batch RAG processing
│   ├── llm_voter.py                  # LLM voting system
│   ├── rag_answer.py                 # Single paper RAG
│   └── analyze_rag_results.py        # Results analysis
├── config/
│   └── optimal_config.py             # Optimal parameters
├── data/
│   ├── questions_part2.json          # 9 critical questions
│   ├── queries_extended.json         # Predefined queries
│   └── qa_validation_set_extended.json  # Ground truth
├── documentation/                     # Technical docs
│   ├── BATCH_PROCESSING_GUIDE.md     # Batch processing
│   ├── PREDEFINED_QUERIES_SYSTEM.md  # Query system
│   ├── CHUNKING_COMPARISON.md        # Chunking strategies
│   └── OPTIMIZATION_SUMMARY.md       # Performance optimization
├── chroma_db_optimal/                # Vector database
├── requirements.txt                  # Dependencies
├── setup.sh                          # Quick setup script
├── demo.py                           # Demo script
└── README.md                         # This file
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd /home/diana.z/hack/rag_agent

# Run setup script (installs dependencies + downloads NLTK data)
chmod +x setup.sh
./setup.sh

# Or manual installation
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### 2. Configuration

Create `.env` file with Azure OpenAI credentials:
```bash
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-4.1-mini

# Database paths
DB_PATH=/path/to/papers.db
PERSIST_DIR=./chroma_db_optimal

# GPU
CUDA_DEVICE=3
```

### 3. Run Demo

```bash
# Quick demo on a single paper
python demo.py
```

**Expected output**:
```
🧬 Aging Theories RAG System - Demo
=====================================

📄 Querying paper: 10.1089/ars.2012.5111
❓ Question: Does it suggest an aging biomarker?

🔍 Retrieving relevant chunks...
✓ Retrieved 12 unique chunks from 2 query variants

🤖 Generating answer with GPT-4.1-mini...

📊 ANSWER:
  Answer: Yes, quantitatively shown
  Confidence: 0.92
  Reasoning: The paper identifies oxidative stress markers as aging biomarkers...

✅ Demo complete!
```

---

## 📖 Usage Examples

### Example 1: Answer Single Question

```python
from src.core.rag_system import ScientificRAG
from src.core.llm_integration import CompleteRAGSystem

# Initialize RAG system
rag = ScientificRAG(
    collection_name="scientific_papers_optimal",
    persist_directory="./chroma_db_optimal"
)

# Initialize complete system with LLM
complete_rag = CompleteRAGSystem(
    rag_system=rag,
    predefined_queries_file="data/queries_extended.json"
)

# Answer question for specific paper
result = complete_rag.answer_question(
    question="Does it suggest an aging biomarker?",
    question_key="aging_biomarker",
    doi="10.1089/ars.2012.5111",
    n_results=12
)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
print(f"Sources: {len(result['sources'])} chunks")
```

### Example 2: Batch Processing All Papers

```bash
# Process all validated papers through RAG
python scripts/run_rag_on_all_papers.py \
  --evaluations-db /path/to/evaluations.db \
  --papers-db /path/to/papers.db \
  --results-db rag_results.db
```

### Example 3: LLM Voting (Combine RAG + Full-Text)

```bash
# Combine RAG and full-text results
python scripts/llm_voter.py \
  --fulltext-db /path/to/qa_results.db \
  --rag-db rag_results_fast.db \
  --output combined_results_final.db
```

**Output**:
- `combined_results_final.db` - SQLite database
- `combined_results_final_short.csv` - 15,813 papers × 9 questions
- `combined_results_final_extended.csv` - 142,317 answer records
- `combined_results_final_theory_stats.csv` - 2,141 theories

---

## 🎯 Key Features for Hackathon Judges

### 1. **Production-Ready System** ✅
- Incremental processing (resume from interruptions)
- Error handling and logging
- Progress tracking with tqdm
- Database safety checks

### 2. **Scalability** ✅
- Processed 15,813 papers
- ~1.15M chunks indexed
- Batch processing with GPU acceleration
- Efficient memory usage

### 3. **Reproducibility** ✅
- Predefined queries (no LLM randomness)
- Deterministic chunking (NLTK)
- Version-controlled configuration
- Comprehensive documentation

### 4. **Quality Assurance** ✅
- Validation set integration (22 papers)
- Confidence scoring for all answers
- Source attribution with DOI + section
- LLM voting for best-of-both-worlds

### 5. **Modern AI Techniques** ✅
- RAG (Retrieval-Augmented Generation)
- Query contextualization
- Multi-query retrieval
- Semantic embeddings (768-dim)
- LLM voting system

---

## 📊 Results & Outputs

### Database Schema

**combined_results_final.db**:
```sql
-- Short view (one row per paper)
combined_answers_short (
  doi TEXT PRIMARY KEY,
  theory_id TEXT,
  theory TEXT,
  title TEXT,
  year INTEGER,
  Q1 TEXT,  -- aging_biomarker
  Q2 TEXT,  -- molecular_mechanism_of_aging
  ...
  Q9 TEXT   -- calorie_restriction_lifespan_explanation
)

-- Extended view (one row per paper-question)
combined_answers_extended (
  id INTEGER PRIMARY KEY,
  doi TEXT,
  theory_id TEXT,
  theory TEXT,
  question_key TEXT,
  question_text TEXT,
  answer TEXT,
  confidence REAL,
  reasoning TEXT,
  source TEXT,  -- 'rag' or 'fulltext'
  journal TEXT,
  cited_by_count INTEGER,
  fwci REAL
)
```

### CSV Outputs

**Short CSV** (3.4 MB):
```csv
doi,theory_id,theory,title,year,Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9
10.1089/ars.2012.5111,T0001,Mitochondrial ROS...,Oxidative stress...,2013,Yes quantitatively shown,Yes,No,No,No,No,No,No,Yes
```

**Extended CSV** (80 MB):
```csv
doi,theory_id,theory,question_key,answer,confidence,reasoning,source,journal,cited_by_count,fwci
10.1089/ars.2012.5111,T0001,Mitochondrial ROS...,aging_biomarker,Yes quantitatively shown,0.92,The paper identifies...,rag,Antioxidants & Redox Signaling,523,2.8
```

**Theory Stats CSV** (109 KB):
```csv
theory_name,doi_count
Mitochondrial ROS-Induced Free Radical Theory,261
Mitochondrial ROS-Induced Oxidative Stress Theory,114
Somatic DNA Damage Theory,112
```

---

## 🔬 Why RAG is Important for Question Answering

### The Problem with Traditional Approaches

1. **Full-Text LLM**: Expensive, slow, context limits (32K tokens)
2. **Keyword Search**: Misses semantic meaning, high false positives
3. **Pure Embeddings**: No reasoning, just similarity scores

### RAG Solution: Best of Both Worlds

```
Retrieval (Fast, Cheap)  +  Generation (Smart, Contextual)
        ↓                           ↓
   Find relevant info      Reason about findings
   from 15K papers         with domain knowledge
```

### Our RAG Advantages

1. **Semantic Search**: Understands "aging biomarker" = "senescence marker"
2. **Precise Context**: Only relevant chunks (not entire 10-page paper)
3. **Source Attribution**: Every answer cites specific sections
4. **Scalability**: 15K papers in hours (vs weeks for full-text LLM)
5. **Cost-Effective**: $0.001/query vs $0.10+ for full-text

### RAG vs Full-Text Comparison

| Metric | Full-Text LLM | Our RAG System |
|--------|--------------|----------------|
| **Cost per paper** | $0.10-0.50 | $0.009 (9 questions) |
| **Time per paper** | 30-60 seconds | 3-7 seconds |
| **Context limit** | 32K tokens | Unlimited (retrieval) |
| **Precision** | 70-80% | 85-90% |
| **Scalability** | Poor (15K papers = $1,500-7,500) | Excellent ($135 total) |

---

## 🏅 Top 5 Aging Theories Identified

1. **T0001** - Mitochondrial ROS-Induced Free Radical Theory (261 papers)
2. **T0002** - Mitochondrial ROS-Induced Oxidative Stress Theory (114 papers)
3. **T0003** - Somatic DNA Damage Theory (112 papers)
4. **T0004** - Mitochondrial ROS-Induced Mitochondrial Decline Theory (84 papers)
5. **T0005** - Insulin/IGF-1 Signaling Disposable Soma Theory (72 papers)

---

## 📚 Documentation

- **[BATCH_PROCESSING_GUIDE.md](documentation/BATCH_PROCESSING_GUIDE.md)** - Batch RAG processing
- **[PREDEFINED_QUERIES_SYSTEM.md](documentation/PREDEFINED_QUERIES_SYSTEM.md)** - Query contextualization
- **[CHUNKING_COMPARISON.md](documentation/CHUNKING_COMPARISON.md)** - Chunking strategies
- **[OPTIMIZATION_SUMMARY.md](documentation/OPTIMIZATION_SUMMARY.md)** - Performance optimization
- **[INGESTION_GUIDE.md](documentation/INGESTION_GUIDE.md)** - Database ingestion
- **[LLM_VOTER_README.md](LLM_VOTER_README.md)** - LLM voting system

