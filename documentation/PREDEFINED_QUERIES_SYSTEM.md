# Predefined Queries System - Implementation Guide

## Overview

The system now uses **predefined scientific queries** from `data/queries_extended.json` instead of LLM-generated query enhancement. This approach:
- ✅ No LLM calls for query enhancement (faster, cheaper)
- ✅ Consistent, reproducible queries
- ✅ Multiple query variants per question (better coverage)
- ✅ Retrieves top 12 chunks per query, returns unique results

---

## How It Works

### 1. Predefined Queries File

**Location**: `data/queries_extended.json`

**Format**:
```json
{
  "aging_biomarker": [
    "The findings indicate identification of a measurable aging biomarker...",
    "The results suggest the presence of an entity that fulfills criteria..."
  ],
  "molecular_mechanism_of_aging": [
    "The study proposes a specific molecular mechanism underlying...",
    "The text provides a hypothetical framework for a molecular mechanism..."
  ],
  ...
}
```

**Structure**:
- **Key**: Question identifier (e.g., `aging_biomarker`)
- **Value**: Array of 1-2 predefined query strings in scientific language

---

### 2. Query Retrieval Strategy

For each question:

1. **Load predefined queries** for the question key
2. **Retrieve top 12 chunks** from each query variant
3. **Merge and deduplicate** by chunk ID
4. **Keep best score** for duplicate chunks
5. **Sort by relevance** and return top N unique chunks

**Example**:
```
Question: "Does it suggest any molecular mechanism of aging?"
Question Key: "molecular_mechanism_of_aging"

Predefined Queries:
  Query 1: "The study proposes a specific molecular mechanism..."
    → Retrieves 12 chunks
  Query 2: "The text provides a hypothetical framework..."
    → Retrieves 12 chunks

Total: Up to 24 chunks retrieved
Unique: Deduplicated by chunk ID
Final: Top 10 unique chunks by relevance score
```

---

## System Flow

```
USER QUESTION
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. QUESTION KEY MAPPING                                      │
│    Question: "Does it suggest any molecular mechanism?"     │
│    Question Key: "molecular_mechanism_of_aging"             │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. LOAD PREDEFINED QUERIES                                   │
│    From: data/queries_extended.json                          │
│                                                              │
│    Queries for "molecular_mechanism_of_aging":              │
│    [                                                         │
│      "The study proposes a specific molecular mechanism     │
│       underlying the aging process, implicating pathways    │
│       and molecular interactions...",                       │
│                                                              │
│      "The text provides a hypothetical framework for a      │
│       molecular mechanism of aging, discussing potential    │
│       molecular pathways..."                                │
│    ]                                                         │
│                                                              │
│    ✓ No LLM call needed                                     │
│    ✓ Queries are predefined and consistent                  │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. MULTI-QUERY RETRIEVAL                                     │
│                                                              │
│    For each predefined query:                               │
│      → Vector search with ChromaDB                          │
│      → Filter by DOI (if specified)                         │
│      → Retrieve top 12 chunks                               │
│                                                              │
│    Query 1: "The study proposes..."                         │
│      Chunk IDs: [A, B, C, D, E, F, G, H, I, J, K, L]       │
│      Scores:    [0.85, 0.82, 0.80, ...]                    │
│                                                              │
│    Query 2: "The text provides..."                          │
│      Chunk IDs: [B, D, M, N, O, P, Q, R, S, T, U, V]       │
│      Scores:    [0.83, 0.81, 0.79, ...]                    │
│                                                              │
│    Merge & Deduplicate:                                     │
│      Unique IDs: [A, B, C, D, E, F, G, H, I, J, K, L,      │
│                   M, N, O, P, Q, R, S, T, U, V]            │
│      Keep best score for duplicates (B, D)                  │
│                                                              │
│    Sort by score and take top 10 unique chunks              │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. CONTEXT FORMATTING                                        │
│    [Source 1] (DOI: ..., Section: ...)                     │
│    <chunk text>                                             │
│    ...                                                      │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. LLM ANSWER GENERATION                                     │
│    Question: Original user question                         │
│    Context: Retrieved unique chunks                         │
│    Format: JSON with answer, confidence, reasoning          │
└─────────────────────────────────────────────────────────────┘
    ↓
FINAL OUTPUT
```

---

## Usage

### Default Mode (with predefined queries):
```bash
python scripts/rag_answer.py \
  --all-questions \
  --doi "10.1016/j.tree.2022.08.003" \
  --output results.json
```

**Output**:
```
✓ Loaded 9 predefined query sets
✓ Complete RAG system ready (Predefined Queries Mode)

Q1: Does it suggest an aging biomarker...
🔍 Retrieving top 12 unique chunks using 2 predefined queries...
```

### Custom Predefined Queries File:
```bash
python scripts/rag_answer.py \
  --all-questions \
  --predefined-queries data/my_custom_queries.json \
  --output results.json
```

### Disable Predefined Queries (use LLM enhancement):
```bash
python scripts/rag_answer.py \
  --all-questions \
  --predefined-queries "" \
  --output results.json
```

---

## Configuration

### Retrieval Parameters

**Per-query retrieval**: 12 chunks (hardcoded in `_query_with_predefined_variants`)
```python
per_query_results = 12  # Retrieve top 12 from each query
```

**Final unique chunks**: Specified by `--n-results` (default: varies by question type)
- Biomarker questions: 12
- Mechanism questions: 10
- Intervention questions: 8
- Species-specific questions: 15

### Question-Specific n_results

Defined in `answer_all_questions()`:
```python
if q_num == 1 or q_num == 5:
    n_results = 12  # Biomarker questions
elif q_num == 2:
    n_results = 10  # Mechanism questions
elif q_num == 3:
    n_results = 8   # Intervention questions
else:
    n_results = 15  # Species-specific questions
```

---

## Predefined Queries Design

### Query Characteristics

Each predefined query is designed to:
1. **Match scientific writing style** (declarative, formal)
2. **Cover different aspects** of the question
3. **Use domain-specific terminology**
4. **Be comprehensive** (1-2 sentences)

### Example: aging_biomarker

**Query 1** (Quantitative evidence):
```
"The findings indicate identification of a measurable aging biomarker 
that reflects the pace of biological aging or health status, with 
demonstrable associations with mortality and age-related conditions. 
The evidence presented in the text is quantitative, providing 
statistical support for the biomarker's relevance."
```

**Query 2** (Qualitative/hypothetical):
```
"The results suggest the presence of an entity that fulfills criteria 
for an aging biomarker by correlating with age-related phenotypes and 
mortality risk. The discussion primarily presents a qualitative or 
hypothetical association, as no direct quantitative analyses of the 
biomarker's predictive validity are offered."
```

**Why 2 variants?**
- Query 1 targets papers with quantitative evidence
- Query 2 targets papers with qualitative/hypothetical discussion
- Together they provide comprehensive coverage

---

## Advantages

### 1. No LLM Enhancement Overhead
- **Before**: LLM call to enhance each question
- **After**: Direct lookup in predefined queries
- **Benefit**: Faster, cheaper, no API calls for enhancement

### 2. Consistent & Reproducible
- Same queries every time
- No variation from LLM temperature
- Easier to debug and optimize

### 3. Better Coverage
- Multiple query variants per question
- Each variant retrieves 12 chunks
- Deduplication ensures unique results

### 4. Domain-Optimized
- Queries crafted by domain experts
- Tailored to specific question types
- Can be refined based on evaluation results

---

## Comparison with Previous System

| Feature | LLM Enhancement | Predefined Queries |
|---------|----------------|-------------------|
| **Query Generation** | LLM call per question | Lookup in JSON file |
| **Speed** | Slower (LLM call) | Faster (no LLM) |
| **Cost** | Higher (API calls) | Lower (no API) |
| **Consistency** | Variable | Deterministic |
| **Coverage** | Single enhanced query | Multiple variants |
| **Chunks Retrieved** | 10 from 1 query | 10 unique from 2×12 |
| **Customization** | Prompt engineering | Edit JSON file |
| **Debugging** | Check LLM output | Check JSON file |

---

## File Structure

```
data/
├── queries_extended.json          # Predefined queries (2 per question)
├── questions_part2.json           # Question definitions
└── question_synonyms.json         # Synonyms (optional, not used yet)

src/core/
├── rag_system.py                  # Added _query_with_predefined_variants()
└── llm_integration.py             # Added predefined_queries_file parameter

scripts/
└── rag_answer.py                  # Added --predefined-queries argument
```

---

## Implementation Details

### Key Methods

**1. `_query_with_predefined_variants()` in `rag_system.py`**
```python
def _query_with_predefined_variants(
    self,
    predefined_queries: List[str],
    n_results: int,
    filter_dict: Optional[Dict]
) -> Dict:
    """
    Query using predefined query variants.
    Retrieves top 12 chunks from each variant and returns unique results.
    """
    all_results = {}  # id -> (doc, metadata, distance, query_index)
    
    per_query_results = 12  # Retrieve top 12 from each query
    
    for query_idx, query_variant in enumerate(predefined_queries):
        results = self.collection.query(
            query_texts=[query_variant],
            n_results=per_query_results,
            where=filter_dict,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Keep best score for each unique document
        for doc, meta, dist, doc_id in zip(...):
            if doc_id not in all_results or dist < all_results[doc_id][2]:
                all_results[doc_id] = (doc, meta, dist, query_idx)
    
    # Sort by distance and take top n_results unique chunks
    sorted_results = sorted(all_results.items(), key=lambda x: x[1][2])[:n_results]
    
    return {
        'documents': documents,
        'metadatas': metadatas,
        'distances': distances,
        'ids': ids,
        'query_variants_used': len(predefined_queries),
        'unique_chunks': len(documents)
    }
```

**2. `CompleteRAGSystem.__init__()` in `llm_integration.py`**
```python
def __init__(
    self,
    rag_system,
    llm_client: Optional[AzureOpenAIClient] = None,
    default_n_results: int = 10,
    use_multi_query: bool = False,
    predefined_queries_file: Optional[str] = None
):
    # Load predefined queries if provided
    self.predefined_queries = {}
    if predefined_queries_file:
        with open(predefined_queries_file, 'r') as f:
            self.predefined_queries = json.load(f)
        print(f"✓ Loaded {len(self.predefined_queries)} predefined query sets")
```

**3. `answer_question()` in `llm_integration.py`**
```python
def answer_question(
    self,
    question: str,
    ...
    question_key: Optional[str] = None
) -> Dict:
    # Get predefined queries if available for this question
    predefined_queries = None
    if question_key and question_key in self.predefined_queries:
        predefined_queries = self.predefined_queries[question_key]
        print(f"🔍 Retrieving top {n_results} unique chunks using "
              f"{len(predefined_queries)} predefined queries...")
    
    rag_response = self.rag.answer_question(
        question=question,
        n_context_chunks=n_results,
        include_metadata=True,
        metadata_filter=metadata_filter,
        use_multi_query=self.use_multi_query,
        predefined_queries=predefined_queries
    )
```

---

## Next Steps

### 1. Evaluate Performance
Run evaluation to compare predefined queries vs LLM enhancement:
```bash
python scripts/evaluate_rag_strategies.py
```

### 2. Refine Queries
Based on evaluation results, refine queries in `data/queries_extended.json`

### 3. Add More Variants
Consider adding 3rd query variant for questions with low accuracy

### 4. Use Synonyms
Integrate `data/question_synonyms.json` for query expansion

---

## Summary

**Current System**:
- ✅ Uses predefined queries from JSON file
- ✅ Retrieves top 12 chunks per query variant
- ✅ Returns unique chunks (deduplicated by ID)
- ✅ No LLM calls for query enhancement
- ✅ Faster and more consistent than LLM enhancement

**To Use**:
```bash
python scripts/rag_answer.py --all-questions --doi "<DOI>"
```

The system automatically uses `data/queries_extended.json` by default!
