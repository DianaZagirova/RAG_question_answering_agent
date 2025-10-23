# RAG System Flow - Default (Enhanced) Mode

## Overview
The default system uses **LLM-Enhanced Query Transformation** to convert questions into scientific text that matches paper chunks better.

---

## Complete Flow Diagram

```
USER QUESTION
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. QUERY ENHANCEMENT (Query Preprocessor)                   │
│    Input: "Does it suggest an aging biomarker?"             │
│                                                              │
│    LLM Enhancement Prompt:                                   │
│    ┌──────────────────────────────────────────────────┐    │
│    │ System: "You are an expert at transforming       │    │
│    │          questions into scientific text."        │    │
│    │                                                   │    │
│    │ User: "Transform this question into a            │    │
│    │        declarative statement that would appear   │    │
│    │        in a scientific paper's abstract..."      │    │
│    │                                                   │    │
│    │ Question: "Does it suggest an aging biomarker?"  │    │
│    │                                                   │    │
│    │ Rules:                                            │    │
│    │ 1. Convert to declarative format                 │    │
│    │ 2. Use scientific terminology                    │    │
│    │ 3. Sound like research paper text                │    │
│    │ 4. Keep concise (1-2 sentences)                  │    │
│    │ 5. Include key scientific terms                  │    │
│    │ 6. Remove question marks                         │    │
│    └──────────────────────────────────────────────────┘    │
│                                                              │
│    LLM Output (Enhanced Query):                             │
│    "This study identifies and validates aging biomarkers    │
│     associated with mortality and age-related               │
│     physiological decline."                                 │
│                                                              │
│    ✓ Cached for reuse across all papers                     │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. VECTOR SEARCH (ChromaDB)                                 │
│    Query: Enhanced scientific text                          │
│    Filter: DOI (if specified)                               │
│    Returns: Top 10 most relevant chunks                     │
│                                                              │
│    Example Retrieved Chunk:                                 │
│    "Aging biomarkers such as telomere length and            │
│     inflammatory markers (IL-6, CRP) are associated         │
│     with increased mortality risk in longitudinal           │
│     studies [Source 3]."                                    │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. CONTEXT FORMATTING                                        │
│    Combines retrieved chunks with metadata:                  │
│                                                              │
│    [Source 1] (DOI: 10.xxx, Section: Results)              │
│    <chunk text>                                             │
│                                                              │
│    [Source 2] (DOI: 10.xxx, Section: Discussion)           │
│    <chunk text>                                             │
│    ...                                                      │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. LLM ANSWER GENERATION                                     │
│                                                              │
│    System Prompt:                                            │
│    ┌──────────────────────────────────────────────────┐    │
│    │ "Your task is to answer questions based on       │    │
│    │  provided context from scientific papers.        │    │
│    │                                                   │    │
│    │  INSTRUCTIONS:                                    │    │
│    │  1. Think step by step                           │    │
│    │  2. Answer based on what is stated in paper      │    │
│    │  3. Answer MUST be one of provided options"      │    │
│    └──────────────────────────────────────────────────┘    │
│                                                              │
│    User Prompt:                                              │
│    ┌──────────────────────────────────────────────────┐    │
│    │ Question: Does it suggest an aging biomarker?    │    │
│    │                                                   │    │
│    │ Context from scientific papers:                  │    │
│    │ [Source 1] (DOI: ..., Section: Results)          │    │
│    │ <retrieved chunk 1>                              │    │
│    │ [Source 2] (DOI: ..., Section: Discussion)       │    │
│    │ <retrieved chunk 2>                              │    │
│    │ ...                                               │    │
│    │                                                   │    │
│    │ INSTRUCTIONS:                                     │    │
│    │ You MUST respond with valid JSON:                │    │
│    │ {                                                 │    │
│    │   "answer": "<one of the options below>",        │    │
│    │   "confidence": <0.0 to 1.0>,                    │    │
│    │   "reasoning": "<cite [Source N]>"               │    │
│    │ }                                                 │    │
│    │                                                   │    │
│    │ Available answer options:                        │    │
│    │   - Yes, quantitatively shown                    │    │
│    │   - Yes, but not shown                           │    │
│    │   - No                                            │    │
│    │                                                   │    │
│    │ Requirements:                                     │    │
│    │ - "answer" must be exactly one option            │    │
│    │ - "confidence" 0.0 to 1.0                        │    │
│    │ - "reasoning" cite [Source N]                    │    │
│    │ - Return ONLY JSON, no extra text                │    │
│    └──────────────────────────────────────────────────┘    │
│                                                              │
│    LLM Response:                                             │
│    {                                                         │
│      "answer": "Yes, but not shown",                        │
│      "confidence": 0.85,                                    │
│      "reasoning": "The paper discusses aging biomarkers     │
│                    such as telomere length [Source 3] but   │
│                    does not provide quantitative evidence." │
│    }                                                         │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. RESPONSE PARSING & VALIDATION                             │
│    - Extract JSON from response                             │
│    - Validate answer is one of allowed options              │
│    - Validate confidence is 0.0-1.0                         │
│    - If parse fails: return error (no fallback)             │
└─────────────────────────────────────────────────────────────┘
    ↓
FINAL OUTPUT
```

---

## Key Components

### 1. Query Enhancement (Step 1)
**Purpose**: Transform user question into scientific text that matches paper chunks

**Input**: 
```
"Does it suggest an aging biomarker?"
```

**LLM Call**:
- **Temperature**: 0.3 (deterministic)
- **Max Tokens**: 150
- **Cached**: Yes (reused across all papers)

**Output**:
```
"This study identifies and validates aging biomarkers associated 
with mortality and age-related physiological decline."
```

**Why this works**: Paper chunks are written in declarative scientific style, not as questions. Matching the style improves retrieval.

---

### 2. Vector Search (Step 2)
**Purpose**: Find most relevant paper chunks

**Query**: Enhanced scientific text (from Step 1)
**Filter**: DOI (if specified)
**Embedding Model**: `sentence-transformers/all-mpnet-base-v2`
**Results**: Top 10 chunks by cosine similarity

**Example Match**:
```
Query: "This study identifies aging biomarkers..."
Chunk: "Aging biomarkers such as telomere length and 
        inflammatory markers are associated with mortality..."
Similarity: 0.85
```

---

### 3. Context Formatting (Step 3)
**Purpose**: Prepare retrieved chunks for LLM

**Format**:
```
[Source 1] (DOI: 10.1089/ars.2012.5111, Section: Results)
Telomere length and inflammatory markers (IL-6, CRP) were 
measured in 500 participants...

[Source 2] (DOI: 10.1089/ars.2012.5111, Section: Discussion)
Our findings suggest that these biomarkers are associated 
with increased mortality risk...
```

---

### 4. Answer Generation (Step 4)
**Purpose**: Generate structured answer with evidence

**LLM Model**: GPT-4.1
**Temperature**: 0.2 (mostly deterministic)
**Max Tokens**: 300

**Prompt Structure**:
1. **System Prompt**: Instructions for answering
2. **User Prompt**: 
   - Original question (NOT enhanced)
   - Retrieved context with sources
   - JSON format requirements
   - Answer options
   - Validation rules

**Output Format**:
```json
{
  "answer": "Yes, but not shown",
  "confidence": 0.85,
  "reasoning": "The paper discusses biomarkers [Source 3] but 
                does not provide quantitative evidence."
}
```

---

### 5. Response Parsing (Step 5)
**Purpose**: Extract and validate JSON response

**Steps**:
1. Remove markdown code blocks if present
2. Extract JSON object
3. Validate structure
4. Validate answer matches allowed options
5. Validate confidence is 0.0-1.0
6. If any validation fails: return `parse_error: True`

**No Fallback**: If JSON parsing fails, the system returns an error rather than guessing.

---

## Important Notes

### What Gets Enhanced:
- ✅ **Retrieval query** (Step 1-2): Enhanced to scientific text
- ❌ **Question to LLM** (Step 4): Original question used

### Why Original Question for LLM?
The LLM sees the **original user question** (not the enhanced version) because:
1. LLM understands questions naturally
2. Enhanced version is optimized for vector search, not LLM comprehension
3. Clearer for the LLM to understand what user is asking

### Caching:
- **Enhanced queries**: Cached and reused across all papers
- **LLM responses**: Not cached (paper-specific)

### No DOI in Query:
- DOI is used as a **filter** in vector search
- DOI is **NOT** added to the query text
- This keeps queries clean and focused on semantic content

---

## Example End-to-End

**User Input**:
```bash
python scripts/rag_answer.py \
  --question "Does it suggest an aging biomarker?" \
  --doi "10.1089/ars.2012.5111"
```

**Step 1 - Query Enhancement**:
```
Original: "Does it suggest an aging biomarker?"
Enhanced: "This study identifies and validates aging biomarkers 
           associated with mortality and age-related decline."
```

**Step 2 - Vector Search**:
```
Query: <enhanced text embedding>
Filter: doi = "10.1089/ars.2012.5111"
Results: 10 chunks (ranked by similarity)
```

**Step 3 - Context**:
```
[Source 1] Telomere length measured...
[Source 2] Inflammatory markers associated...
...
```

**Step 4 - LLM Generation**:
```
Question: "Does it suggest an aging biomarker?"
Context: [Sources 1-10]
Options: Yes, quantitatively shown / Yes, but not shown / No
→ LLM generates JSON response
```

**Step 5 - Output**:
```json
{
  "answer": "Yes, but not shown",
  "confidence": 0.85,
  "reasoning": "..."
}
```

---

## Performance Characteristics

- **Speed**: Fast (1 query enhancement + 1 vector search + 1 LLM call)
- **Accuracy**: 68.89% on validation set
- **Cost**: ~4000 tokens per question
- **Caching**: Enhanced queries cached, saves LLM calls

---

## Configuration

**Default Settings** (in `scripts/rag_answer.py`):
```python
complete_rag = CompleteRAGSystem(
    rag_system=rag,
    llm_client=llm_client,
    default_n_results=10,
    use_multi_query=False  # Default: single enhanced query
)
```

**To Use Multi-Query** (not recommended based on evaluation):
```bash
python scripts/rag_answer.py --use-multi-query
```
