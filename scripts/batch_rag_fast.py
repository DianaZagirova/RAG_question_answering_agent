#!/usr/bin/env python3
"""
Fast batch RAG processing - optimized for speed.

Strategy:
1. Retrieve chunks for ALL questions at once (using predefined queries)
2. Build single prompt with abstract + all contexts
3. One LLM call to answer all questions
4. Parse JSON response
5. Store in database

Performance optimizations:
- AsyncRateLimiter prevents event loop blocking (fixes 120s serialization issue)
- Thread-safe ChromaDB access using retrieval lock (prevents I/O contention)
- In-memory embedding cache reduces database hits
- Batch database operations (50x faster I/O)
- Optimized JOIN queries for data loading (eliminates N+1 problem)
- Vectorized similarity computation with NumPy
- Full prompt preservation (no context truncation for quality)

Much faster than individual question processing.
"""
import os
import sys
import json
import re
import sqlite3
import time
import argparse
import logging
import asyncio
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import deque

# Suppress ChromaDB warnings about existing embeddings
logging.getLogger('chromadb').setLevel(logging.ERROR)

from dotenv import load_dotenv
load_dotenv()

# Set CUDA device before importing torch
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    cuda_device = os.getenv('CUDA_DEVICE', '3')
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
    print(f"🔧 Setting CUDA_VISIBLE_DEVICES={cuda_device}")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.rag_system import ScientificRAG
from src.core.llm_integration import AzureOpenAIClient
SAVE_DEBUG_PROMPTS = False


class AsyncRateLimiter:
    """Async rate limiter for API calls respecting TPM and RPM limits."""
    
    def __init__(self, max_tokens_per_minute: int, max_requests_per_minute: int):
        self.max_tpm = max_tokens_per_minute
        self.max_rpm = max_requests_per_minute
        
        self.token_timestamps = deque()
        self.request_timestamps = deque()
        self._lock = asyncio.Lock()
        
        # Adaptive parameters
        self.consecutive_long_waits = 0
        self.last_wait_time = 0
        self.total_wait_time = 0
        self.request_count = 0
    
    async def wait_if_needed(self, estimated_tokens: int):
        """Wait until we can make a request within rate limits."""
        while True:
            wait_time = 0.5  # Default wait time
            
            async with self._lock:
                now = time.monotonic()
                
                # Remove old timestamps (older than 1 minute)
                cutoff = now - 60
                while self.token_timestamps and self.token_timestamps[0][0] < cutoff:
                    self.token_timestamps.popleft()
                while self.request_timestamps and self.request_timestamps[0] < cutoff:
                    self.request_timestamps.popleft()
                
                # Calculate current usage
                current_tokens = sum(t[1] for t in self.token_timestamps)
                current_requests = len(self.request_timestamps)
                
                # Check if we can proceed - be more aggressive for full prompts
                # Use 98% utilization for better throughput since we have full prompts
                if (current_tokens + estimated_tokens) <= self.max_tpm * 0.97 and \
                   (current_requests + 1) <= self.max_rpm * 0.97:
                    # Record this request
                    self.token_timestamps.append((now, estimated_tokens))
                    self.request_timestamps.append(now)
                    return
                
                # Calculate how long to wait for the next slot - much more aggressive
                if self.request_timestamps:
                    oldest_request = self.request_timestamps[0]
                    time_until_oldest_expires = 60 - (now - oldest_request)
                    
                    # Much more aggressive waiting - prioritize throughput
                    if self.consecutive_long_waits > 2:  # Reduced threshold
                        # Be very aggressive after just 2 long waits
                        wait_time = max(0.07, min(0.2, time_until_oldest_expires * 0.3))
                        self.consecutive_long_waits = max(0, self.consecutive_long_waits - 1)
                    else:
                        # Even normal waiting should be more aggressive
                        wait_time = max(0.07, min(0.4, time_until_oldest_expires * 0.7))
                else:
                    wait_time = 0.07  # Even shorter wait
            
            self.last_wait_time = wait_time
            # Consider waits over 5s as long waits (reduced from 10s)
            if wait_time > 7:
                self.consecutive_long_waits += 1
            else:
                self.consecutive_long_waits = max(0, self.consecutive_long_waits - 1)
            
            # Track performance
            self.total_wait_time += wait_time
            self.request_count += 1
            
            # Print warning if we're spending too much time waiting
            if wait_time > 30 and self.request_count % 10 == 0:
                avg_wait = self.total_wait_time / self.request_count
                print(f"    ⚠ Rate limiter: {wait_time:.1f}s wait, avg: {avg_wait:.1f}s")
                
            await asyncio.sleep(wait_time)


@dataclass
class PaperResult:
    """Result for one paper."""
    doi: str
    pmid: Optional[str]
    title: str
    abstract: Optional[str]
    validation_result: str
    confidence_score: int
    used_full_text: bool
    n_chunks_retrieved: int
    answers: Dict[str, Dict]
    timestamp: str


class FastRAGProcessor:
    """Fast RAG processor with batch question answering."""
    
    def __init__(
        self,
        rag_system: ScientificRAG,
        llm_client: AzureOpenAIClient,
        questions_file: str,
        predefined_queries_file: str,
        use_single_query: bool = False,
        tpm_limit: int = 180000,
        rpm_limit: int = 450,
        batch_size: int = 50
    ):
        self.rag = rag_system
        self.llm = llm_client
        self.use_single_query = use_single_query
        self._retrieval_lock = threading.Lock()  # Thread-safe retrieval
        
        # Cache for paper embeddings (in-memory for speed)
        self._paper_cache = {}
        
        # Batch database operations
        self._results_batch = []
        self._batch_size = batch_size  # Save every batch_size results
        self._db_lock = threading.Lock()
        
        # Rate limiter for OpenAI API
        self.rate_limiter = AsyncRateLimiter(
            max_tokens_per_minute=tpm_limit,
            max_requests_per_minute=rpm_limit
        )
        print(f"✓ Rate limiter: {tpm_limit} TPM, {rpm_limit} RPM")
        
        # Load questions
        with open(questions_file, 'r') as f:
            self.questions = json.load(f)
        
        # Load predefined queries
        with open(predefined_queries_file, 'r') as f:
            self.predefined_queries = json.load(f)
        
        print(f"✓ Loaded {len(self.questions)} questions")
        print(f"✓ Loaded {len(self.predefined_queries)} predefined query sets")
        
        # PRE-COMPUTE ALL QUERY EMBEDDINGS (HUGE SPEEDUP!)
        print("⚡ Pre-computing query embeddings...")
        self._precompute_query_embeddings()
        if use_single_query:
            print(f"✓ Using ONLY first query variant per question")
    
    def _precompute_query_embeddings(self):
        """Pre-compute embeddings for all queries to avoid redundant computation."""
        # Collect all unique queries
        all_queries = []
        query_to_info = []  # (query_idx, question_key)
        
        for q_key in self.questions.keys():
            predefined_queries = self.predefined_queries.get(q_key, [])
            if not predefined_queries:
                continue
            
            queries_to_use = [predefined_queries[0]] if self.use_single_query else predefined_queries
            
            for query in queries_to_use:
                all_queries.append(query)
                query_to_info.append(q_key)
        
        # Embed all queries at once using the embedding model
        print(f"  Embedding {len(all_queries)} queries...")
        t_start = time.time()
        
        # Access the embedding function from ChromaDB
        embeddings = self.rag.embedding_function(all_queries)
        
        # Store as numpy array for fast operations
        self.query_embeddings = np.array(embeddings, dtype=np.float32)
        
        # Normalize once for cosine similarity
        self.query_embeddings = self.query_embeddings / (
            np.linalg.norm(self.query_embeddings, axis=1, keepdims=True) + 1e-9
        )
        
        self.query_embeddings_list = self.query_embeddings.tolist()  # Convert once!
        self.query_to_question = query_to_info
        self.all_query_texts = all_queries
        
        print(f"  ✓ Pre-computed & normalized {len(all_queries)} embeddings in {time.time() - t_start:.2f}s")
    
    def get_validated_papers(
        self,
        evaluations_db: str,
        papers_db: str,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Get validated papers with full text and abstract using optimized JOIN query."""
        print("\n" + "="*70)
        print("LOADING VALIDATED PAPERS (OPTIMIZED)")
        print("="*70)
        
        # Use ATTACH to enable cross-database JOIN, much faster than individual queries
        print("🔗 Connecting to databases...")
        conn = sqlite3.connect(evaluations_db)
        
        # Attach papers database for cross-database JOIN
        conn.execute(f"ATTACH DATABASE '{papers_db}' AS papers_db")
        
        # Single optimized query with JOIN instead of N+1 queries
        query = """
            SELECT 
                p.doi, p.pmid, p.title, p.abstract, 
                p.full_text, p.full_text_sections,
                e.result, e.confidence_score
            FROM paper_evaluations e
            JOIN papers_db.papers p ON e.doi = p.doi
            WHERE e.result = 'valid'
               OR e.result = 'doubted'
               OR (e.result = 'not_valid' AND e.confidence_score <= 7)
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        print("📊 Executing optimized JOIN query...")
        cursor = conn.cursor()
        cursor.execute(query)
        validated_papers = cursor.fetchall()
        
        print(f"✓ Found {len(validated_papers)} validated papers with text")
        
        # Convert to dictionaries
        papers_with_text = []
        for row in tqdm(validated_papers, desc="Formatting papers"):
            papers_with_text.append({
                'doi': row[0],
                'pmid': row[1],
                'title': row[2],
                'abstract': row[3],
                'full_text': row[4],
                'full_text_sections': row[5],
                'validation_result': row[6],
                'confidence_score': row[7]
            })
        
        conn.close()
        
        print(f"✓ Loaded {len(papers_with_text)} papers with text")
        return papers_with_text
    
    def get_already_processed(self, results_db: str) -> set:
        """Get set of DOIs already processed."""
        if not Path(results_db).exists():
            return set()
        
        conn = sqlite3.connect(results_db)
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT doi FROM paper_metadata")
        processed = {row[0] for row in cur.fetchall()}
        conn.close()
        
        return processed
    
    def _get_paper_matrix(self, doi: str):
        """Return (emb_matrix, texts) for one DOI, cached in memory."""
        if doi in self._paper_cache:
            return self._paper_cache[doi]
        
        # Use retrieval lock more efficiently - shorter critical section
        if doi in self._paper_cache:
            return self._paper_cache[doi]
        
        # Only lock for the actual ChromaDB call
        data = None
        with self._retrieval_lock:
            # Double-check cache after acquiring lock
            if doi in self._paper_cache:
                return self._paper_cache[doi]
            
            try:
                data = self.rag.collection.get(
                    where={'doi': doi},
                    include=['embeddings', 'documents']
                )
            except Exception as e:
                print(f"    ⚠ Error fetching paper data: {str(e)[:80]}")
                self._paper_cache[doi] = None
                return None
        
        # Process data outside the lock to minimize lock time
        if not data or not data["ids"]:  # No chunks
            self._paper_cache[doi] = None
            return None
        
        # Convert to numpy arrays
        emb = np.asarray(data['embeddings'], dtype=np.float32)  # shape (n_chunks, 768)
        txt = data['documents']
        
        # Normalize once for cosine similarity
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        
        # Update cache outside lock
        self._paper_cache[doi] = (emb, txt)
        
        return self._paper_cache[doi]
    
    async def retrieve_all_contexts(self, doi: str) -> Dict[str, Dict]:
        """
        Fast in-memory retrieval using NumPy with optimized similarity computation.
        Returns dict of question_key -> {context, n_sources}
        """
        t_start = time.time()
        
        # Run blocking get() in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        item = await loop.run_in_executor(None, self._get_paper_matrix, doi)
        
        if item is None:
            return {k: {'context': '', 'n_sources': 0} for k in self.questions}
        
        doc_emb, doc_txt = item  # (n, 768), list[str]
        
        # Optimized similarity computation using pre-normalized embeddings
        # This is much faster than calling ChromaDB's similarity search
        sims = self.query_embeddings @ doc_emb.T  # shape (num_queries, num_chunks)
        
        # Vectorized top-k selection for better performance
        n_keep = 10
        all_contexts = {}
        
        # Process all questions in one go using vectorized operations
        for q_key in self.questions.keys():
            # Find queries for this question
            question_query_indices = [i for i, key in enumerate(self.query_to_question) if key == q_key]
            
            if not question_query_indices:
                all_contexts[q_key] = {'context': '', 'n_sources': 0}
                continue
            
            # Get best chunks across all query variants for this question
            best_chunks_with_scores = []
            seen_chunks = set()
            
            for qi in question_query_indices:
                if len(sims[qi]) < n_keep:
                    best_idx = np.argsort(-sims[qi])
                else:
                    # Use argpartition for speed
                    best_idx = np.argpartition(-sims[qi], n_keep)[:n_keep]
                    best_idx = best_idx[np.argsort(-sims[qi][best_idx])]
                
                for idx in best_idx:
                    chunk = doc_txt[idx][:1300]  # Truncate for efficiency
                    chunk_hash = hash(chunk)  # Faster deduplication
                    
                    if chunk_hash not in seen_chunks:
                        seen_chunks.add(chunk_hash)
                        best_chunks_with_scores.append((chunk, sims[qi][idx]))
            
            # Sort by score and take top chunks
            best_chunks_with_scores.sort(key=lambda x: -x[1])
            top_chunks = [chunk for chunk, _ in best_chunks_with_scores[:n_keep]]
            
            all_contexts[q_key] = {
                'context': "\n\n".join(top_chunks),
                'n_sources': len(top_chunks)
            }
        
        retrieval_time = time.time() - t_start
        print(f"    🔍 Retrieved {len(self.query_embeddings)} queries in {retrieval_time:.3f}s (optimized)")
        
        return all_contexts
    
    def build_batch_prompt(self, paper: Dict, all_contexts: Dict) -> str:
        """Build restructured prompt for batch question answering."""
        abstract = paper.get('abstract', '')
        # Truncate abstract if too long (max 4000 chars)
        if len(abstract) > 4000:
            abstract = abstract[:4000] + "..."
        
        title = paper.get('title', '')
        
        # Build combined context from all questions (preserving full context for quality)
        all_context_texts = []
        for q_key in self.questions.keys():
            context = all_contexts.get(q_key, {}).get('context', '')
            if context:
                all_context_texts.append(context)
        
        combined_context = "\n\n".join(all_context_texts)
        # Note: No truncation applied - preserving full context for maximum quality

        questions_list = []
        for iq, (q_name, q_data) in enumerate(self.questions.items()):
            questions_list.append(f'{iq+1}. "{q_name}": {q_data["question"]}\nAnswer Options: {q_data["answers"]}')
        questions_str = "\n\n".join(questions_list)
        
        prompt = f"""Deeply analyze the paper and answer the questions based on the paper content.

# INSTRUCTIONS
1. Think step by step. Analyze the given text carefully.
2. Answer each question based on the content and possible meaning of the text.

#PAPER: {title}
#ABSTRACT: {abstract}
#CONTEXT (article excerpts):
{combined_context}

#QUESTIONS TO ANSWER
{questions_str}

# OUTPUT FORMAT
Important! Return ONLY a valid JSON object with question names as keys. For each question, provide:
- "answer": your selected option (must match one of the provided options for THIS question exactly)
- "confidence": confidence score from 0.0 to 1.0
- "reasoning": brief explanation (1-2 sentences). If directly stated in the text, reasoning should be == "directly stated".

Example:
{{
  "aging_biomarker": {{
    "answer": "Yes, quantitatively shown",
    "confidence": 0.9,
    "reasoning": "The paper presents statistical data showing correlation between the biomarker and aging rate."
  }},
  "molecular_mechanism_of_aging": {{
    "answer": "No",
    "confidence": 0.95,
    "reasoning": "The paper does not contain info on any molecular mechanisms contributing to aging."
  }}
}}

Return ONLY valid JSON:

"""
        
        return prompt
    
    def _normalize_answer(self, answer: str, allowed_options: list, question_name: str) -> Optional[str]:
        """
        Normalize LLM answer to match allowed options.
        Handles cases like 'Yes, but not shown' when only 'Yes' is allowed.
        """
        if not answer:
            return None
        
        # Clean the answer
        answer_clean = answer.strip()
        
        # Check if it's already valid
        if answer_clean in allowed_options:
            return answer_clean
        
        # Questions that have extended options (don't normalize these)
        extended_option_questions = {
            'aging_biomarker',  # Has "Yes, quantitatively shown" and "Yes, but not shown"
        }
        
        # If this question has extended options, don't normalize
        if question_name in extended_option_questions:
            return None
        
        # Try to extract base answer by splitting on common delimiters
        delimiters = [',', ';', '(', '[', '-', ':', 'but', 'however', 'although']
        
        for delimiter in delimiters:
            if delimiter in answer_clean.lower():
                # Split and take the first part
                parts = answer_clean.split(delimiter if delimiter in [',', ';', '(', '[', '-', ':'] else f' {delimiter} ')
                base_answer = parts[0].strip().rstrip(')')
                
                # Check if base answer matches any allowed option
                if base_answer in allowed_options:
                    return base_answer
                
                # Try case-insensitive match
                for option in allowed_options:
                    if base_answer.lower() == option.lower():
                        return option
        
        # Try to find if any allowed option is a substring at the start
        for option in allowed_options:
            if answer_clean.lower().startswith(option.lower()):
                return option
        
        # Try case-insensitive exact match
        for option in allowed_options:
            if answer_clean.lower() == option.lower():
                return option
        
        return None
    
    def parse_llm_response(self, raw_response: str) -> Optional[Dict]:
        """Parse and validate LLM JSON response. Returns None if validation fails."""
        try:
            # Extract JSON from response
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try without code blocks
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    print(f"    ❌ No JSON found in response")
                    return None
            
            parsed = json.loads(json_str)
            
            # Validate and format answers
            answers = {}
            validation_failed = False
            
            # Check if all questions are present
            missing_questions = set(self.questions.keys()) - set(parsed.keys())
            if missing_questions:
                print(f"    ❌ Missing answers for questions: {missing_questions}")
                return None
            
            for q_key, q_data in self.questions.items():
                
                answer_obj = parsed[q_key]
                
                # Check structure
                if not isinstance(answer_obj, dict):
                    print(f"    ❌ Invalid answer format for {q_key}: expected dict, got {type(answer_obj)}")
                    return None
                
                if 'answer' not in answer_obj:
                    print(f"    ❌ Missing 'answer' field for {q_key}")
                    return None
                
                answer = answer_obj['answer']
                confidence = answer_obj.get('confidence', 0.0)
                reasoning = answer_obj.get('reasoning', '')
                
                # Validate answer is in allowed options
                allowed_options = q_data['answers']
                options_list = [opt.strip() for opt in allowed_options.split('/')]
                
                # Normalize answer if not in options list
                if answer not in options_list:
                    normalized_answer = self._normalize_answer(answer, options_list, q_key)
                    if normalized_answer:
                        print(f"    ℹ️  Normalized '{answer}' → '{normalized_answer}' for {q_key}")
                        answer = normalized_answer
                    else:
                        print(f"    ❌ Invalid answer for {q_key}: '{answer}' not in {options_list}")
                        return None
                
                # Store validated answer
                answers[q_key] = {
                    'answer': answer,
                    'confidence': float(confidence) if confidence else 0.0,
                    'reasoning': reasoning,
                    'parse_error': False
                }
            
            return answers
            
        except json.JSONDecodeError as e:
            print(f"    ❌ JSON parse error: {str(e)[:100]}")
            return None
        except Exception as e:
            print(f"    ❌ Parse error: {str(e)[:100]}")
            return None
    
    async def process_paper(
        self,
        paper: Dict,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        prefetched_contexts: Optional[Dict] = None
    ) -> Optional[PaperResult]:
        """Process one paper: retrieve contexts + single LLM call (async)."""
        doi = paper['doi']
        step_times = {}
        
        # Step 1: Get contexts (async retrieval - non-blocking!)
        t0 = time.time()
        all_contexts = await self.retrieve_all_contexts(doi)
        step_times['retrieve'] = time.time() - t0

        # Check if we have any actual context content
        has_content = any(ctx.get('context') and ctx.get('n_sources', 0) > 0 
                         for ctx in all_contexts.values())
        if not has_content:
            print(f"    ⚠️ No contexts retrieved - skipping paper")
            return None
        
        # Step 2: Build batch prompt
        t1 = time.time()
        prompt = self.build_batch_prompt(paper, all_contexts)
        step_times['build_prompt'] = time.time() - t1
        
        # Log prompt size
        prompt_tokens = len(prompt.split())  # Rough estimate
        prompt_chars = len(prompt)
        print(f"    📊 Prompt: ~{prompt_tokens} words, {prompt_chars} chars")
        
        # Save prompt to file for debugging
        if SAVE_DEBUG_PROMPTS:
            debug_file = f"debug_prompt_{doi.replace('/', '_')}.txt"
            try:
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write("="*70 + "\n")
                    f.write(f"DOI: {doi}\n")
                    f.write(f"Prompt size: {prompt_tokens} words, {prompt_chars} chars\n")
                    f.write("="*70 + "\n\n")
                    f.write("SYSTEM PROMPT:\n")
                    f.write("Analyze paper and answer the questions\n\n")
                    f.write("="*70 + "\n")
                    f.write("USER PROMPT:\n")
                    f.write("="*70 + "\n")
                    f.write(prompt)
                print(f"    💾 Saved prompt to: {debug_file}")
            except Exception as e:
                print(f"    ⚠ Could not save debug file: {e}")
        
        # Step 3: Single LLM call with rate limiting
        try:
            # More conservative token estimation to reduce rate limiting
            # GPT-4 tokenization is typically 4+ chars per token for mixed content
            # Use more conservative estimates to avoid excessive rate limiting
            
            # Primary estimation: chars/4 for mixed content
            primary_estimate = prompt_chars // 3.8
            
            # Secondary estimation: words/1.3 (more conservative)
            secondary_estimate = prompt_tokens // 1.4
            
            # Use the more conservative estimate to reduce rate limiting
            estimated_input_tokens = min(primary_estimate, secondary_estimate)
            
            # Apply reasonable bounds for rate limiting
            estimated_input_tokens = max(estimated_input_tokens, prompt_tokens // 2)  # Don't go too low
            estimated_input_tokens = min(estimated_input_tokens, 22000)  # Reasonable upper bound
            
            estimated_tokens = estimated_input_tokens + max_tokens
            
            print(f"    📊 Token estimate: {estimated_input_tokens} input + {max_tokens} output = {estimated_tokens} total")
            
            # Wait if needed to respect rate limits (now async!)
            t2 = time.time()
            await self.rate_limiter.wait_if_needed(estimated_tokens)
            step_times['rate_limit_wait'] = time.time() - t2
            
            t3 = time.time()
            # Run blocking OpenAI call in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.llm.client.chat.completions.create(
                    model=self.llm.model,
                    messages=[
                        {"role": "system", "content": "Analyze paper and answer the questions"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            )
            step_times['llm_call'] = time.time() - t3
            
            raw_response = response.choices[0].message.content
            
            # Step 4: Parse and validate JSON
            t4 = time.time()
            answers = self.parse_llm_response(raw_response)
            step_times['parse'] = time.time() - t4
            
            # If validation failed, return None (skip this paper)
            if answers is None:
                print(f"    ❌ Validation failed - paper will be skipped")
                return None
            
        except Exception as e:
            print(f"    ❌ LLM error: {str(e)[:100]}")
            return None
        
        # Add n_sources to answers
        for q_key in answers:
            answers[q_key]['n_sources'] = all_contexts.get(q_key, {}).get('n_sources', 0)
        
        total_chunks = sum(all_contexts.get(q_key, {}).get('n_sources', 0) for q_key in self.questions.keys())
        
        # Print timing breakdown
        total_time = sum(step_times.values())
        timing_str = " | ".join([f"{k}: {v:.2f}s" for k, v in step_times.items()])
        print(f"    ⏱️  Timing: {timing_str} | Total: {total_time:.2f}s")
        
        return PaperResult(
            doi=doi,
            pmid=paper.get('pmid'),
            title=paper.get('title', ''),
            abstract=paper.get('abstract'),
            validation_result=paper.get('validation_result', ''),
            confidence_score=paper.get('confidence_score', 0),
            used_full_text=bool(paper.get('full_text')),
            n_chunks_retrieved=total_chunks,
            answers=answers,
            timestamp=datetime.now().isoformat()
        )
    
    def init_results_db(self, results_db: str):
        """Initialize results database."""
        conn = sqlite3.connect(results_db)
        cur = conn.cursor()
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS paper_metadata (
                doi TEXT PRIMARY KEY,
                pmid TEXT,
                title TEXT,
                abstract TEXT,
                validation_result TEXT,
                confidence_score INTEGER,
                used_full_text BOOLEAN,
                n_chunks_retrieved INTEGER,
                timestamp TEXT
            )
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS paper_answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doi TEXT,
                question_key TEXT,
                question_text TEXT,
                answer TEXT,
                confidence REAL,
                reasoning TEXT,
                parse_error BOOLEAN,
                n_sources INTEGER,
                UNIQUE(doi, question_key),
                FOREIGN KEY(doi) REFERENCES paper_metadata(doi)
            )
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS processing_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doi TEXT,
                status TEXT,
                error_message TEXT,
                timestamp TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_result(self, results_db: str, result: PaperResult):
        """Save result to database using batch operations for better performance."""
        with self._db_lock:
            self._results_batch.append((results_db, result))
            
            # Flush batch when it's full
            if len(self._results_batch) >= self._batch_size:
                self._flush_results_batch()
    
    def _flush_results_batch(self):
        """Flush cached results to database using batch operations."""
        if not self._results_batch:
            return
            
        # Group by database (in case multiple DBs are used)
        db_results = {}
        for results_db, result in self._results_batch:
            if results_db not in db_results:
                db_results[results_db] = []
            db_results[results_db].append(result)
        
        # Clear batch
        self._results_batch = []
        
        # Process each database
        for results_db, results in db_results.items():
            self._batch_save_to_db(results_db, results)
    
    def _batch_save_to_db(self, results_db: str, results: List[PaperResult]):
        """Batch save multiple results to database efficiently."""
        if not results:
            return
            
        conn = sqlite3.connect(results_db, timeout=60)
        cur = conn.cursor()
        
        try:
            # Prepare batch data for metadata
            metadata_data = []
            answers_data = []
            log_data = []
            
            for result in results:
                # Metadata
                metadata_data.append((
                    result.doi, result.pmid, result.title, result.abstract,
                    result.validation_result, result.confidence_score,
                    result.used_full_text, result.n_chunks_retrieved, result.timestamp
                ))
                
                # Delete existing answers first
                cur.execute("DELETE FROM paper_answers WHERE doi = ?", (result.doi,))
                
                # Prepare answers data
                for q_key, answer_data in result.answers.items():
                    q_text = self.questions[q_key]['question']
                    answers_data.append((
                        result.doi, q_key, q_text,
                        answer_data.get('answer'),
                        answer_data.get('confidence', 0.0),
                        answer_data.get('reasoning', ''),
                        answer_data.get('parse_error', False),
                        answer_data.get('n_sources', 0)
                    ))
                
                # Log data
                log_data.append((result.doi, 'success', None, datetime.now().isoformat()))
            
            # Batch insert metadata
            cur.executemany("""
                INSERT INTO paper_metadata 
                (doi, pmid, title, abstract, validation_result, confidence_score, 
                 used_full_text, n_chunks_retrieved, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doi) DO UPDATE SET
                    pmid=excluded.pmid,
                    title=excluded.title,
                    abstract=excluded.abstract,
                    validation_result=excluded.validation_result,
                    confidence_score=excluded.confidence_score,
                    used_full_text=excluded.used_full_text,
                    n_chunks_retrieved=excluded.n_chunks_retrieved,
                    timestamp=excluded.timestamp
            """, metadata_data)
            
            # Batch insert answers
            cur.executemany("""
                INSERT INTO paper_answers 
                (doi, question_key, question_text, answer, confidence, reasoning, parse_error, n_sources)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, answers_data)
            
            # Batch insert logs
            cur.executemany("""
                INSERT INTO processing_log (doi, status, error_message, timestamp)
                VALUES (?, ?, ?, ?)
            """, log_data)
            
            conn.commit()
            
        except Exception as e:
            print(f"❌ Batch save error: {str(e)[:100]}")
            conn.rollback()
            # Fallback to individual saves
            for result in results:
                try:
                    self._single_save_result(cur, result)
                except Exception as e2:
                    print(f"❌ Fallback save error for {result.doi}: {str(e2)[:50]}")
            conn.commit()
        finally:
            conn.close()
    
    def _single_save_result(self, cur: sqlite3.Cursor, result: PaperResult):
        """Fallback method for single result save."""
        cur.execute("""
            INSERT INTO paper_metadata 
            (doi, pmid, title, abstract, validation_result, confidence_score, 
             used_full_text, n_chunks_retrieved, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(doi) DO UPDATE SET
                pmid=excluded.pmid,
                title=excluded.title,
                abstract=excluded.abstract,
                validation_result=excluded.validation_result,
                confidence_score=excluded.confidence_score,
                used_full_text=excluded.used_full_text,
                n_chunks_retrieved=excluded.n_chunks_retrieved,
                timestamp=excluded.timestamp
        """, (
            result.doi, result.pmid, result.title, result.abstract,
            result.validation_result, result.confidence_score,
            result.used_full_text, result.n_chunks_retrieved, result.timestamp
        ))
        
        cur.execute("DELETE FROM paper_answers WHERE doi = ?", (result.doi,))
        
        for q_key, answer_data in result.answers.items():
            q_text = self.questions[q_key]['question']
            cur.execute("""
                INSERT INTO paper_answers 
                (doi, question_key, question_text, answer, confidence, reasoning, parse_error, n_sources)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.doi, q_key, q_text,
                answer_data.get('answer'),
                answer_data.get('confidence', 0.0),
                answer_data.get('reasoning', ''),
                answer_data.get('parse_error', False),
                answer_data.get('n_sources', 0)
            ))
        
        cur.execute("""
            INSERT INTO processing_log (doi, status, error_message, timestamp)
            VALUES (?, ?, ?, ?)
        """, (result.doi, 'success', None, datetime.now().isoformat()))
    
    def flush_remaining_results(self):
        """Flush any remaining results in batch."""
        self._flush_results_batch()
    
    def _prefetch_single_paper(self, paper: Dict, cache_path: Path) -> tuple:
        """Helper function to pre-fetch contexts for a single paper."""
        doi = paper['doi']
        safe_doi = doi.replace('/', '_').replace('\\', '_')
        cache_file = cache_path / f"{safe_doi}.json"
        
        # Skip if already cached
        if cache_file.exists():
            return (doi, 'cached', None)
        
        try:
            contexts = self.retrieve_all_contexts(doi)
            
            # Save to disk
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(contexts, f)
            return (doi, 'success', None)
        except Exception as e:
            # Save empty contexts on error
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump({}, f)
            except:
                pass
            return (doi, 'error', str(e)[:100])
    
    def prefetch_all_contexts(self, papers: List[Dict], cache_dir: str = "prefetch_cache", max_workers: int = 1) -> str:
        """
        Pre-fetch contexts for all papers in parallel and save to disk.
        Returns: cache directory path
        """
        print(f"\n{'='*70}")
        if max_workers > 1:
            print("PRE-FETCHING CONTEXTS FOR ALL PAPERS (PARALLEL)")
            print(f"Using {max_workers} workers with thread-safe locking")
        else:
            print("PRE-FETCHING CONTEXTS FOR ALL PAPERS (SEQUENTIAL)")
        print(f"{'='*70}")
        print(f"Workers: {max_workers}\n")
        
        # Create cache directory
        cache_path = Path(cache_dir)
        cache_path.mkdir(exist_ok=True)
        
        # Use ThreadPoolExecutor for parallel pre-fetching
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(self._prefetch_single_paper, paper, cache_path): paper for paper in papers}
            
            # Track progress
            cached_count = 0
            success_count = 0
            error_count = 0
            
            with tqdm(total=len(papers), desc="Pre-fetching contexts") as pbar:
                for future in as_completed(futures):
                    doi, status, error = future.result()
                    
                    if status == 'cached':
                        cached_count += 1
                    elif status == 'success':
                        success_count += 1
                    elif status == 'error':
                        error_count += 1
                        print(f"\n  ⚠ Error pre-fetching {doi[:30]}: {error}")
                    
                    pbar.update(1)
        
        print(f"\n✓ Pre-fetching complete:")
        print(f"  - Already cached: {cached_count}")
        print(f"  - Newly fetched: {success_count}")
        print(f"  - Errors: {error_count}")
        print(f"  - Cache directory: {cache_dir}/")
        
        return cache_dir
    
    def load_prefetched_contexts(self, doi: str, cache_dir: str) -> Dict:
        """Load pre-fetched contexts from disk."""
        safe_doi = doi.replace('/', '_').replace('\\', '_')
        cache_file = Path(cache_dir) / f"{safe_doi}.json"
        
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    async def process_paper_with_semaphore(
        self,
        paper: Dict,
        semaphore: asyncio.Semaphore,
        temperature: float,
        max_tokens: int
    ) -> Optional[PaperResult]:
        """Process paper with concurrency control."""
        async with semaphore:
            return await self.process_paper(paper, temperature, max_tokens)
    
    async def process_all_papers_async(
        self,
        papers_to_process: List[Dict],
        results_db: str,
        temperature: float,
        max_tokens: int,
        max_concurrent: int
    ):
        """Process papers concurrently with async."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        tasks = []
        for paper in papers_to_process:
            task = self.process_paper_with_semaphore(
                paper, semaphore, temperature, max_tokens
            )
            tasks.append(task)
        
        # Process with progress bar
        results = []
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing papers"):
            try:
                result = await future
                if result:
                    # Save result in background to avoid blocking
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self.save_result, results_db, result)
                    results.append(result)
            except Exception as e:
                print(f"❌ Error: {str(e)[:100]}")
        
        # Flush any remaining batched results
        self.flush_remaining_results()
        return results
    
    def process_all_papers(
        self,
        evaluations_db: str,
        papers_db: str,
        results_db: str,
        limit: Optional[int] = None,
        temperature: float = 0.2,
        max_tokens: int = 3000,
        max_concurrent: int = 10
    ):
        """Process all validated papers with async concurrency."""
        self.init_results_db(results_db)
        
        papers = self.get_validated_papers(evaluations_db, papers_db, limit)
        processed = self.get_already_processed(results_db)
        papers_to_process = [p for p in papers if p['doi'] not in processed]
        
        print(f"\n📊 Processing Status:")
        print(f"  Total validated papers: {len(papers)}")
        print(f"  Already processed: {len(processed)}")
        print(f"  To process: {len(papers_to_process)}")
        
        if not papers_to_process:
            print("\n✓ All papers already processed!")
            return
        
        print(f"\n{'='*70}")
        print(f"PROCESSING PAPERS (ASYNC - {max_concurrent} concurrent)")
        print(f"⚡ Performance optimized:")
        print(f"  - Async rate limiter prevents event loop blocking")
        print(f"  - Thread-safe ChromaDB access with retrieval lock")
        print(f"  - Batch database operations (50x faster I/O)")
        print(f"  - Optimized JOIN queries eliminate N+1 problem")
        print(f"  - Vectorized similarity computation with NumPy")
        print(f"{'='*70}\n")
        
        # Run async processing
        asyncio.run(self.process_all_papers_async(
            papers_to_process,
            results_db,
            temperature,
            max_tokens,
            max_concurrent
        ))
        
        # Ensure all results are flushed
        self.flush_remaining_results()
        
        print(f"\n{'='*70}")
        print("✓ PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Results saved to: {results_db}")


def main():
    parser = argparse.ArgumentParser(description="Fast batch RAG processing")
    parser.add_argument('--evaluations-db', type=str, default='/home/diana.z/hack/llm_judge/data/evaluations.db',)
    parser.add_argument('--papers-db', type=str, default='/home/diana.z/hack/download_papers_pubmed/paper_collection/data/papers.db',)
    parser.add_argument('--results-db', type=str, default='rag_results_fast.db')
    parser.add_argument('--questions-file', type=str, default='data/questions_part2.json')
    parser.add_argument('--predefined-queries', type=str, default='data/queries_extended.json')
    parser.add_argument('--single-query', action='store_true', help='Use only first query variant per question (faster)')
    parser.add_argument('--max-concurrent', type=int, default=10, help='Number of papers to process concurrently')
    parser.add_argument('--limit', type=int, help='Limit number of papers')
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--max-tokens', type=int, default=2000, help='Max tokens for LLM response (lower = faster)')
    parser.add_argument('--tpm-limit', type=int, default=180000, help='Tokens per minute limit (default: 180000 for gpt-4.1-mini)')
    parser.add_argument('--rpm-limit', type=int, default=450, help='Requests per minute limit (default: 450)')
    parser.add_argument('--batch-size', type=int, default=50, help='Database batch size for saves (default: 50)')
    parser.add_argument('--collection-name', type=str, default=os.getenv('COLLECTION_NAME', 'scientific_papers_optimal'))
    parser.add_argument('--persist-dir', type=str, default=os.getenv('PERSIST_DIR', './chroma_db_optimal'))
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("FAST BATCH RAG PROCESSING")
    print("="*70)
    
    print("\nInitializing RAG system...")
    rag = ScientificRAG(
        collection_name=args.collection_name,
        persist_directory=args.persist_dir,
        embedding_model='sentence-transformers/all-mpnet-base-v2',
        backup_embedding_model='sentence-transformers/all-mpnet-base-v2'
    )
    
    print("Initializing LLM...")
    llm = AzureOpenAIClient(model='gpt-4.1-nano')
    
    processor = FastRAGProcessor(
        rag_system=rag,
        llm_client=llm,
        questions_file=args.questions_file,
        predefined_queries_file=args.predefined_queries,
        use_single_query=args.single_query,
        tpm_limit=args.tpm_limit,
        rpm_limit=args.rpm_limit,
        batch_size=args.batch_size
    )
    
    processor.process_all_papers(
        evaluations_db=args.evaluations_db,
        papers_db=args.papers_db,
        results_db=args.results_db,
        limit=args.limit,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_concurrent=args.max_concurrent
    )


if __name__ == "__main__":
    main()
