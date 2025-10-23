"""
LLM Integration for RAG System using Azure OpenAI.
Provides complete question-answering with context from vector database.
"""
import os
import re
import time
import json
from typing import Dict, List, Optional, Tuple
from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class AzureOpenAIClient:
    """Client for Azure OpenAI API."""
    
    def __init__(
        self,
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize Azure OpenAI client.
        
        Args:
            azure_endpoint: Azure OpenAI endpoint (from .env if not provided)
            api_key: API key (from .env if not provided)
            api_version: API version (from .env if not provided)
            model: Model name (from .env if not provided)
        """
        self.endpoint = azure_endpoint or os.getenv('AZURE_OPENAI_ENDPOINT')
        self.api_key = api_key or os.getenv('AZURE_OPENAI_API_KEY')
        self.api_version = api_version or os.getenv('AZURE_OPENAI_API_VERSION')
        self.model = model or os.getenv('OPENAI_MODEL', 'gpt-4.1-mini')
        self.mode =  os.getenv('USE_CLIENT', 'openai')
        self.api_key_openai = os.getenv('OPENAI_API_KEY')
        if not self.endpoint or not self.api_key or not self.api_key_openai:
            raise ValueError(
                "Azure OpenAI credentials not found. "
                "Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env file."
            )
        
        if self.mode == 'azure':
            print(' Azure OpenAI client initialized')
            self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        
        )
        elif self.mode == 'openai':
            print(' OpenAI client initialized')
            self.client = OpenAI(
                api_key=self.api_key_openai
            )
        
        print(f"  Model: {self.model}")
        print(f"  Endpoint: {self.endpoint}")
    
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> Dict:
        """
        Generate response using Azure OpenAI.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            **kwargs: Additional parameters for API
            
        Returns:
            Dictionary with response and metadata
        """
        max_retries = kwargs.pop('max_retries', 5)
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                
                return {
                    'content': response.choices[0].message.content,
                    'finish_reason': response.choices[0].finish_reason,
                    'usage': {
                        'prompt_tokens': response.usage.prompt_tokens,
                        'completion_tokens': response.usage.completion_tokens,
                        'total_tokens': response.usage.total_tokens
                    },
                    'model': response.model
                }
            except Exception as e:
                is_rate_limit = False
                status_code = getattr(e, 'status_code', None)
                if status_code == 429:
                    is_rate_limit = True
                else:
                    message_text = str(e)
                    if '429' in message_text or 'rate limit' in message_text.lower():
                        is_rate_limit = True
                if is_rate_limit and attempt < max_retries:
                    retry_after = self._extract_retry_after_seconds(e)
                    sleep_seconds = retry_after or min(2 ** attempt, 30)
                    print(f"⚠ Rate limit (attempt {attempt}/{max_retries}). Retrying in {sleep_seconds}s...")
                    time.sleep(sleep_seconds)
                    continue
                return {
                    'content': None,
                    'error': str(e),
                    'finish_reason': 'error'
                }
    
    def answer_with_context(
        self,
        question: str,
        context: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
        answer_options: Optional[List[str]] = None
    ) -> Dict:
        """
        Answer question using provided context.
        
        Args:
            question: The user's question
            context: Retrieved context from RAG system
            system_prompt: Optional custom system prompt
            temperature: Lower for factual, higher for creative
            max_tokens: Maximum response length
            
        Returns:
            Dictionary with answer and metadata
        """
        if system_prompt is None:
            system_prompt = """Your task is to answer a question based on the provided context from this scientific papers.

# INSTRUCTIONS
1. Think step by step. Analyze the given text carefully.
2. Answer each question based on what is stated in the paper.
3. Your answer MUST be one of the provided options for each question.
"""
        
        if answer_options:
            formatted_options = "\n".join(f"  - {opt.strip()}" for opt in answer_options if opt and opt.strip())
            
            user_content = f"""#QUESTION {question}
# CONTEXT
{context}

INSTRUCTIONS    
You MUST respond with a valid JSON object in this exact format:
{{
  "answer": "<should be one of {formatted_options}>",
  "confidence": <number between 0.0 and 1.0>,
  "reasoning": "<brief explanation>"
}}

"""
        else:
            user_content = f"""#QUESTION {question}
# CONTEXT
{context}

Please answer the question based on the provided context. If citing specific information, use [Source N] notation."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        return self.generate_response(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )


def parse_structured_answer(raw_answer: str, answer_options: List[str]) -> Dict:
    """
    Parse structured JSON answer from LLM response.
    
    Args:
        raw_answer: Raw LLM response that should contain JSON
        answer_options: List of valid answer options
        
    Returns:
        Dictionary with parsed answer, confidence, and reasoning
    """
    if not raw_answer:
        return {
            'answer': None,
            'confidence': 0.0,
            'reasoning': 'No response from LLM',
            'parse_error': True
        }
    
    # Clean up the response text
    json_str = raw_answer.strip()
    
    # Remove markdown code blocks if present
    if json_str.startswith('```'):
        # Split by ``` and get the content between first and second ```
        parts = json_str.split('```')
        if len(parts) >= 3:
            json_str = parts[1]
            # Remove 'json' language identifier if present
            if json_str.strip().startswith('json'):
                json_str = json_str.strip()[4:].strip()
        else:
            # Malformed code block, try to extract JSON
            json_str = json_str.replace('```json', '').replace('```', '').strip()
    
    # Try to find JSON object if not already clean
    if not json_str.startswith('{'):
        json_match = re.search(r'\{[^{}]*"answer"[^{}]*\}', json_str, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
    
    try:
        parsed = json.loads(json_str)
        
        # Validate structure
        if not isinstance(parsed, dict):
            raise ValueError("Response is not a JSON object")
        
        answer = parsed.get('answer', '').strip()
        confidence = parsed.get('confidence', 0.5)
        reasoning = parsed.get('reasoning', '')
        
        # Validate confidence
        if not isinstance(confidence, (int, float)):
            confidence = 0.5
        confidence = max(0.0, min(1.0, float(confidence)))
        
        # Validate answer against options
        normalized_options = [opt.strip() for opt in answer_options if opt and opt.strip()]
        option_map = {opt.lower(): opt for opt in normalized_options}
        
        if answer.lower() in option_map:
            answer = option_map[answer.lower()]
        else:
            # Try partial matching
            for opt in normalized_options:
                if answer.lower() in opt.lower() or opt.lower() in answer.lower():
                    answer = opt
                    break
        
        return {
            'answer': answer,
            'confidence': confidence,
            'reasoning': reasoning,
            'parse_error': False
        }
        
    except (json.JSONDecodeError, ValueError) as e:
        # JSON parsing failed - mark as error, don't use raw text as fallback
        return {
            'answer': None,
            'confidence': 0.0,
            'reasoning': f'Failed to parse JSON response: {str(e)}',
            'parse_error': True,
            'raw_response': raw_answer[:200]  # Keep snippet for debugging
        }


class CompleteRAGSystem:
    """Complete RAG system combining vector retrieval with LLM generation."""
    
    def __init__(
        self,
        rag_system,
        llm_client: Optional[AzureOpenAIClient] = None,
        default_n_results: int = 10,
        use_multi_query: bool = False,
        predefined_queries_file: Optional[str] = None
    ):
        """
        Initialize complete RAG system.
        
        Args:
            rag_system: Instance of ScientificRAG for retrieval
            llm_client: Instance of AzureOpenAIClient (creates new if None)
            default_n_results: Default number of chunks to retrieve
            use_multi_query: Use multi-query retrieval strategy
            predefined_queries_file: Path to JSON file with predefined queries
        """
        self.rag = rag_system
        self.llm = llm_client or AzureOpenAIClient()
        self.default_n_results = default_n_results
        self.use_multi_query = use_multi_query
        
        # Load predefined queries if provided
        self.predefined_queries = {}
        if predefined_queries_file:
            try:
                with open(predefined_queries_file, 'r') as f:
                    self.predefined_queries = json.load(f)
                print(f"✓ Loaded {len(self.predefined_queries)} predefined query sets")
            except Exception as e:
                print(f"⚠ Could not load predefined queries: {e}")
        
        # Connect LLM to RAG system for query enhancement
        self.rag.set_llm_client(self.llm)
        
        if predefined_queries_file:
            print("✓ Complete RAG system ready (Predefined Queries Mode)")
        elif use_multi_query:
            print("✓ Complete RAG system ready (Multi-Query Mode: Enhanced + HyDE + Expanded)")
        else:
            print("✓ Complete RAG system ready (Single Query Mode: LLM-Enhanced)")
    
    def answer_question(
        self,
        question: str,
        n_results: Optional[int] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
        include_sources: bool = True,
        system_prompt: Optional[str] = None,
        answer_options: Optional[List[str]] = None,
        doi: Optional[str] = None,
        question_key: Optional[str] = None,
        include_abstract: bool = False,
        abstract: Optional[str] = None
    ) -> Dict:
        """
        Complete RAG pipeline: Retrieve context + Generate answer.
        
        Args:
            question: The user's question
            n_results: Number of chunks to retrieve (uses default if None)
            temperature: LLM temperature (0-2)
            max_tokens: Maximum response length
            include_sources: Whether to include source metadata in response
            system_prompt: Optional custom system prompt
            
        Returns:
            Dictionary with answer, context, sources, and metadata
        """
        n_results = n_results or self.default_n_results
        
        # Get predefined queries if available for this question
        predefined_queries = None
        if question_key and question_key in self.predefined_queries:
            predefined_queries = self.predefined_queries[question_key]
            print(f"🔍 Retrieving top {n_results} unique chunks using {len(predefined_queries)} predefined queries...")
        elif self.use_multi_query:
            print(f"🔍 Retrieving top {n_results} relevant chunks (multi-query mode)...")
        else:
            print(f"🔍 Retrieving top {n_results} relevant chunks...")
        
        metadata_filter = {'doi': doi} if doi else None
        rag_response = self.rag.answer_question(
            question=question,
            n_context_chunks=n_results,
            include_metadata=True,
            metadata_filter=metadata_filter,
            use_multi_query=self.use_multi_query,
            predefined_queries=predefined_queries
        )
        
        # Add abstract to context if provided
        context = rag_response['context']
        if include_abstract and abstract:
            abstract_section = f"\n\n{'='*70}\nPAPER ABSTRACT\n{'='*70}\n{abstract}\n{'='*70}\n\n"
            context = abstract_section + context
        
        # Step 2: Generate answer using LLM
        print(f"🤖 Generating answer with {self.llm.model}...")
        llm_response = self.llm.answer_with_context(
            question=question,
            context=context,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            answer_options=answer_options
        )
        
        # Step 3: Combine results
        raw_answer = llm_response.get('content')
        
        # Parse structured answer if options provided
        if answer_options and raw_answer:
            structured = parse_structured_answer(raw_answer, answer_options)
            final_answer = structured['answer']
            confidence = structured['confidence']
            reasoning = structured['reasoning']
            parse_error = structured['parse_error']
        else:
            final_answer = raw_answer
            confidence = None
            reasoning = None
            parse_error = False
        
        result = {
            'question': question,
            'answer': final_answer,
            'confidence': confidence,
            'reasoning': reasoning,
            'sources': rag_response['sources'] if include_sources else None,
            'n_sources': rag_response['n_sources'],
            'context_used': rag_response['context'] if include_sources else None,
            'llm_metadata': {
                'model': llm_response.get('model'),
                'usage': llm_response.get('usage'),
                'finish_reason': llm_response.get('finish_reason')
            },
            'error': llm_response.get('error'),
            'raw_answer': raw_answer,
            'parse_error': parse_error,
            'answer_options': [opt for opt in answer_options] if answer_options else None
        }
        
        if result['error']:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✓ Answer generated ({llm_response.get('usage', {}).get('total_tokens', 0)} tokens)")
        
        return result
    
    def answer_aging_questions(
        self,
        questions: List[Tuple],
        output_file: Optional[str] = None,
        temperature: float = 0.3,
        doi: Optional[str] = None,
        max_tokens: int = 200
    ) -> Dict:
        """
        Answer multiple aging research questions.
        
        Args:
            questions: List of tuples (question_num, question_text, answer_options, question_key)
            output_file: Optional file to save results
            temperature: LLM temperature
            doi: Optional DOI to filter by
            max_tokens: Maximum tokens for answer
            
        Returns:
            Dictionary with all answers
        """
        print(f"\n{'='*70}")
        print("Answering Aging Research Questions with Complete RAG")
        print(f"{'='*70}\n")
        
        all_answers = {}
        
        for question_tuple in questions:
            # Handle both old format (3 items) and new format (4 items)
            if len(question_tuple) == 4:
                q_num, q_text, q_options, q_key = question_tuple
            else:
                q_num, q_text, q_options = question_tuple
                q_key = f"Q{q_num}"
            print(f"\nQ{q_num}: {q_text[:70]}...")
            if q_options:
                print("Options:")
                for opt in q_options:
                    print(f"  - {opt}")

            # Determine optimal n_results based on question type
            if q_num == 1 or q_num == 5:
                n_results = 12  # Biomarker questions
            elif q_num == 2:
                n_results = 10  # Mechanism questions
            elif q_num == 3:
                n_results = 8   # Intervention questions
            else:
                n_results = 15  # Species-specific questions

            # Use question text directly - DOI filter is applied at retrieval level
            # No need to add DOI to query as it doesn't help semantic matching
            answer = self.answer_question(
                question=q_text,
                n_results=n_results,
                temperature=temperature,
                max_tokens=max(max_tokens, 300) if q_options else 500,  # Ensure enough tokens for JSON
                answer_options=q_options,
                doi=doi,
                question_key=q_key  # Pass question key to use predefined queries
            )

            all_answers[q_key] = answer
            all_answers[q_key]['question_number'] = q_num
            all_answers[q_key]['question_text'] = q_text
            all_answers[q_key]['question_key'] = q_key

            # Print answer
            if answer.get('parse_error'):
                print(f"\n{'─'*70}")
                print(f"❌ PARSE ERROR: {answer.get('reasoning', 'Unknown error')}")
                print(f"{'─'*70}")
                print(f"Sources: {answer['n_sources']}")
            elif answer['answer']:
                print(f"\n{'─'*70}")
                print(f"Answer: {answer['answer']}")
                if answer.get('confidence') is not None:
                    print(f"Confidence: {answer['confidence']:.2f}")
                if answer.get('reasoning'):
                    print(f"Reasoning: {answer['reasoning']}")
                print(f"{'─'*70}")
                print(f"Sources: {answer['n_sources']}")
                if answer['llm_metadata'].get('usage'):
                    print(f"Tokens: {answer['llm_metadata']['usage']['total_tokens']}")
            print()
        
        # Save if requested
        if output_file:
            import json
            # Remove context to keep file smaller
            save_data = {}
            for key, value in all_answers.items():
                save_data[key] = {
                    'question': value['question'],
                    'answer': value['answer'],
                    'confidence': value.get('confidence'),
                    'reasoning': value.get('reasoning'),
                    'raw_answer': value.get('raw_answer'),
                    'parse_error': value.get('parse_error', False),
                    'answer_options': value.get('answer_options'),
                    'question_number': value.get('question_number'),
                    'question_text': value.get('question_text'),
                    'question_key': value.get('question_key'),
                    'sources': value['sources'],
                    'n_sources': value['n_sources'],
                    'llm_metadata': value['llm_metadata']
                }
            
            with open(output_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            print(f"✓ Results saved to: {output_file}\n")
        
        # Show cache statistics and enhanced queries
        if self.rag.query_preprocessor:
            cache_stats = self.rag.query_preprocessor.get_cache_stats()
            if cache_stats['total_cached'] > 0:
                print(f"\n{'='*70}")
                print("Query Cache Statistics")
                print(f"{'='*70}")
                print(f"Enhanced queries cached: {cache_stats['enhanced_queries']}")
                print(f"Total preprocessed queries: {cache_stats['preprocessed_queries']}")
                print(f"💡 Queries are cached and reused across all papers for efficiency")
                print(f"{'='*70}\n")
                
                # Show enhanced queries
                self.rag.query_preprocessor.print_enhanced_queries()
        
        return all_answers

    def _extract_retry_after_seconds(self, error: Exception) -> Optional[int]:
        message_text = str(error).lower()
        match = re.search(r"retry after (\d+)", message_text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None

    def _select_option(self, raw_answer: Optional[str], options: List[str]) -> Optional[str]:
        if not raw_answer:
            return None
        normalized_options = [opt.strip() for opt in options if opt and opt.strip()]
        if not normalized_options:
            return None
        first_line = raw_answer.strip().splitlines()[0].strip()
        if not first_line:
            return None
        variants = []
        variants.append(first_line)
        variants.append(first_line.rstrip('.'))
        variants.append(first_line.rstrip('!'))
        variants.append(first_line.rstrip('?'))
        variants.append(first_line.split('.')[0].strip())
        variants.append(first_line.split('!')[0].strip())
        variants.append(first_line.split('?')[0].strip())
        variants.append(first_line.split(',')[0].strip())
        lowered_variants = [v.lower() for v in variants if v]
        option_map = {opt.lower(): opt for opt in normalized_options}
        for variant in lowered_variants:
            if variant in option_map:
                return option_map[variant]
        option_prefix_map = {opt.lower().split(',')[0].strip(): opt for opt in normalized_options}
        for variant in lowered_variants:
            if variant in option_prefix_map:
                return option_prefix_map[variant]
        for variant in lowered_variants:
            for opt in normalized_options:
                opt_lower = opt.lower()
                if variant in opt_lower or opt_lower in variant:
                    return opt
        return None


if __name__ == "__main__":
    # Test Azure OpenAI connection
    print("Testing Azure OpenAI connection...")
    
    try:
        client = AzureOpenAIClient()
        
        test_response = client.generate_response(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Connection successful!' if you can read this."}
            ],
            temperature=0.3,
            max_tokens=50
        )
        
        if test_response.get('content'):
            print(f"\n✓ Azure OpenAI connection successful!")
            print(f"Response: {test_response['content']}")
            print(f"Usage: {test_response.get('usage')}")
        else:
            print(f"\n❌ Error: {test_response.get('error')}")
    
    except Exception as e:
        print(f"\n❌ Failed to connect: {e}")
        print("\nMake sure your .env file contains:")
        print("  AZURE_OPENAI_ENDPOINT")
        print("  AZURE_OPENAI_API_KEY")
        print("  AZURE_OPENAI_API_VERSION")
        print("  OPENAI_MODEL")
