"""
LLM Integration for RAG System using Azure OpenAI.
Provides complete question-answering with context from vector database.
"""
import os
import re
import time
from typing import Dict, List, Optional, Tuple
from openai import AzureOpenAI
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
        
        if not self.endpoint or not self.api_key:
            raise ValueError(
                "Azure OpenAI credentials not found. "
                "Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env file."
            )
        
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )
        
        print(f"✓ Azure OpenAI client initialized")
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
            system_prompt = """You are a scientific research assistant specializing in analyzing biomedical literature, particularly aging research.

Your task is to answer questions based STRICTLY on the provided context from scientific papers.

Guidelines:
1. **Base answers ONLY on provided context** - Do not use external knowledge
2. **Cite sources** using [Source N] notation when referencing specific information
3. **Be precise** - Distinguish between "quantitatively shown" vs "suggested" vs "mentioned"
4. **Acknowledge uncertainty** - If context doesn't contain enough information, say so
5. **Use scientific terminology** appropriately
6. **Quote key findings** when relevant (with source citations)
7. **For yes/no questions** - Provide clear answer, then brief evidence
8. **Synthesize across sources** when multiple papers provide relevant information

Format for specific question types:
- Biomarker questions: State whether quantitatively validated, cite statistics
- Mechanism questions: Describe pathway/process, cite supporting evidence
- Species-specific questions: State whether explicitly discussed, quote relevant passages
- Intervention questions: Distinguish proposed vs tested interventions"""
        
        options_instruction = ""
        if answer_options:
            formatted_options = "\n".join(opt.strip() for opt in answer_options if opt and opt.strip())
            if formatted_options:
                options_instruction = (
                    "\nYou must respond with exactly one of the following options. Return only the option text and nothing else:\n"
                    f"{formatted_options}\n"
                )

        user_content = f"""Question: {question}

Context from scientific papers:

{context}

Please answer the question based on the provided context. If citing specific information, use [Source N] notation."""

        if options_instruction:
            user_content += f"{options_instruction}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        return self.generate_response(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )


class CompleteRAGSystem:
    """Complete RAG system combining vector retrieval with LLM generation."""
    
    def __init__(
        self,
        rag_system,
        llm_client: Optional[AzureOpenAIClient] = None,
        default_n_results: int = 10
    ):
        """
        Initialize complete RAG system.
        
        Args:
            rag_system: Instance of ScientificRAG for retrieval
            llm_client: Instance of AzureOpenAIClient (creates new if None)
            default_n_results: Default number of chunks to retrieve
        """
        self.rag = rag_system
        self.llm = llm_client or AzureOpenAIClient()
        self.default_n_results = default_n_results
        
        print("✓ Complete RAG system ready")
    
    def answer_question(
        self,
        question: str,
        n_results: Optional[int] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
        include_sources: bool = True,
        system_prompt: Optional[str] = None,
        answer_options: Optional[List[str]] = None,
        doi: Optional[str] = None
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
        
        # Step 1: Retrieve relevant context
        print(f"🔍 Retrieving top {n_results} relevant chunks...")
        metadata_filter = {'doi': doi} if doi else None
        rag_response = self.rag.answer_question(
            question=question,
            n_context_chunks=n_results,
            include_metadata=True,
            metadata_filter=metadata_filter
        )
        
        # Step 2: Generate answer using LLM
        print(f"🤖 Generating answer with {self.llm.model}...")
        llm_response = self.llm.answer_with_context(
            question=question,
            context=rag_response['context'],
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            answer_options=answer_options
        )
        
        # Step 3: Combine results
        raw_answer = llm_response.get('content')
        selected_option = None
        if answer_options:
            selected_option = self._select_option(raw_answer, answer_options)
        final_answer = selected_option or raw_answer
        result = {
            'question': question,
            'answer': final_answer,
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
            'selected_option': selected_option,
            'answer_options': [opt for opt in answer_options] if answer_options else None
        }
        
        if result['error']:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✓ Answer generated ({llm_response.get('usage', {}).get('total_tokens', 0)} tokens)")
        
        return result
    
    def answer_aging_questions(
        self,
        questions: List[Tuple[int, str, List[str]]],
        output_file: Optional[str] = None,
        temperature: float = 0.3,
        doi: Optional[str] = None
    ) -> Dict:
        """
        Answer multiple aging research questions.
        
        Args:
            questions: List of (question_num, question_text, answer_options) tuples
            output_file: Optional file to save results
            temperature: LLM temperature
            
        Returns:
            Dictionary with all answers
        """
        print(f"\n{'='*70}")
        print("Answering Aging Research Questions with Complete RAG")
        print(f"{'='*70}\n")
        
        all_answers = {}
        
        for q_num, q_text, q_options in questions:
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

            prompt_question = q_text
            if doi:
                prompt_question = f"For the paper with DOI {doi}, {q_text}"

            answer = self.answer_question(
                question=prompt_question,
                n_results=n_results,
                temperature=temperature,
                max_tokens=64 if q_options else 200,
                answer_options=q_options,
                doi=doi
            )

            all_answers[f"Q{q_num}"] = answer
            all_answers[f"Q{q_num}"]['question_number'] = q_num
            all_answers[f"Q{q_num}"]['question_text'] = q_text
            all_answers[f"Q{q_num}"]['prompt_question'] = prompt_question

            # Print answer
            if answer['answer']:
                print(f"\n{'─'*70}")
                print(f"Answer:\n{answer['answer']}")
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
                    'raw_answer': value.get('raw_answer'),
                    'selected_option': value.get('selected_option'),
                    'answer_options': value.get('answer_options'),
                    'question_number': value.get('question_number'),
                    'question_text': value.get('question_text'),
                    'prompt_question': value.get('prompt_question'),
                    'sources': value['sources'],
                    'n_sources': value['n_sources'],
                    'llm_metadata': value['llm_metadata']
                }
            
            with open(output_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            print(f"✓ Results saved to: {output_file}\n")
        
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
