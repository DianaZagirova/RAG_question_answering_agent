"""
Advanced Query Preprocessing for RAG System.
Implements best practices: HyDE, query expansion, and query decomposition.
"""
from typing import List, Dict, Optional, Tuple
import re


class QueryPreprocessor:
    """
    Preprocesses queries to improve retrieval quality using multiple techniques:
    1. HyDE (Hypothetical Document Embeddings) - Generate hypothetical answer
    2. Query Expansion - Add synonyms and related terms
    3. Query Decomposition - Break complex queries into sub-queries
    4. Contextualization - Add domain-specific context
    5. LLM Enhancement - Transform query to match scientific text style
    """
    
    def __init__(self, llm_client=None, use_cache=True):
        """
        Initialize query preprocessor.
        
        Args:
            llm_client: Optional LLM client for advanced preprocessing (HyDE)
            use_cache: Whether to cache preprocessed queries
        """
        self.llm_client = llm_client
        self.use_cache = use_cache
        self._query_cache = {}  # Cache for preprocessed queries
        self._enhanced_query_cache = {}  # Cache for LLM-enhanced queries
        
        # Domain-specific term mappings for aging research
        self.aging_synonyms = {
            'aging': ['ageing', 'senescence', 'age-related decline', 'biological aging'],
            'biomarker': ['marker', 'indicator', 'predictor', 'measure'],
            'longevity': ['lifespan', 'life span', 'life expectancy', 'survival'],
            'intervention': ['treatment', 'therapy', 'manipulation', 'strategy'],
            'mechanism': ['pathway', 'process', 'molecular basis', 'biological process'],
            'calorie restriction': ['CR', 'dietary restriction', 'caloric restriction', 'food restriction'],
            'naked mole rat': ['Heterocephalus glaber', 'NMR', 'naked mole-rat'],
            'lifespan': ['longevity', 'life span', 'survival time', 'maximum lifespan'],
        }
        
        # Key concepts to expand for better retrieval
        self.concept_expansions = {
            'molecular mechanism': [
                'gene expression', 'protein function', 'signaling pathway',
                'metabolic process', 'cellular process', 'molecular pathway'
            ],
            'biomarker': [
                'DNA methylation', 'telomere length', 'epigenetic clock',
                'blood marker', 'molecular marker', 'physiological marker'
            ],
            'longevity intervention': [
                'rapamycin', 'metformin', 'senolytics', 'NAD+ precursors',
                'calorie restriction', 'exercise', 'dietary intervention'
            ]
        }
    
    def preprocess_query(
        self,
        query: str,
        use_hyde: bool = False,
        use_expansion: bool = True,
        use_contextualization: bool = True,
        use_llm_enhancement: bool = False
    ) -> Dict[str, any]:
        """
        Preprocess query using multiple techniques.
        
        Args:
            query: Original query text
            use_hyde: Whether to generate hypothetical document (requires LLM)
            use_expansion: Whether to expand with synonyms
            use_contextualization: Whether to add domain context
            use_llm_enhancement: Whether to use LLM to enhance query to match scientific style
            
        Returns:
            Dictionary with original query, expanded query, and metadata
        """
        # Check cache first
        cache_key = f"{query}|{use_hyde}|{use_expansion}|{use_contextualization}|{use_llm_enhancement}"
        if self.use_cache and cache_key in self._query_cache:
            return self._query_cache[cache_key]
        
        result = {
            'original_query': query,
            'processed_query': query,
            'expanded_terms': [],
            'hyde_document': None,
            'sub_queries': [],
            'enhanced_query': None
        }
        
        # 1. Query Expansion with synonyms
        if use_expansion:
            expanded_query, expanded_terms = self._expand_query(query)
            result['processed_query'] = expanded_query
            result['expanded_terms'] = expanded_terms
        
        # 2. Contextualization - add domain-specific context
        if use_contextualization:
            contextualized = self._contextualize_query(result['processed_query'])
            result['processed_query'] = contextualized
        
        # 3. HyDE - Generate hypothetical document (if LLM available)
        if use_hyde and self.llm_client:
            hyde_doc = self._generate_hyde_document(query)
            result['hyde_document'] = hyde_doc
        
        # 4. Query Decomposition for complex questions
        if self._is_complex_query(query):
            sub_queries = self._decompose_query(query)
            result['sub_queries'] = sub_queries
        
        # 5. LLM Enhancement - Transform to scientific style (BEST for matching chunks)
        if use_llm_enhancement and self.llm_client:
            enhanced = self._enhance_query_with_llm(query)
            if enhanced:
                result['enhanced_query'] = enhanced
                result['processed_query'] = enhanced  # Use enhanced as primary
        
        # Cache result
        if self.use_cache:
            self._query_cache[cache_key] = result
        
        return result
    
    def _expand_query(self, query: str) -> Tuple[str, List[str]]:
        """
        Expand query with synonyms and related terms.
        
        Returns:
            Tuple of (expanded_query, list_of_added_terms)
        """
        query_lower = query.lower()
        expanded_terms = []
        
        # Find matching terms and add synonyms
        for term, synonyms in self.aging_synonyms.items():
            if term in query_lower:
                # Add most relevant synonym (first one)
                if synonyms:
                    expanded_terms.append(synonyms[0])
        
        # Add concept expansions for key phrases
        for concept, expansions in self.concept_expansions.items():
            if concept in query_lower:
                # Add top 2 most relevant expansions
                expanded_terms.extend(expansions[:2])
        
        # Construct expanded query
        if expanded_terms:
            # Add expanded terms as additional context
            expanded_query = f"{query} (related: {', '.join(expanded_terms)})"
        else:
            expanded_query = query
        
        return expanded_query, expanded_terms
    
    def _contextualize_query(self, query: str) -> str:
        """
        Add domain-specific context to improve retrieval.
        """
        query_lower = query.lower()
        
        # Add context based on question type
        if 'biomarker' in query_lower:
            context = "In the context of aging research and biomarkers: "
        elif 'mechanism' in query_lower:
            context = "Regarding molecular mechanisms of aging: "
        elif 'intervention' in query_lower or 'treatment' in query_lower:
            context = "Concerning longevity interventions and therapies: "
        elif any(species in query_lower for species in ['naked mole rat', 'bird', 'mammal', 'species']):
            context = "In comparative biology and species longevity: "
        elif 'calorie restriction' in query_lower or 'dietary' in query_lower:
            context = "Regarding dietary interventions and calorie restriction: "
        else:
            context = "In aging and longevity research: "
        
        return context + query
    
    def _is_complex_query(self, query: str) -> bool:
        """
        Determine if query is complex and should be decomposed.
        """
        # Complex if it has multiple clauses or conditions
        complexity_indicators = [
            ' and ' in query.lower(),
            ' or ' in query.lower(),
            query.count('?') > 1,
            'if yes' in query.lower(),
            'whether' in query.lower() and 'or' in query.lower(),
        ]
        
        return sum(complexity_indicators) >= 2
    
    def _decompose_query(self, query: str) -> List[str]:
        """
        Decompose complex query into simpler sub-queries.
        """
        sub_queries = []
        
        # Split on common conjunctions
        if 'if yes' in query.lower():
            parts = re.split(r'\?\s*if yes,?', query, flags=re.IGNORECASE)
            if len(parts) == 2:
                sub_queries.append(parts[0].strip() + '?')
                sub_queries.append(parts[1].strip())
        
        # Split on 'and' for compound questions
        elif ' and ' in query.lower():
            parts = query.split(' and ')
            for part in parts:
                if part.strip():
                    sub_queries.append(part.strip())
        
        # If no decomposition worked, return original
        if not sub_queries:
            sub_queries = [query]
        
        return sub_queries
    
    def _generate_hyde_document(self, query: str) -> Optional[str]:
        """
        Generate hypothetical document using HyDE technique.
        Creates a hypothetical answer that would appear in relevant documents.
        """
        if not self.llm_client:
            return None
        
        hyde_prompt = f"""Generate a brief, factual passage (2-3 sentences) that would appear in a scientific paper answering this question:

Question: {query}

Write as if you are excerpting from the results or discussion section of a research paper. Be specific and use scientific terminology."""
        
        try:
            response = self.llm_client.generate_response(
                messages=[
                    {"role": "system", "content": "You are a scientific writing assistant. Generate hypothetical scientific text."},
                    {"role": "user", "content": hyde_prompt}
                ],
                temperature=0.5,
                max_tokens=150
            )
            
            if response.get('content'):
                return response['content']
        except Exception as e:
            print(f"⚠ HyDE generation failed: {e}")
        
        return None
    
    def _enhance_query_with_llm(self, query: str) -> Optional[str]:
        """
        Use LLM to transform query into scientific text style that matches paper chunks.
        This is the most effective technique for improving retrieval.
        """
        # Check cache
        if self.use_cache and query in self._enhanced_query_cache:
            return self._enhanced_query_cache[query]
        
        if not self.llm_client:
            return None
        
        enhancement_prompt = f"""Transform this question into a declarative statement or passage that would appear in a scientific paper's abstract, results, or discussion section.

Question: {query}

Rules:
1. Convert from question format to declarative/descriptive format
2. Use scientific terminology and formal academic language
3. Make it sound like it's from a research paper (not a question)
4. Keep it concise (1-2 sentences max)
5. Include key scientific terms and concepts from the question
6. Remove question marks and interrogative structure

Example:
Question: "Does it suggest an aging biomarker?"
Transformed: "This study identifies and validates aging biomarkers associated with mortality and age-related physiological decline."

Now transform the given question:"""
        
        try:
            response = self.llm_client.generate_response(
                messages=[
                    {"role": "system", "content": "You are an expert at transforming questions into scientific text. Output only the transformed text, nothing else."},
                    {"role": "user", "content": enhancement_prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            if response.get('content'):
                enhanced = response['content'].strip()
                # Remove quotes if LLM added them
                enhanced = enhanced.strip('"\'""''')
                
                # Cache the result
                if self.use_cache:
                    self._enhanced_query_cache[query] = enhanced
                
                return enhanced
        except Exception as e:
            print(f"⚠ Query enhancement failed: {e}")
        
        return None
    
    def clear_cache(self):
        """Clear all cached queries."""
        self._query_cache.clear()
        self._enhanced_query_cache.clear()
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'preprocessed_queries': len(self._query_cache),
            'enhanced_queries': len(self._enhanced_query_cache),
            'total_cached': len(self._query_cache) + len(self._enhanced_query_cache)
        }
    
    def get_enhanced_queries(self) -> Dict[str, str]:
        """
        Get all cached enhanced queries.
        
        Returns:
            Dictionary mapping original query to enhanced query
        """
        return dict(self._enhanced_query_cache)
    
    def print_enhanced_queries(self):
        """Print all enhanced queries in a readable format."""
        if not self._enhanced_query_cache:
            print("No enhanced queries in cache.")
            return
        
        print("\n" + "="*70)
        print("Enhanced Queries Cache")
        print("="*70)
        for i, (original, enhanced) in enumerate(self._enhanced_query_cache.items(), 1):
            print(f"\n[{i}] Original:")
            print(f"    {original[:100]}{'...' if len(original) > 100 else ''}")
            print(f"    Enhanced:")
            print(f"    {enhanced}")
        print("="*70 + "\n")
    
    def create_multi_query_retrieval(
        self,
        query: str,
        use_hyde: bool = False
    ) -> List[str]:
        """
        Create multiple query variants for improved retrieval coverage.
        
        Returns:
            List of query variants to use for retrieval
        """
        queries = []
        
        # Original query
        queries.append(query)
        
        # Preprocessed query
        preprocessed = self.preprocess_query(
            query,
            use_hyde=use_hyde,
            use_expansion=True,
            use_contextualization=True
        )
        
        if preprocessed['processed_query'] != query:
            queries.append(preprocessed['processed_query'])
        
        # Sub-queries if complex
        if preprocessed['sub_queries'] and len(preprocessed['sub_queries']) > 1:
            queries.extend(preprocessed['sub_queries'])
        
        # HyDE document if available
        if preprocessed['hyde_document']:
            queries.append(preprocessed['hyde_document'])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)
        
        return unique_queries


def preprocess_for_scientific_papers(query: str) -> str:
    """
    Quick preprocessing function for scientific paper queries.
    Optimizes query format to match typical scientific paper structure.
    """
    # Remove conversational elements
    query = query.strip()
    
    # Convert questions to declarative form for better matching
    declarative_patterns = [
        (r'^does it (suggest|explain|claim|contain)', r'suggests \1'),
        (r'^does the paper (suggest|explain|claim|contain)', r'the paper \1'),
        (r'^is there', r'there is'),
        (r'^are there', r'there are'),
    ]
    
    query_lower = query.lower()
    for pattern, replacement in declarative_patterns:
        query_lower = re.sub(pattern, replacement, query_lower)
    
    # Add scientific context markers
    if 'biomarker' in query_lower:
        query = f"aging biomarker research: {query}"
    elif 'mechanism' in query_lower:
        query = f"molecular mechanism: {query}"
    
    return query


if __name__ == "__main__":
    # Test query preprocessor
    print("Testing Query Preprocessor\n")
    
    preprocessor = QueryPreprocessor()
    
    test_queries = [
        "Does it suggest an aging biomarker?",
        "Does it explain why the naked mole rat can live 40+ years despite its small size?",
        "Does it suggest a molecular mechanism of aging? If yes, does it provide quantitative evidence?",
    ]
    
    for query in test_queries:
        print(f"\nOriginal: {query}")
        result = preprocessor.preprocess_query(query)
        print(f"Processed: {result['processed_query']}")
        if result['expanded_terms']:
            print(f"Expanded terms: {result['expanded_terms']}")
        if result['sub_queries'] and len(result['sub_queries']) > 1:
            print(f"Sub-queries: {result['sub_queries']}")
        print("-" * 70)
