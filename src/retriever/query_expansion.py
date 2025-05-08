"""
Query expansion module that implements techniques like HyDE (Hypothetical Document Embeddings)
"""
import os
import logging
from typing import List, Dict, Any, Optional
import time
import re
from src.vector_store.supabase_store import SupabaseVectorStore

logger = logging.getLogger(__name__)

class QueryExpander:
    """Expands user queries to improve retrieval performance"""
    
    def __init__(self, hyde_enabled: bool = True, use_stored_queries: bool = True):
        self.hyde_enabled = hyde_enabled and os.getenv("ENABLE_HYDE", "true").lower() == "true"
        self.use_stored_queries = use_stored_queries
        self.llm_provider = os.getenv("DEFAULT_LLM", "openai")
        # Use the factory to create the appropriate vector store
        self.vector_store = SupabaseVectorStore()
        
        # Set up appropriate client for HyDE
        if self.hyde_enabled:
            if self.llm_provider == "openai":
                import openai
                openai.api_key = os.getenv("OPENAI_API_KEY")
                self.client = openai
            elif self.llm_provider == "anthropic":
                import anthropic
                self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    def expand_query(self, query: str) -> Dict[str, Any]:
        """
        Expand a user query using techniques like HyDE
        
        Args:
            query: The original user query
            
        Returns:
            Dict containing the original query and expanded queries
        """
        result = {"original_query": query, "expanded_queries": []}
        
        if self.hyde_enabled:
            # First try to find relevant stored HyDE queries from the database
            if self.use_stored_queries:
                stored_queries = self._get_relevant_stored_queries(query)
                if stored_queries:
                    result["expanded_queries"].extend(stored_queries)
                    result["source"] = "database"
                    logger.info(f"Using {len(stored_queries)} stored HyDE queries from database")
            
            # If we didn't find any relevant stored queries, generate a new one
            if not result["expanded_queries"]:
                try:
                    hyde_query = self._generate_hyde_document(query)
                    if hyde_query:
                        result["expanded_queries"].append(hyde_query)
                        result["source"] = "generated"
                        logger.info(f"Generated new HyDE query: {hyde_query[:50]}...")
                except Exception as e:
                    logger.error(f"Error generating HyDE query: {str(e)}")
                
        # Add more query expansion methods here as needed
        # For example: query reformulation, query paraphrasing, etc.
        
        logger.info(f"Expanded query with {len(result['expanded_queries'])} alternatives")
        return result
    
    def _get_relevant_stored_queries(self, query: str) -> List[str]:
        """
        Retrieve relevant stored HyDE queries from the database
        
        Args:
            query: The user query
            
        Returns:
            List of relevant HyDE query texts
        """
        relevant_queries = []
        
        try:
            # Get stored HyDE queries with sensible limits
            if hasattr(self.vector_store, 'get_hyde_queries'):
                # Only get recent queries (last 30 days) and limit to 50 records
                hyde_query_records = self.vector_store.get_hyde_queries(
                    limit=50,
                    max_age_days=30
                )
                
                # Extract the query texts and compare with the current query
                for record in hyde_query_records:
                    queries_data = record.get("queries", {})
                    if isinstance(queries_data, dict) and "queries" in queries_data:
                        for query_obj in queries_data["queries"]:
                            stored_query = query_obj.get("query", "")
                            # Check if the stored query is relevant to the current query
                            if self._is_query_relevant(query, stored_query):
                                relevant_queries.append(stored_query)
                
                logger.info(f"Found {len(relevant_queries)} relevant stored HyDE queries")
            else:
                logger.warning("Vector store does not support retrieving HyDE queries")
                
        except Exception as e:
            logger.error(f"Error retrieving stored HyDE queries: {str(e)}")
            
        return relevant_queries
    
    def _is_query_relevant(self, user_query: str, stored_query: str, threshold: float = 0.3) -> bool:
        """
        Determine if a stored query is relevant to the current user query
        
        Args:
            user_query: The current user query
            stored_query: A stored query to check for relevance
            threshold: Minimum similarity threshold
            
        Returns:
            True if the stored query is relevant
        """
        # Simple keyword-based relevance check
        # In a real implementation, this would use semantic similarity
        
        # Normalize and tokenize both queries
        user_tokens = set(re.findall(r'\b\w+\b', user_query.lower()))
        stored_tokens = set(re.findall(r'\b\w+\b', stored_query.lower()))
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "in", "on", "at", "is", "are", "what", "how", "when", "where", "why", "who", "which"}
        user_tokens = user_tokens - stop_words
        stored_tokens = stored_tokens - stop_words
        
        # Calculate token overlap ratio
        if not user_tokens or not stored_tokens:
            return False
            
        # Find common tokens
        common_tokens = user_tokens.intersection(stored_tokens)
        
        # Calculate Jaccard similarity
        similarity = len(common_tokens) / len(user_tokens.union(stored_tokens))
        
        return similarity >= threshold
    
    def _generate_hyde_document(self, query: str) -> str:
        """
        Generate a hypothetical document that would answer the query
        
        Args:
            query: The user query
            
        Returns:
            Hypothetical document text
        """
        if not self.hyde_enabled:
            return ""
            
        prompt = f"""Generate a detailed passage that directly answers the following question. 
        Include specific information that would be helpful and relevant.
        
        Question: {query}
        
        Detailed answer:"""
        
        try:
            if self.llm_provider == "openai":
                response = self.client.chat.completions.create(
                    model=os.getenv("DEFAULT_MODEL", "gpt-4o-2024-11-20"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,  # Low temperature for more factual responses
                    max_tokens=500
                )
                return response.choices[0].message.content
                
            elif self.llm_provider == "anthropic":
                response = self.client.messages.create(
                    model=os.getenv("DEFAULT_MODEL", "claude-3-opus-20240229"),
                    max_tokens=500,
                    temperature=0.2,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
                
            return ""
            
        except Exception as e:
            logger.error(f"Error in HyDE generation: {str(e)}")
            return ""