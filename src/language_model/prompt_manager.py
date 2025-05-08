"""
Prompt manager module for creating structured prompts for language models
"""
import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class PromptManager:
    """Manages prompt templates and generation for language models"""
    
    def __init__(self):
        # Load default system prompts
        self.default_rag_system_prompt = (
            "You are a helpful AI assistant that answers questions based on provided context. "
            "Be accurate, helpful, concise, and only respond based on the provided context. "
            "If the context doesn't contain the information needed, say you don't know rather than making up an answer. "
            "Always cite your sources when providing information by adding a citation marker like [Doc1], [Doc2], etc. "
            "at the end of sentences or paragraphs that reference information from that source."
        )
        
        self.default_chat_system_prompt = (
            "You are a helpful AI assistant. Respond to the user in a helpful, accurate, "
            "and thoughtful manner. When you don't know the answer, admit it rather than "
            "making up information."
        )
        
    def create_rag_prompt(
        self, 
        query: str, 
        context: Dict[str, Any],
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Create a prompt for RAG-based question answering
        
        Args:
            query: User query
            context: Context information including formatted_context and sources
            chat_history: Optional conversation history
            
        Returns:
            Prompt dictionary with messages and metadata
        """
        logger.info("Creating RAG prompt")
        
        formatted_context = context.get("formatted_context", "")
        sources = context.get("sources", [])
        
        # Start with system message
        system_prompt = self.default_rag_system_prompt
        
        # Add source citation instructions
        system_prompt += "\n\nSOURCE DOCUMENTS:"
        for i, source in enumerate(sources):
            doc_id = f"Doc{i+1}"
            source_name = source.get("file_name", "Unknown")
            source_page = source.get("page", "")
            source_info = f"{source_name}"
            if source_page:
                source_info += f", Page {source_page}"
            system_prompt += f"\n{doc_id}: {source_info}"
            
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add chat history if available
        if chat_history:
            messages.extend(chat_history)
        
        # Construct the user message with context and query
        user_message = f"""Answer my question based on the following context:

CONTEXT:
{formatted_context}

QUESTION:
{query}

Please provide a detailed answer with relevant information from the context. If the information is not in the context, let me know you don't have enough information to answer. Include citation markers like [Doc1] after statements that reference information from a specific source.
"""
        
        messages.append({"role": "user", "content": user_message})
        
        # Return the final prompt
        return {
            "messages": messages,
            "metadata": {
                "context_length": len(formatted_context),
                "document_count": context.get("document_count", 0),
                "has_history": bool(chat_history),
                "source_count": len(sources)
            }
        }
        
    def create_hyde_prompt(self, query: str) -> Dict[str, Any]:
        """
        Create a prompt for generating a hypothetical document (HyDE)
        
        Args:
            query: User query
            
        Returns:
            Prompt dictionary with messages
        """
        system_prompt = (
            "You are an expert document generator. Given a question, generate a detailed passage "
            "that would serve as a perfect answer to the question. Include specific information, "
            "facts, and details that would be helpful and relevant."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {query}\n\nGenerate a detailed passage that answers this question:"}
        ]
        
        return {
            "messages": messages,
            "metadata": {
                "purpose": "hyde",
                "original_query": query
            }
        }
        
    def create_conflict_check_prompt(
        self,
        texts: List[str]
    ) -> Dict[str, Any]:
        """
        Create a prompt for checking conflicts between different texts
        
        Args:
            texts: List of text passages to check for conflicts
            
        Returns:
            Prompt dictionary with messages
        """
        system_prompt = (
            "You are a logical analysis expert. Your task is to identify factual or logical conflicts "
            "between different passages of text. Focus only on clear contradictions in facts, figures, "
            "dates, or logical reasoning. Minor differences in perspective or emphasis are not conflicts."
        )
        
        formatted_texts = "\n\n".join([f"PASSAGE {i+1}:\n{text}" for i, text in enumerate(texts)])
        
        user_message = f"""Analyze the following passages for factual or logical conflicts:

{formatted_texts}

Identify any direct contradictions between these passages. If you find conflicts, explain each conflict clearly. 
If there are no conflicts, state that the passages are consistent with each other.
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        return {
            "messages": messages,
            "metadata": {
                "purpose": "conflict_check",
                "passage_count": len(texts)
            }
        }
        
    def create_fact_check_prompt(
        self,
        response: str,
        context_docs: List[str]
    ) -> Dict[str, Any]:
        """
        Create a prompt for fact-checking a response against source documents
        
        Args:
            response: Generated response to fact-check
            context_docs: List of context documents used to generate the response
            
        Returns:
            Prompt dictionary with messages
        """
        system_prompt = (
            "You are a fact-checking expert. Your task is to verify that statements in the RESPONSE "
            "are supported by the provided SOURCE DOCUMENTS. Identify any claims in the response that "
            "are not supported by or contradict the sources. Focus on factual accuracy only."
        )
        
        formatted_sources = "\n\n".join([f"SOURCE {i+1}:\n{doc}" for i, doc in enumerate(context_docs)])
        
        user_message = f"""Fact-check the following response against the source documents:

RESPONSE:
{response}

SOURCE DOCUMENTS:
{formatted_sources}

Verify that the information in the response is accurate and supported by the source documents.
List any claims in the response that are:
1. Not supported by the sources
2. Contradicted by the sources
3. Adequately supported by the sources

Explain your reasoning for each assessment.
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        return {
            "messages": messages,
            "metadata": {
                "purpose": "fact_check",
                "source_count": len(context_docs)
            }
        }
        
    def create_custom_prompt(
        self,
        system_prompt: str,
        user_message: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Create a custom prompt with specified system and user messages
        
        Args:
            system_prompt: System instruction
            user_message: User message
            chat_history: Optional conversation history
            
        Returns:
            Prompt dictionary with messages
        """
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add chat history if available
        if chat_history:
            messages.extend(chat_history)
            
        messages.append({"role": "user", "content": user_message})
        
        return {
            "messages": messages,
            "metadata": {
                "custom": True,
                "has_history": bool(chat_history)
            }
        }