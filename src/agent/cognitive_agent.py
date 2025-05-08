"""
Cognitive agent module for coordinating RAG components with cognitive abilities
"""
import os
import logging
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from langchain.schema import Document

from src.data_ingestion.document_processor import DocumentProcessor
from src.vector_store.supabase_store import SupabaseVectorStore
from src.vector_store.supabase_store import SupabaseVectorStore
from src.retriever.semantic_retriever import SemanticRetriever
from src.retriever.query_expansion import QueryExpander
from src.retriever.guardrails import GuardrailsValidator
from src.language_model.context_builder import ContextBuilder
from src.language_model.prompt_manager import PromptManager
from src.language_model.model_interface import ModelInterface
from .knowledge_state import KnowledgeState

logger = logging.getLogger(__name__)

class CognitiveAgent:
    """Intelligent agent with cognitive abilities for RAG"""
    
    def __init__(self):
        # Initialize component modules
        self.document_processor = DocumentProcessor()
        # Use the factory to create the appropriate vector store
        self.vector_store = SupabaseVectorStore()
        self.semantic_retriever = SemanticRetriever()
        self.query_expander = QueryExpander()
        self.guardrails = GuardrailsValidator()
        self.context_builder = ContextBuilder()
        self.prompt_manager = PromptManager()
        self.model_interface = ModelInterface()
        self.knowledge_state = KnowledgeState()
        
        # Track conversation history
        self.conversation_history = []
    
    def process_document(self, file_path: str, file_content: bytes) -> Dict[str, Any]:
        """
        Process and ingest a document
        
        Args:
            file_path: Path to the document
            file_content: Binary content of the file
            
        Returns:
            Processing results
        """
        logger.info(f"Processing document: {file_path}")
        
        # Store the file in Supabase
        stored_path = self.vector_store.store_file(file_path, file_content)
        
        # Process the document (load, chunk, embed)
        # We need to temporarily save the file locally
        import os
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_path)[1]) as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name
            
        try:
            # Process the document
            processed = self.document_processor.process_document(temp_path)
            
            # Store documents and embeddings in vector database
            stored_ids = self.vector_store.store_documents(
                processed["documents"],
                processed["embeddings"]
            )
            
            if processed.get("hyde_queries") and processed["hyde_queries"].get("enabled"):
                for doc_id in stored_ids:
                    self.vector_store.store_hyde_queries(doc_id, processed["hyde_queries"])
            
            result = {
                "file_path": stored_path,
                "document_count": len(processed["documents"]),
                "stored_ids": stored_ids,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            result = {
                "file_path": stored_path,
                "error": str(e),
                "success": False
            }
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        return result
    
    def _process_response_citations(self, response_content: str, sources: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process citation markers in the response and extract citation information
        
        Args:
            response_content: The raw response content from the model
            sources: List of source documents
            
        Returns:
            Tuple of (processed_response, citations)
        """
        # Look for citation patterns like [Doc1], [Doc2], etc.
        citation_pattern = r'\[Doc(\d+)\]'
        citations = []
        
        # Extract all citations
        matches = re.finditer(citation_pattern, response_content)
        for match in matches:
            doc_num = int(match.group(1))
            if 1 <= doc_num <= len(sources):
                source = sources[doc_num-1]
                citation = {
                    "text": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                    "source_id": source.get("id", f"doc_{doc_num-1}"),
                    "file_name": source.get("file_name", "Unknown"),
                    "page": source.get("page", None)
                }
                citations.append(citation)
        
        # Sort citations by position
        citations.sort(key=lambda x: x["start"])
        
        # Return original response and citations
        return response_content, citations
    
    def answer_query(
        self, 
        query: str, 
        conversation_id: Optional[str] = None,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Answer a user query using the RAG pipeline
        
        Args:
            query: The user query
            conversation_id: Optional conversation ID for history
            filter_criteria: Optional filters for retrieval
            
        Returns:
            Response dictionary
        """
        logger.info(f"Processing query: {query}")
        
        # Step 1: Query expansion
        expanded_query = self.query_expander.expand_query(query)
        
        # Step 2: Retrieve relevant documents
        retrieved_docs = self.semantic_retriever.retrieve(
            query=query,
            filter_criteria=filter_criteria
        )
        
        # Step 3: Apply guardrails
        validated_docs, validation_meta = self.guardrails.validate_context(retrieved_docs)
        
        # Step 4: Update knowledge state
        self.knowledge_state.update_from_context(
            [doc.metadata for doc in validated_docs], 
            query
        )
        
        # Step 5: Build context from validated documents
        context = self.context_builder.build_context(validated_docs)
        
        # Step 6: Create prompt with context
        prompt = self.prompt_manager.create_rag_prompt(
            query=query,
            context=context,
            chat_history=self.conversation_history[-5:] if self.conversation_history else None
        )
        
        # Step 7: Generate response
        response = self.model_interface.generate_response(
            messages=prompt["messages"]
        )
        
        # Step 8: Update conversation history
        self.conversation_history.append({
            "role": "user",
            "content": query
        })
        
        processed_response = ""
        citations = []
        
        if response["success"]:
            # Process citations in the response
            processed_response, citations = self._process_response_citations(
                response["content"],
                context["sources"]
            )
            
            self.conversation_history.append({
                "role": "assistant",
                "content": processed_response
            })
            
            # Update knowledge state based on response
            self.knowledge_state.update_from_response(
                processed_response, 
                context["sources"]
            )
        
        # Step 9: Apply fact-checking to response
        fact_check_results = None
        if response["success"]:
            fact_check_results = self.guardrails.fact_check_response(
                processed_response,
                validated_docs
            )
        
        # Step 10: Prepare final response with metadata
        result = {
            "query": query,
            "response": processed_response if response["success"] else "I encountered an error.",
            "sources": context["sources"],
            "citations": citations,  # Include the citations in the response
            "cognitive_state": self.knowledge_state.get_state(),
            "fact_check": fact_check_results,
            "success": response["success"]
        }
        
        logger.info(f"Query processed successfully: {query[:50]}...")
        return result
        
    def get_agent_state(self) -> Dict[str, Any]:
        """Get the current state of the agent"""
        return {
            "knowledge_state": self.knowledge_state.get_state(),
            "conversation_length": len(self.conversation_history) // 2,  # Pairs of messages
        }
        
    def reset_state(self) -> None:
        """Reset the agent's state"""
        self.knowledge_state = KnowledgeState()
        self.conversation_history = []