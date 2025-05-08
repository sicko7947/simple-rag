"""
Context builder for constructing prompt context from retrieved documents
"""
import logging
from typing import List, Dict, Any, Optional
from langchain.schema import Document

logger = logging.getLogger(__name__)

class ContextBuilder:
    """Builds context for language model prompts from retrieved documents"""
    
    def __init__(
        self,
        max_tokens: int = 3000,
        use_metadata: bool = True,
        window_size: int = 5,  # Number of documents per window
        enable_sliding_window: bool = True
    ):
        self.max_tokens = max_tokens
        self.use_metadata = use_metadata
        self.window_size = window_size
        self.enable_sliding_window = enable_sliding_window
        
    def build_context(
        self, 
        documents: List[Document],
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build context from retrieved documents
        
        Args:
            documents: List of retrieved Document objects
            query: Optional query to prioritize relevant content
            
        Returns:
            Dictionary with formatted context and metadata
        """
        logger.info(f"Building context from {len(documents)} documents")
        
        # Sort documents by confidence score if available
        sorted_docs = sorted(
            documents,
            key=lambda x: x.metadata.get("confidence_score", 0.0),
            reverse=True
        )
        
        # Apply sliding window if enabled and needed
        if self.enable_sliding_window and len(sorted_docs) > self.window_size:
            windowed_docs = self._apply_sliding_window(sorted_docs, query)
        else:
            windowed_docs = sorted_docs
        
        # Track token usage
        current_tokens = 0
        included_docs = []
        
        for doc in windowed_docs:
            # Estimate token count (rough approximation)
            doc_tokens = len(doc.page_content.split()) * 1.3  # ~1.3 tokens per word
            
            if current_tokens + doc_tokens <= self.max_tokens:
                included_docs.append(doc)
                current_tokens += doc_tokens
            else:
                # Try to include a truncated version of the document
                remaining_tokens = self.max_tokens - current_tokens
                if remaining_tokens > 200:  # Only if we can include something substantial
                    truncated_content = self._truncate_content(doc.page_content, remaining_tokens)
                    truncated_doc = Document(
                        page_content=truncated_content,
                        metadata=doc.metadata
                    )
                    included_docs.append(truncated_doc)
                    
                logger.info(f"Context limit reached after {len(included_docs)} documents")
                break
                
        # Construct formatted context
        context_parts = []
        sources = []
        
        for i, doc in enumerate(included_docs):
            # Format document content with optional metadata
            doc_text = doc.page_content.strip()
            
            if self.use_metadata:
                # Include relevant metadata in the context
                source_info = self._format_source_info(doc.metadata)
                context_parts.append(f"[Document {i+1}] {source_info}\n{doc_text}")
            else:
                context_parts.append(f"[Document {i+1}]\n{doc_text}")
            
            # Track source information for citation
            sources.append({
                "id": doc.metadata.get("document_id", f"doc_{i}"),
                "file_name": doc.metadata.get("file_name", "Unknown"),
                "page": doc.metadata.get("page", None),
                "confidence": doc.metadata.get("confidence_score", 1.0),
                "creation_date": doc.metadata.get("creation_date", None)
            })
            
        formatted_context = "\n\n".join(context_parts)
        
        return {
            "formatted_context": formatted_context,
            "sources": sources,
            "document_count": len(included_docs),
            "token_estimate": current_tokens,
            "has_truncated_docs": len(included_docs) < len(sorted_docs)
        }
    
    def _apply_sliding_window(
        self, 
        documents: List[Document],
        query: Optional[str] = None
    ) -> List[Document]:
        """
        Apply sliding window to balance document coverage
        
        Args:
            documents: List of Document objects
            query: Optional query to prioritize relevant content
            
        Returns:
            Documents organized in a sliding window
        """
        # If no query or short document list, return as is
        if not query or len(documents) <= self.window_size:
            return documents
            
        # Create a balanced selection:
        # - First few high confidence docs
        # - Some mid-range docs
        # - Last few docs for diversity
        
        high_priority = documents[:min(3, len(documents))]
        mid_priority = documents[3:max(3, len(documents) - 2)]
        low_priority = documents[max(0, len(documents) - 2):]
        
        # Select a portion from the mid range
        mid_count = min(self.window_size - len(high_priority) - len(low_priority), len(mid_priority))
        if mid_count > 0:
            mid_selection = mid_priority[:mid_count]
        else:
            mid_selection = []
            
        # Combine the selections
        return high_priority + mid_selection + low_priority
        
    def _format_source_info(self, metadata: Dict[str, Any]) -> str:
        """Format source information from metadata"""
        parts = []
        
        if "file_name" in metadata:
            parts.append(f"Source: {metadata['file_name']}")
            
        if "page" in metadata and metadata["page"] is not None:
            parts.append(f"Page: {metadata['page']}")
            
        if "creation_date" in metadata and metadata["creation_date"]:
            # Format date nicely
            try:
                date = metadata["creation_date"].split("T")[0]
                parts.append(f"Date: {date}")
            except (AttributeError, IndexError):
                pass
                
        if "author" in metadata and metadata["author"]:
            parts.append(f"Author: {metadata['author']}")
            
        if "confidence_score" in metadata:
            confidence = float(metadata["confidence_score"])
            if confidence < 0.7:
                parts.append("Confidence: Low")
            elif confidence < 0.9:
                parts.append("Confidence: Medium")
            else:
                parts.append("Confidence: High")
                
        return " | ".join(parts)
        
    def _truncate_content(self, content: str, token_limit: int) -> str:
        """
        Truncate content to fit within token limit
        
        Args:
            content: Original content text
            token_limit: Approximate token limit
            
        Returns:
            Truncated content
        """
        # Rough approximation: 1 token ≈ 0.75 words
        word_limit = int(token_limit * 0.75)
        words = content.split()
        
        if len(words) <= word_limit:
            return content
            
        truncated_words = words[:word_limit]
        return " ".join(truncated_words) + "... [truncated]"