"""
Embedding module for generating vector embeddings of documents
"""
import os
import logging
from typing import List, Dict, Any
import numpy as np
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings  # Updated import from langchain_openai

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generates embeddings for documents using OpenAI embeddings"""
    
    def __init__(self):
        self.model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables. Using dummy embeddings.")
            self.use_dummy_embeddings = True
            self.embedding_engine = None
        else:
            self.use_dummy_embeddings = False
            try:
                self.embedding_engine = OpenAIEmbeddings(api_key=api_key, model=self.model)
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI embeddings: {str(e)}")
                logger.warning("Falling back to dummy embeddings")
                self.use_dummy_embeddings = True
                self.embedding_engine = None
    
    def generate_embeddings(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Generate embeddings for a list of documents
        
        Args:
            documents: List of Document objects
            
        Returns:
            Dictionary mapping document IDs to their embeddings
        """
        if self.use_dummy_embeddings:
            return self._generate_dummy_embeddings(documents)
        
        logger.info(f"Generating embeddings for {len(documents)} documents using {self.model}")
        
        try:
            texts = [doc.page_content for doc in documents]
            embeddings = self.embedding_engine.embed_documents(texts)
            
            # Create a mapping of document IDs to embeddings
            document_embeddings = {}
            for i, doc in enumerate(documents):
                document_id = doc.metadata.get("document_id", f"doc_{i}")
                document_embeddings[document_id] = embeddings[i]
                
            return document_embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            logger.warning("Falling back to dummy embeddings")
            return self._generate_dummy_embeddings(documents)
    
    def _generate_dummy_embeddings(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Generate deterministic dummy embeddings when API is not available
        
        Args:
            documents: List of Document objects
            
        Returns:
            Dictionary mapping document IDs to dummy embeddings
        """
        logger.info(f"Generating dummy embeddings for {len(documents)} documents")
        
        document_embeddings = {}
        dimension = 1536  # Standard OpenAI embedding dimension
        
        for i, doc in enumerate(documents):
            document_id = doc.metadata.get("document_id", f"doc_{i}")
            
            # Create a deterministic embedding based on document content
            # We hash the content to get a seed for our dummy embedding
            import hashlib
            content_hash = int(hashlib.md5(doc.page_content.encode()).hexdigest(), 16)
            np.random.seed(content_hash)
            
            # Generate a random embedding vector
            dummy_embedding = np.random.rand(dimension).tolist()
            
            document_embeddings[document_id] = dummy_embedding
            
        logger.info("Dummy embeddings generated successfully")
        return document_embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query string
        
        Args:
            query: Query string to embed
            
        Returns:
            Embedding vector
        """
        if self.use_dummy_embeddings:
            # Generate a deterministic dummy embedding for the query
            import hashlib
            query_hash = int(hashlib.md5(query.encode()).hexdigest(), 16)
            np.random.seed(query_hash)
            
            dimension = 1536  # Standard OpenAI embedding dimension
            return np.random.rand(dimension).tolist()
        
        try:
            return self.embedding_engine.embed_query(query)
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            logger.warning("Falling back to dummy query embedding")
            
            # Generate a deterministic dummy embedding for the query
            import hashlib
            query_hash = int(hashlib.md5(query.encode()).hexdigest(), 16)
            np.random.seed(query_hash)
            
            dimension = 1536  # Standard OpenAI embedding dimension
            return np.random.rand(dimension).tolist()