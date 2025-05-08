"""
Document processor module for end-to-end document processing
"""
import logging
import os
import re
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .document_loader import DocumentLoader
from .metadata_extractor import MetadataExtractor
from src.vector_store.embedding import EmbeddingGenerator
from src.language_model.model_interface import ModelInterface

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Processes documents from loading to chunking and embedding"""
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        hyde_enabled: bool = False,
        extract_authors: bool = True,
        extract_categories: bool = True,
        max_hyde_queries: int = 3
    ):
        self.document_loader = DocumentLoader()
        self.metadata_extractor = MetadataExtractor()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        self.embedding_generator = EmbeddingGenerator()
        self.hyde_enabled = hyde_enabled and os.getenv("ENABLE_HYDE", "true").lower() == "true"
        self.extract_authors = extract_authors
        self.extract_categories = extract_categories
        self.max_hyde_queries = max_hyde_queries
        
        # Initialize model interface only if HyDE is enabled
        if self.hyde_enabled:
            try:
                self.model_interface = ModelInterface()
            except Exception as e:
                logger.warning(f"Failed to initialize ModelInterface for HyDE: {str(e)}")
                self.hyde_enabled = False
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process document from file path to chunked documents with embeddings
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with processed documents and their embeddings
        """
        logger.info(f"Processing document: {file_path}")
        
        # Step 1: Load document
        raw_documents = self.document_loader.load_document(file_path)
        logger.info(f"Loaded {len(raw_documents)} document chunks")
        
        # Step 2: Extract metadata
        documents_with_metadata = self.metadata_extractor.extract_metadata(raw_documents, file_path)
        
        # Step 3: Enhance metadata with additional extraction
        if self.extract_authors or self.extract_categories:
            documents_with_metadata = self._enhance_metadata(documents_with_metadata)
        
        # Step 4: Split documents into chunks
        chunked_documents = self.text_splitter.split_documents(documents_with_metadata)
        logger.info(f"Split into {len(chunked_documents)} chunks")
        
        # Step 5: Generate embeddings for chunks
        documents_with_embeddings = self.embedding_generator.generate_embeddings(chunked_documents)
        
        # Step 6: Optionally generate HyDE queries
        if self.hyde_enabled:
            hyde_results = self._generate_hyde_queries(chunked_documents)
            
        return {
            "documents": chunked_documents,
            "embeddings": documents_with_embeddings,
            "hyde_queries": hyde_results if self.hyde_enabled else None,
            "metadata": {
                "source": file_path,
                "chunk_count": len(chunked_documents),
                "processing_date": datetime.now().isoformat(),
                "has_hyde": self.hyde_enabled
            }
        }
    
    def _enhance_metadata(self, documents: List[Document]) -> List[Document]:
        """
        Enhance document metadata with additional extraction
        
        Args:
            documents: List of Document objects
            
        Returns:
            Documents with enhanced metadata
        """
        for doc in documents:
            text_content = doc.page_content
            
            # Extract potential authors
            if self.extract_authors and "author" not in doc.metadata:
                authors = self._extract_authors(text_content)
                if authors:
                    doc.metadata["author"] = ", ".join(authors[:3])  # Limit to 3 authors
            
            # Extract potential categories/topics
            if self.extract_categories and "categories" not in doc.metadata:
                categories = self._extract_categories(text_content)
                if categories:
                    doc.metadata["categories"] = categories
                    # Add primary category if available
                    if categories:
                        doc.metadata["primary_category"] = categories[0]
            
            # Update confidence score based on metadata quality
            metadata_quality = self._calculate_metadata_quality(doc.metadata)
            # Blend with existing confidence score or set new one
            current_confidence = doc.metadata.get("confidence_score", 0.7)
            doc.metadata["confidence_score"] = (current_confidence * 0.7) + (metadata_quality * 0.3)
            
        return documents
    
    def _extract_authors(self, text: str) -> List[str]:
        """Extract potential author names from text"""
        # Simple pattern matching for author extraction
        # In a real implementation, this would use NER or more sophisticated techniques
        author_patterns = [
            r"(?:Author|Written by|By)[:\s]+([A-Z][a-z]+(?: [A-Z][a-z]+){1,3})",
            r"(?:©|Copyright)(?:\s+\d{4})?\s+by\s+([A-Z][a-z]+(?: [A-Z][a-z]+){1,3})"
        ]
        
        potential_authors = []
        for pattern in author_patterns:
            matches = re.findall(pattern, text)
            if matches:
                potential_authors.extend(matches)
                
        return list(set(potential_authors))  # Deduplicate
    
    def _extract_categories(self, text: str) -> List[str]:
        """Extract potential categories or topics from text"""
        # This would typically use topic modeling or keyword extraction
        # For simplicity, we'll use a basic approach
        
        # Common categories by domain
        domain_categories = {
            "finance": ["banking", "investment", "trading", "financial", "budget", "economy"],
            "technology": ["software", "hardware", "database", "cloud", "AI", "programming"],
            "healthcare": ["medical", "patient", "doctor", "treatment", "diagnosis", "health"],
            "legal": ["contract", "agreement", "compliance", "regulation", "legal", "law"]
        }
        
        # Check frequency of category keywords
        categories = []
        text_lower = text.lower()
        
        for domain, keywords in domain_categories.items():
            matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
            if matches >= 2:  # Threshold for category relevance
                categories.append(domain)
                
        return categories
    
    def _calculate_metadata_quality(self, metadata: Dict[str, Any]) -> float:
        """Calculate a quality score for metadata completeness"""
        # Key metadata fields that indicate quality
        quality_fields = ["author", "creation_date", "title", "categories"]
        
        # Count how many quality fields are present
        present_fields = sum(1 for field in quality_fields if field in metadata)
        
        # Calculate quality score from 0.5 to 1.0
        quality_score = 0.5 + (0.5 * present_fields / len(quality_fields))
        
        return quality_score
    
    def _generate_hyde_queries(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Generate hypothetical document embeddings (HyDE) for documents
        
        Args:
            documents: List of Document objects
            
        Returns:
            Dictionary with HyDE results
        """
        if not self.hyde_enabled:
            return {"enabled": False}
            
        logger.info("Generating HyDE queries for documents")
        
        # Group documents by topic/category for more coherent HyDE generation
        document_groups = self._group_documents_by_topic(documents)
        
        hyde_results = {
            "enabled": True,
            "query_count": 0,
            "queries": []
        }
        
        try:
            # For each group, generate representative queries
            for group_name, group_docs in document_groups.items():
                # Skip if we've reached the maximum number of queries
                if hyde_results["query_count"] >= self.max_hyde_queries:
                    break
                    
                # Select a representative document from the group
                if group_docs:
                    representative_doc = group_docs[0]
                    
                    # Create a prompt for generating queries
                    prompt = f"""Generate a natural question that someone might ask where the following 
                    text would be the perfect answer. Focus on the main points and be specific:
                    
                    Text: {representative_doc.page_content[:500]}...
                    
                    Question:"""
                    
                    # Generate a hypothetical question using the LLM
                    messages = [{"role": "user", "content": prompt}]
                    response = self.model_interface.generate_response(
                        messages=messages,
                        temperature=0.7,
                        max_tokens=100
                    )
                    
                    if response["success"]:
                        # Add the generated query
                        query = response["content"].strip()
                        hyde_results["queries"].append({
                            "query": query,
                            "document_id": representative_doc.metadata.get("document_id", ""),
                            "group": group_name
                        })
                        hyde_results["query_count"] += 1
                        
        except Exception as e:
            logger.error(f"Error generating HyDE queries: {str(e)}")
            
        logger.info(f"Generated {hyde_results['query_count']} HyDE queries")
        return hyde_results
    
    def _group_documents_by_topic(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """Group documents by topic or category for coherent HyDE generation"""
        # Initialize groups
        groups = {}
        
        for doc in documents:
            # Try to group by category first
            if "primary_category" in doc.metadata:
                group_key = doc.metadata["primary_category"]
            elif "categories" in doc.metadata and doc.metadata["categories"]:
                group_key = doc.metadata["categories"][0]
            else:
                # Fall back to basic grouping by source
                group_key = doc.metadata.get("file_name", "unknown")
                
            if group_key not in groups:
                groups[group_key] = []
                
            groups[group_key].append(doc)
            
        return groups