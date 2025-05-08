"""
Metadata extractor module for extracting metadata from documents
"""
import os
import logging
from datetime import datetime
from typing import Dict, Any, List
from langchain.schema import Document
import hashlib

logger = logging.getLogger(__name__)

class MetadataExtractor:
    """Extracts metadata from documents"""
    
    def extract_metadata(self, documents: List[Document], file_path: str) -> List[Document]:
        """
        Extract metadata from documents
        
        Args:
            documents: List of Document objects
            file_path: Path to the original document file
            
        Returns:
            List of Document objects with metadata
        """
        file_name = os.path.basename(file_path)
        file_extension = os.path.splitext(file_name)[1].lower()
        file_size = os.path.getsize(file_path)
        creation_time = os.path.getctime(file_path)
        modification_time = os.path.getmtime(file_path)
        
        # Basic metadata common to all document types
        base_metadata = {
            "file_name": file_name,
            "file_extension": file_extension,
            "file_size": file_size,
            "file_path": file_path,
            "creation_date": datetime.fromtimestamp(creation_time).isoformat(),
            "modification_date": datetime.fromtimestamp(modification_time).isoformat(),
            "ingestion_date": datetime.now().isoformat(),
            "confidence_score": 1.0,  # Default confidence score
        }
        
        # Add document-type specific metadata extraction
        if file_extension == ".pdf":
            self._enhance_pdf_metadata(documents, base_metadata)
        elif file_extension in [".jpg", ".jpeg", ".png"]:
            self._enhance_image_metadata(documents, base_metadata)
        
        # Generate document ID for each chunk
        for i, doc in enumerate(documents):
            # Combine document content and metadata for unique hash
            content_hash = hashlib.md5((doc.page_content + str(i)).encode()).hexdigest()
            doc.metadata.update(base_metadata)
            doc.metadata["document_id"] = content_hash
            
        return documents
    
    def _enhance_pdf_metadata(self, documents: List[Document], base_metadata: Dict[str, Any]) -> None:
        """Add PDF-specific metadata to documents"""
        # Add page numbers to each document chunk
        for i, doc in enumerate(documents):
            if "page" not in doc.metadata:
                doc.metadata["page"] = i + 1  # Start page numbers from 1
    
    def _enhance_image_metadata(self, documents: List[Document], base_metadata: Dict[str, Any]) -> None:
        """Add image-specific metadata to documents"""
        # For images, we typically get a single document with OCR text
        if documents:
            base_metadata["content_type"] = "image_ocr"
            base_metadata["confidence_score"] = 0.8  # OCR has lower confidence