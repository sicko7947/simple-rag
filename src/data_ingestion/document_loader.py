"""
Document loader module for handling various file types
"""
import os
import logging
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import (
    PyPDFLoader, 
    UnstructuredHTMLLoader,
    TextLoader,
    UnstructuredImageLoader
)
from langchain.schema import Document

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Handles loading and processing of various document types"""
    
    def __init__(self):
        self.supported_extensions = {
            ".pdf": self._load_pdf,
            ".html": self._load_html,
            ".txt": self._load_text,
            ".jpg": self._load_image,
            ".jpeg": self._load_image,
            ".png": self._load_image
        }
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load document from file path and return list of Document objects
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {ext}")
        
        logger.info(f"Loading document: {file_path}")
        return self.supported_extensions[ext](file_path)
    
    def _load_pdf(self, file_path: str) -> List[Document]:
        """Load PDF document"""
        loader = PyPDFLoader(file_path)
        return loader.load()
    
    def _load_html(self, file_path: str) -> List[Document]:
        """Load HTML document"""
        loader = UnstructuredHTMLLoader(file_path)
        return loader.load()
    
    def _load_text(self, file_path: str) -> List[Document]:
        """Load text document"""
        loader = TextLoader(file_path)
        return loader.load()
    
    def _load_image(self, file_path: str) -> List[Document]:
        """Load and process image document using OCR"""
        loader = UnstructuredImageLoader(file_path)
        return loader.load()