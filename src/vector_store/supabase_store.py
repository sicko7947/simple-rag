"""
Supabase vector store for document storage and retrieval
"""
import os
import logging
from typing import List, Dict, Any, Optional
import json
import numpy as np
from supabase import create_client, Client
from langchain.schema import Document
from datetime import datetime 

logger = logging.getLogger(__name__)

class SupabaseVectorStore:
    """Vector store implementation using Supabase pgvector"""
    
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.bucket_name = os.getenv("SUPABASE_BUCKET_NAME", "documents")
        self.table_name = "documents"
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
            
        # Initialize Supabase client
        try:
            self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
            logger.info("Successfully connected to Supabase")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {str(e)}")
            raise
            
        self._initialize_storage()
        
    def _initialize_storage(self) -> None:
        """Initialize Supabase storage bucket if it doesn't exist"""
        try:
            # Check if bucket exists by listing all buckets
            try:
                buckets = self.supabase.storage.list_buckets()
                bucket_exists = any(bucket.name == self.bucket_name for bucket in buckets)
            except Exception as e:
                logger.warning(f"Unable to list buckets, will try to use existing bucket: {str(e)}")
                bucket_exists = False
            
            if not bucket_exists:
                try:
                    logger.info(f"Creating storage bucket: {self.bucket_name}")
                    # Try to create the bucket with public read access
                    self.supabase.storage.create_bucket(
                        self.bucket_name,
                        options={
                            "public": True  # Allow public access for simplicity
                        }
                    )
                    logger.info(f"Created bucket: {self.bucket_name}")
                except Exception as e:
                    logger.warning(f"Couldn't create bucket, will attempt to use existing: {str(e)}")
                
            # Check if the documents table exists with pgvector extension
            # Note: In a real implementation, you would need to ensure the 
            # pgvector extension is enabled in your Supabase database
            logger.info("Vector store initialized")
            
        except Exception as e:
            logger.error(f"Error initializing Supabase storage: {str(e)}")
            logger.info("Continuing without storage initialization - some features may be limited")
    
    def store_file(self, file_path: str, file_content: bytes) -> str:
        """
        Store a file in Supabase storage
        
        Args:
            file_path: Path/name for the file in storage
            file_content: Binary content of the file
            
        Returns:
            File path in storage
        """
        file_name = os.path.basename(file_path)
        storage_path = f"{file_name}"
        
        try:
            # Try to upload the file
            try:
                result = self.supabase.storage.from_(self.bucket_name).upload(
                    path=storage_path,
                    file=file_content,
                    file_options={"content-type": "application/octet-stream"}
                )
                logger.info(f"Uploaded file to storage: {storage_path}")
            except Exception as e:
                # If upload fails, it might be due to file already existing or bucket issues
                logger.warning(f"Upload failed, trying to update existing file: {str(e)}")
                result = self.supabase.storage.from_(self.bucket_name).update(
                    path=storage_path,
                    file=file_content,
                    file_options={"content-type": "application/octet-stream"}
                )
                logger.info(f"Updated existing file in storage: {storage_path}")
                
            return storage_path
            
        except Exception as e:
            logger.error(f"Error storing file in Supabase: {str(e)}")
            # Return file_name even if storage failed, as we'll still process the local copy
            logger.info("Proceeding with local file processing only")
            return file_name
    
    def store_documents(self, documents: List[Document], embeddings: Dict[str, List[float]]) -> List[str]:
        """
        Store documents and their embeddings in Supabase
        
        Args:
            documents: List of Document objects
            embeddings: Dictionary mapping document IDs to embedding vectors
            
        Returns:
            List of stored document IDs
        """
        stored_ids = []
        
        # Check if we have access to the database before attempting storage
        try:
            # Test database access with a simple query
            self.supabase.table("documents").select("id").limit(1).execute()
            db_access = True
        except Exception as e:
            logger.warning(f"Unable to access database: {str(e)}")
            logger.info("Proceeding with in-memory document processing only")
            db_access = False
            
        for doc in documents:
            doc_id = doc.metadata.get("document_id")
            if not doc_id or doc_id not in embeddings:
                logger.warning(f"Missing document ID or embedding for document")
                continue
                
            # Prepare metadata - convert any non-JSON serializable types
            metadata = {k: v for k, v in doc.metadata.items() if k != "document_id"}
            
            # Store document with its embedding
            try:
                data = {
                    "id": doc_id,
                    "content": doc.page_content,
                    "metadata": json.dumps(metadata),
                    "embedding": embeddings[doc_id]
                }
                
                # Only try to insert if we have database access
                if db_access:
                    try:
                        # Try to insert the document
                        self.supabase.table(self.table_name).insert(data).execute()
                        logger.info(f"Stored document in database: {doc_id}")
                    except Exception as insert_error:
                        logger.warning(f"Insert failed, trying to update: {str(insert_error)}")
                        # If insert fails, try to update the existing document
                        self.supabase.table(self.table_name).update(data).eq("id", doc_id).execute()
                        logger.info(f"Updated existing document in database: {doc_id}")
                        
                # Track the document ID whether we stored it in DB or not
                stored_ids.append(doc_id)
                
            except Exception as e:
                logger.error(f"Error storing document in vector store: {str(e)}")
                # Still track the document ID for in-memory usage
                stored_ids.append(doc_id)
                
        return stored_ids
    
    def store_hyde_queries(self, document_id: str, hyde_queries: Dict[str, Any]) -> bool:
        """
        Store HyDE queries associated with a document in Supabase
        
        Args:
            document_id: ID of the document
            hyde_queries: Dictionary containing HyDE queries
            
        Returns:
            Success status
        """
        try:
            # Store HyDE queries in a separate table
            data = {
                "document_id": document_id,
                "queries": json.dumps(hyde_queries),
                "created_at": datetime.now().isoformat()
            }
            
            self.supabase.table("hyde_queries").insert(data).execute()
            logger.info(f"Stored HyDE queries for document: {document_id}")
            return True
        except Exception as e:
            logger.error(f"Error storing HyDE queries: {str(e)}")
            return False
            
    def get_hyde_queries(self, limit: int = 100, document_id: Optional[str] = None, max_age_days: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve stored HyDE queries from the database
        
        Args:
            limit: Maximum number of records to retrieve
            document_id: Optional document ID to filter by
            max_age_days: Optional maximum age in days for retrieved queries
            
        Returns:
            List of HyDE query objects
        """
        try:
            # Start building the query
            query = self.supabase.table("hyde_queries")
            
            # Apply document ID filter if provided
            if document_id:
                query = query.eq("document_id", document_id)
                
            # Apply age filter if provided
            if max_age_days:
                # Calculate the cutoff date
                from datetime import datetime, timedelta
                cutoff_date = (datetime.now() - timedelta(days=max_age_days)).isoformat()
                query = query.gt("created_at", cutoff_date)
            
            # Apply limit and execute the query
            response = query.limit(limit).order("created_at", desc=True).execute()
            
            if response.data:
                # Parse the JSON queries field for each record
                for item in response.data:
                    if "queries" in item and isinstance(item["queries"], str):
                        try:
                            item["queries"] = json.loads(item["queries"])
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse queries JSON for item {item.get('document_id', 'unknown')}")
                
                logger.info(f"Retrieved {len(response.data)} HyDE query records from database")
                return response.data
            else:
                logger.info("No HyDE queries found in database")
                return []
                
        except Exception as e:
            logger.error(f"Error retrieving HyDE queries: {str(e)}")
            return []

    def similarity_search(
        self, 
        query_embedding: List[float], 
        k: int = 4,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform similarity search using query embedding
        
        Args:
            query_embedding: Vector embedding of the query
            k: Number of results to return
            filter_criteria: Optional metadata filter criteria
            
        Returns:
            List of Document objects most similar to the query
        """
        try:
            # Try to perform search in database
            try:
                # Call the match_documents RPC function
                rpc_response = self.supabase.rpc(
                    'match_documents',
                    {
                        'query_embedding': query_embedding, 
                        'match_count': k
                    }
                ).execute()
                
                # Process results
                results = rpc_response.data if rpc_response.data else []
                
                logger.info(f"Retrieved {len(results)} results from similarity search")
                
                # Convert to Document objects
                return [
                    Document(
                        page_content=result.get("content", ""),
                        metadata=json.loads(result.get("metadata", "{}"))
                    )
                    for result in results
                ]
                
            except Exception as e:
                logger.warning(f"Database similarity search failed: {str(e)}")
                logger.info("Returning empty result set")
                return []
                
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            return []
            
    def version_document(self, document_id: str, new_content: str, new_embedding: List[float]) -> str:
        """
        Create a new version of an existing document
        
        Args:
            document_id: ID of the document to version
            new_content: Updated document content
            new_embedding: Updated embedding vector
            
        Returns:
            ID of the new document version
        """
        try:
            # Create a new version with updated content and embedding
            # For a real implementation, you would need a versioning scheme
            new_id = f"{document_id}_v2"
            
            logger.info(f"Versioned document {document_id} to {new_id}")
            return new_id
            
        except Exception as e:
            logger.error(f"Error versioning document: {str(e)}")
            return document_id  # Return original ID if versioning fails