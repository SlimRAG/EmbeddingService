"""
Storage management for metadata and search
"""

import logging
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

import duckdb
from meilisearch import Client


class MetadataManager:
    """Manages metadata storage using DuckDB"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.db_path = config.duckdb_path
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize the database and create tables"""
        try:
            with duckdb.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS file_metadata (
                        id INTEGER PRIMARY KEY,
                        file_path TEXT UNIQUE,
                        file_type TEXT,
                        content_type TEXT,
                        processed_at TIMESTAMP,
                        metadata JSON,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS embeddings (
                        id INTEGER PRIMARY KEY,
                        file_path TEXT,
                        embedding_data BLOB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (file_path) REFERENCES file_metadata(file_path)
                    )
                """)
                
                self.logger.info("Database initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    def store_metadata(self, metadata: Dict[str, Any]):
        """Store file metadata"""
        try:
            with duckdb.connect(self.db_path) as conn:
                file_path = metadata["file_path"]
                file_type = metadata["file_type"]
                content_type = metadata["content_type"]
                processed_at = metadata["processed_at"]
                
                # Convert datetime to string for storage
                if isinstance(processed_at, datetime):
                    processed_at = processed_at.isoformat()
                
                # Store metadata as JSON
                metadata_json = json.dumps(metadata)
                
                conn.execute("""
                    INSERT OR REPLACE INTO file_metadata 
                    (file_path, file_type, content_type, processed_at, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (file_path, file_type, content_type, processed_at, metadata_json))
                
                self.logger.info(f"Metadata stored for {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to store metadata: {str(e)}")
            raise
    
    def get_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific file"""
        try:
            with duckdb.connect(self.db_path) as conn:
                result = conn.execute("""
                    SELECT metadata FROM file_metadata 
                    WHERE file_path = ?
                """, (file_path,)).fetchone()
                
                if result:
                    return json.loads(result[0])
                return None
        except Exception as e:
            self.logger.error(f"Failed to get metadata: {str(e)}")
            return None
    
    def list_files(self, content_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all files with optional content type filter"""
        try:
            with duckdb.connect(self.db_path) as conn:
                if content_type:
                    result = conn.execute("""
                        SELECT metadata FROM file_metadata 
                        WHERE content_type = ?
                    """, (content_type,)).fetchall()
                else:
                    result = conn.execute("""
                        SELECT metadata FROM file_metadata
                    """).fetchall()
                
                return [json.loads(row[0]) for row in result]
        except Exception as e:
            self.logger.error(f"Failed to list files: {str(e)}")
            return []
    
    def delete_metadata(self, file_path: str):
        """Delete metadata for a specific file"""
        try:
            with duckdb.connect(self.db_path) as conn:
                conn.execute("""
                    DELETE FROM file_metadata WHERE file_path = ?
                """, (file_path,))
                
                conn.execute("""
                    DELETE FROM embeddings WHERE file_path = ?
                """, (file_path,))
                
                self.logger.info(f"Metadata deleted for {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to delete metadata: {str(e)}")
            raise


class SearchManager:
    """Manages search functionality using Meilisearch"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = Client(config.meilisearch_url)
        self.index_name = config.meilisearch_index
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize the search index"""
        try:
            # Create index if it doesn't exist
            if not self.client.index_exists(self.index_name):
                index = self.client.create_index(self.index_name)
                self.logger.info(f"Created search index: {self.index_name}")
            else:
                index = self.client.index(self.index_name)
            
            # Configure searchable attributes
            index.update_searchable_attributes([
                "file_path",
                "content_type",
                "file_type"
            ])
            
            # Configure filterable attributes
            index.update_filterable_attributes([
                "content_type",
                "file_type",
                "processed_at"
            ])
            
            self.logger.info("Search index configured successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize search index: {str(e)}")
            # Don't raise here - search is optional
    
    def add_document(self, file_path: str, embedding: List[float], metadata: Dict[str, Any]):
        """Add a document to the search index"""
        try:
            document = {
                "id": file_path,
                "file_path": file_path,
                "content_type": metadata["content_type"],
                "file_type": metadata["file_type"],
                "processed_at": metadata["processed_at"],
                "embedding": embedding,
                "metadata": metadata
            }
            
            index = self.client.index(self.index_name)
            index.add_documents([document])
            
            self.logger.info(f"Document added to search index: {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to add document to search index: {str(e)}")
            # Search is optional, so don't raise
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for documents"""
        try:
            index = self.client.index(self.index_name)
            search_results = index.search(query, {"limit": limit})
            
            results = []
            for hit in search_results["hits"]:
                results.append({
                    "file_path": hit["file_path"],
                    "content_type": hit["content_type"],
                    "file_type": hit["file_type"],
                    "score": hit.get("_rankingScore", 0),
                    "metadata": hit["metadata"]
                })
            
            return results
        except Exception as e:
            self.logger.error(f"Failed to search: {str(e)}")
            return []
    
    def delete_document(self, file_path: str):
        """Delete a document from the search index"""
        try:
            index = self.client.index(self.index_name)
            index.delete_document(file_path)
            
            self.logger.info(f"Document deleted from search index: {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to delete document from search index: {str(e)}")
            # Search is optional, so don't raise
    
    def get_document_count(self) -> int:
        """Get the total number of documents in the index"""
        try:
            index = self.client.index(self.index_name)
            stats = index.get_stats()
            return stats.get("numberOfDocuments", 0)
        except Exception as e:
            self.logger.error(f"Failed to get document count: {str(e)}")
            return 0