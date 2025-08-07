"""
Command-line interface for EmbeddingService
"""

import fire
import json
import logging
from pathlib import Path
from typing import List, Optional

from . import EmbeddingService
from .config import Config


class CLI:
    """Command-line interface for EmbeddingService"""
    
    def __init__(self):
        self.service = None
    
    def _get_service(self, config_path: Optional[str] = None) -> EmbeddingService:
        """Get or create the service instance"""
        if self.service is None:
            if config_path:
                config = Config.from_file(config_path)
            else:
                config = Config.from_env()
            self.service = EmbeddingService(config)
        return self.service
    
    def process_file(self, file_path: str, config_path: Optional[str] = None) -> str:
        """Process a single file and generate embedding
        
        Args:
            file_path: Path to the file to process
            config_path: Optional path to configuration file
        
        Returns:
            JSON string with processing result
        """
        service = self._get_service(config_path)
        result = service.process_file(file_path)
        
        output = {
            "file_path": result.file_path,
            "success": result.success,
            "content_type": result.metadata.get("content_type") if result.metadata else None,
            "embedding_dimension": len(result.embedding) if result.embedding else 0,
            "error": result.error
        }
        
        return json.dumps(output, indent=2, ensure_ascii=False)
    
    def process_directory(self, directory_path: str, config_path: Optional[str] = None) -> str:
        """Process all files in a directory
        
        Args:
            directory_path: Path to the directory to process
            config_path: Optional path to configuration file
        
        Returns:
            JSON string with processing results
        """
        service = self._get_service(config_path)
        results = service.process_directory(directory_path)
        
        output = {
            "directory": directory_path,
            "total_files": len(results),
            "successful": len([r for r in results if r.success]),
            "failed": len([r for r in results if not r.success]),
            "results": [
                {
                    "file_path": r.file_path,
                    "success": r.success,
                    "content_type": r.metadata.get("content_type") if r.metadata else None,
                    "error": r.error
                }
                for r in results
            ]
        }
        
        return json.dumps(output, indent=2, ensure_ascii=False)
    
    def search(self, query: str, limit: int = 10, config_path: Optional[str] = None) -> str:
        """Search for similar documents
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            config_path: Optional path to configuration file
        
        Returns:
            JSON string with search results
        """
        service = self._get_service(config_path)
        results = service.search(query, limit)
        
        output = {
            "query": query,
            "limit": limit,
            "total_results": len(results),
            "results": results
        }
        
        return json.dumps(output, indent=2, ensure_ascii=False)
    
    def get_metadata(self, file_path: str, config_path: Optional[str] = None) -> str:
        """Get metadata for a specific file
        
        Args:
            file_path: Path to the file
            config_path: Optional path to configuration file
        
        Returns:
            JSON string with file metadata
        """
        service = self._get_service(config_path)
        metadata = service.get_file_metadata(file_path)
        
        if metadata:
            return json.dumps(metadata, indent=2, ensure_ascii=False)
        else:
            return json.dumps({"error": "File not found"}, indent=2, ensure_ascii=False)
    
    def list_files(self, content_type: Optional[str] = None, config_path: Optional[str] = None) -> str:
        """List all processed files
        
        Args:
            content_type: Optional content type filter
            config_path: Optional path to configuration file
        
        Returns:
            JSON string with file list
        """
        service = self._get_service(config_path)
        files = service.metadata_manager.list_files(content_type)
        
        output = {
            "content_type_filter": content_type,
            "total_files": len(files),
            "files": [
                {
                    "file_path": f["file_path"],
                    "file_type": f["file_type"],
                    "content_type": f["content_type"],
                    "processed_at": f["processed_at"]
                }
                for f in files
            ]
        }
        
        return json.dumps(output, indent=2, ensure_ascii=False)
    
    def config(self, action: str = "show", config_path: Optional[str] = None) -> str:
        """Configuration management
        
        Args:
            action: Action to perform (show, create)
            config_path: Path to configuration file
        
        Returns:
            JSON string with configuration information
        """
        if action == "show":
            config = Config.from_env()
            return json.dumps(config.to_dict(), indent=2, ensure_ascii=False)
        
        elif action == "create":
            if not config_path:
                return json.dumps({"error": "config_path is required for create action"}, indent=2)
            
            config = Config.from_env()
            # Save configuration to file
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
            
            return json.dumps({"success": f"Configuration saved to {config_path}"}, indent=2, ensure_ascii=False)
        
        else:
            return json.dumps({"error": f"Unknown action: {action}"}, indent=2)
    
    def status(self, config_path: Optional[str] = None) -> str:
        """Get service status information
        
        Args:
            config_path: Optional path to configuration file
        
        Returns:
            JSON string with status information
        """
        service = self._get_service(config_path)
        
        try:
            # Get database stats
            with service.metadata_manager.conn as conn:
                total_files = conn.execute("SELECT COUNT(*) FROM file_metadata").fetchone()[0]
                total_embeddings = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            
            # Get search index stats
            search_docs = service.search_manager.get_document_count()
            
            output = {
                "service_status": "running",
                "database_path": service.config.duckdb_path,
                "search_index": service.config.meilisearch_index,
                "total_files": total_files,
                "total_embeddings": total_embeddings,
                "search_documents": search_docs,
                "models_loaded": list(service.embedding_registry.models.keys()),
                "processors_available": len(service.processor_registry.processors)
            }
            
            return json.dumps(output, indent=2, ensure_ascii=False)
        
        except Exception as e:
            return json.dumps({
                "service_status": "error",
                "error": str(e)
            }, indent=2, ensure_ascii=False)


def main():
    """Main entry point for the CLI"""
    logging.basicConfig(level=logging.INFO)
    fire.Fire(CLI())


if __name__ == "__main__":
    main()