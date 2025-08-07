"""
EmbeddingService - A multi-modal embedding service for SlimRAG

This service processes various file types and generates embeddings for vector search.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from .processors import FileProcessorRegistry
from .embeddings import EmbeddingModelRegistry
from .storage import MetadataManager, SearchManager
from .config import Config


@dataclass
class ProcessingResult:
    """Result of file processing"""
    file_path: str
    success: bool
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class EmbeddingService:
    """Main embedding service class"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = self._setup_logger()
        
        # Initialize components
        self.processor_registry = FileProcessorRegistry()
        self.embedding_registry = EmbeddingModelRegistry(self.config)
        self.metadata_manager = MetadataManager(self.config)
        self.search_manager = SearchManager(self.config)
        
        self.logger.info("EmbeddingService initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def process_file(self, file_path: Union[str, Path]) -> ProcessingResult:
        """Process a single file and generate embedding"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return ProcessingResult(
                file_path=str(file_path),
                success=False,
                error=f"File not found: {file_path}"
            )
        
        try:
            # Get appropriate processor for file type
            processor = self.processor_registry.get_processor(file_path.suffix.lower())
            if not processor:
                return ProcessingResult(
                    file_path=str(file_path),
                    success=False,
                    error=f"No processor available for file type: {file_path.suffix}"
                )
            
            # Extract content from file
            content_result = processor.process(file_path)
            if not content_result.success:
                return ProcessingResult(
                    file_path=str(file_path),
                    success=False,
                    error=content_result.error
                )
            
            # Get appropriate embedding model
            embedding_model = self.embedding_registry.get_model(content_result.content_type)
            if not embedding_model:
                return ProcessingResult(
                    file_path=str(file_path),
                    success=False,
                    error=f"No embedding model available for content type: {content_result.content_type}"
                )
            
            # Generate embedding
            embedding = embedding_model.embed(content_result.content)
            
            # Store metadata
            metadata = {
                "file_path": str(file_path),
                "file_type": file_path.suffix.lower(),
                "content_type": content_result.content_type,
                "processed_at": content_result.processed_at,
                "content_metadata": content_result.metadata
            }
            
            self.metadata_manager.store_metadata(metadata)
            
            # Add to search index
            self.search_manager.add_document(
                str(file_path),
                embedding,
                metadata
            )
            
            return ProcessingResult(
                file_path=str(file_path),
                success=True,
                embedding=embedding,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            return ProcessingResult(
                file_path=str(file_path),
                success=False,
                error=str(e)
            )
    
    def process_directory(self, directory_path: Union[str, Path]) -> List[ProcessingResult]:
        """Process all files in a directory"""
        directory_path = Path(directory_path)
        
        if not directory_path.exists() or not directory_path.is_dir():
            raise ValueError(f"Directory not found: {directory_path}")
        
        results = []
        for file_path in directory_path.rglob("*"):
            if file_path.is_file():
                result = self.process_file(file_path)
                results.append(result)
        
        return results
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        return self.search_manager.search(query, limit)
    
    def get_file_metadata(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific file"""
        return self.metadata_manager.get_metadata(str(file_path))