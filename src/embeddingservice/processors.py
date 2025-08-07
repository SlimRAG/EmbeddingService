"""
File processors for different file types
"""

import os
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

import markitdown
from mineru import SingleFilePipeline


@dataclass
class ProcessingResult:
    """Result of content processing"""
    success: bool
    content: str
    content_type: str
    metadata: Dict[str, Any]
    processed_at: datetime
    error: Optional[str] = None


class FileProcessor(ABC):
    """Abstract base class for file processors"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def can_process(self, file_extension: str) -> bool:
        """Check if this processor can handle the file type"""
        pass
    
    @abstractmethod
    def process(self, file_path: Path) -> ProcessingResult:
        """Process the file and extract content"""
        pass


class MarkdownProcessor(FileProcessor):
    """Processor for Markdown files"""
    
    def can_process(self, file_extension: str) -> bool:
        return file_extension.lower() == ".md"
    
    def process(self, file_path: Path) -> ProcessingResult:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            metadata = {
                "file_size": file_path.stat().st_size,
                "word_count": len(content.split()),
                "char_count": len(content)
            }
            
            return ProcessingResult(
                success=True,
                content=content,
                content_type="text",
                metadata=metadata,
                processed_at=datetime.now()
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                content="",
                content_type="text",
                metadata={},
                processed_at=datetime.now(),
                error=str(e)
            )


class DocumentProcessor(FileProcessor):
    """Processor for PDF, DOC, DOCX files using MinerU"""
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = {".pdf", ".doc", ".docx"}
    
    def can_process(self, file_extension: str) -> bool:
        return file_extension.lower() in self.supported_extensions
    
    def process(self, file_path: Path) -> ProcessingResult:
        try:
            # Use MinerU to extract text from documents
            pipeline = SingleFilePipeline(str(file_path))
            result = pipeline.run()
            
            content = result.get("text", "")
            metadata = {
                "file_size": file_path.stat().st_size,
                "pages": result.get("pages", 0),
                "word_count": len(content.split()),
                "char_count": len(content),
                "mineru_result": result
            }
            
            return ProcessingResult(
                success=True,
                content=content,
                content_type="text",
                metadata=metadata,
                processed_at=datetime.now()
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                content="",
                content_type="text",
                metadata={},
                processed_at=datetime.now(),
                error=str(e)
            )


class HTMLProcessor(FileProcessor):
    """Processor for HTML files using markitdown"""
    
    def can_process(self, file_extension: str) -> bool:
        return file_extension.lower() == ".html"
    
    def process(self, file_path: Path) -> ProcessingResult:
        try:
            # Use markitdown to convert HTML to Markdown
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            markdown_content = markitdown.convert(html_content)
            
            metadata = {
                "file_size": file_path.stat().st_size,
                "word_count": len(markdown_content.split()),
                "char_count": len(markdown_content)
            }
            
            return ProcessingResult(
                success=True,
                content=markdown_content,
                content_type="text",
                metadata=metadata,
                processed_at=datetime.now()
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                content="",
                content_type="text",
                metadata={},
                processed_at=datetime.now(),
                error=str(e)
            )


class ImageProcessor(FileProcessor):
    """Processor for image files"""
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    
    def can_process(self, file_extension: str) -> bool:
        return file_extension.lower() in self.supported_extensions
    
    def process(self, file_path: Path) -> ProcessingResult:
        try:
            # For images, we'll return the file path and let the embedding model handle the actual image processing
            metadata = {
                "file_size": file_path.stat().st_size,
                "image_format": file_path.suffix.lower(),
                "file_path": str(file_path)
            }
            
            return ProcessingResult(
                success=True,
                content=str(file_path),  # Return file path for image processing
                content_type="image",
                metadata=metadata,
                processed_at=datetime.now()
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                content="",
                content_type="image",
                metadata={},
                processed_at=datetime.now(),
                error=str(e)
            )


class VideoProcessor(FileProcessor):
    """Processor for video files"""
    
    def can_process(self, file_extension: str) -> bool:
        return file_extension.lower() == ".mp4"
    
    def process(self, file_path: Path) -> ProcessingResult:
        try:
            # For videos, we'll extract keyframes and return their paths
            keyframe_paths = self._extract_keyframes(file_path)
            
            metadata = {
                "file_size": file_path.stat().st_size,
                "keyframe_count": len(keyframe_paths),
                "keyframe_paths": keyframe_paths
            }
            
            return ProcessingResult(
                success=True,
                content=str(file_path),  # Return video file path
                content_type="video",
                metadata=metadata,
                processed_at=datetime.now()
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                content="",
                content_type="video",
                metadata={},
                processed_at=datetime.now(),
                error=str(e)
            )
    
    def _extract_keyframes(self, video_path: Path) -> List[str]:
        """Extract keyframes from video"""
        # This is a placeholder implementation
        # In a real implementation, you would use OpenCV or similar
        # to extract keyframes from the video
        self.logger.warning("Video keyframe extraction not implemented yet")
        return []


class FileProcessorRegistry:
    """Registry for file processors"""
    
    def __init__(self):
        self.processors = [
            MarkdownProcessor(),
            DocumentProcessor(),
            HTMLProcessor(),
            ImageProcessor(),
            VideoProcessor()
        ]
    
    def get_processor(self, file_extension: str) -> Optional[FileProcessor]:
        """Get the appropriate processor for a file extension"""
        for processor in self.processors:
            if processor.can_process(file_extension):
                return processor
        return None
    
    def register_processor(self, processor: FileProcessor):
        """Register a new processor"""
        self.processors.append(processor)