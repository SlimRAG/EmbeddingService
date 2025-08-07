"""
Configuration management for EmbeddingService
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class Config:
    """Configuration class for EmbeddingService"""
    
    # Model configurations
    bge_m3_model_path: str = "BAAI/bge-m3"
    qwen3_embedding_path: str = "Qwen/Qwen3-Embedding"
    chinese_clip_path: str = "OFA-Sys/Chinese-CLIP"
    
    # Processing configurations
    batch_size: int = 32
    max_text_length: int = 512
    image_size: tuple = (224, 224)
    
    # Storage configurations
    duckdb_path: str = "metadata.duckdb"
    meilisearch_url: str = "http://localhost:7700"
    meilisearch_index: str = "embeddings"
    
    # Video processing
    video_keyframe_interval: int = 5  # seconds
    
    # Logging
    log_level: str = "INFO"
    
    # Temporary directories
    temp_dir: str = "temp"
    
    # Additional settings
    device: str = "auto"  # auto, cpu, cuda
    embedding_dim: int = 1024
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables"""
        config = cls()
        
        # Override with environment variables if they exist
        env_mappings = {
            "BGE_M3_MODEL_PATH": "bge_m3_model_path",
            "QWEN3_EMBEDDING_PATH": "qwen3_embedding_path",
            "CHINESE_CLIP_PATH": "chinese_clip_path",
            "BATCH_SIZE": "batch_size",
            "MAX_TEXT_LENGTH": "max_text_length",
            "DUCKDB_PATH": "duckdb_path",
            "MEILISEARCH_URL": "meilisearch_url",
            "MEILISEARCH_INDEX": "meilisearch_index",
            "VIDEO_KEYFRAME_INTERVAL": "video_keyframe_interval",
            "LOG_LEVEL": "log_level",
            "TEMP_DIR": "temp_dir",
            "DEVICE": "device",
            "EMBEDDING_DIM": "embedding_dim"
        }
        
        for env_var, config_attr in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert to appropriate type
                if config_attr in ["batch_size", "max_text_length", "video_keyframe_interval", "embedding_dim"]:
                    value = int(value)
                elif config_attr == "image_size":
                    # Parse tuple from string like "224,224"
                    if "," in value:
                        width, height = map(int, value.split(","))
                        value = (width, height)
                
                setattr(config, config_attr, value)
        
        return config
    
    def ensure_temp_dir(self) -> Path:
        """Ensure temporary directory exists"""
        temp_path = Path(self.temp_dir)
        temp_path.mkdir(exist_ok=True)
        return temp_path
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "bge_m3_model_path": self.bge_m3_model_path,
            "qwen3_embedding_path": self.qwen3_embedding_path,
            "chinese_clip_path": self.chinese_clip_path,
            "batch_size": self.batch_size,
            "max_text_length": self.max_text_length,
            "image_size": self.image_size,
            "duckdb_path": self.duckdb_path,
            "meilisearch_url": self.meilisearch_url,
            "meilisearch_index": self.meilisearch_index,
            "video_keyframe_interval": self.video_keyframe_interval,
            "log_level": self.log_level,
            "temp_dir": self.temp_dir,
            "device": self.device,
            "embedding_dim": self.embedding_dim
        }