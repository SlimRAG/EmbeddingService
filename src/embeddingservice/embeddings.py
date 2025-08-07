"""
Embedding model integrations
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Any
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image

from .config import Config


class EmbeddingModel(ABC):
    """Abstract base class for embedding models"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = self._get_device()
    
    def _get_device(self) -> str:
        """Get the appropriate device for the model"""
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device
    
    @abstractmethod
    def embed(self, content: Any) -> List[float]:
        """Generate embedding for the given content"""
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get the dimension of the embedding"""
        pass


class BGE_M3_Model(EmbeddingModel):
    """BGE-M3 embedding model for text"""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.model_name = config.bge_m3_model_path
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the BGE-M3 model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self.logger.info(f"BGE-M3 model loaded successfully on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load BGE-M3 model: {str(e)}")
            raise
    
    def embed(self, content: str) -> List[float]:
        """Generate embedding for text content"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded")
        
        try:
            # Truncate content if too long
            if len(content) > self.config.max_text_length:
                content = content[:self.config.max_text_length]
            
            # Tokenize input
            inputs = self.tokenizer(
                content,
                padding=True,
                truncation=True,
                max_length=self.config.max_text_length,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding or mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
                embedding = embeddings.cpu().numpy()[0].tolist()
            
            return embedding
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def get_embedding_dim(self) -> int:
        return self.config.embedding_dim


class Chinese_CLIP_Model(EmbeddingModel):
    """Chinese-CLIP model for image embeddings"""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.model_name = config.chinese_clip_path
        self.processor = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Chinese-CLIP model"""
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self.logger.info(f"Chinese-CLIP model loaded successfully on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load Chinese-CLIP model: {str(e)}")
            raise
    
    def embed(self, content: str) -> List[float]:
        """Generate embedding for image content"""
        if not self.model or not self.processor:
            raise RuntimeError("Model not loaded")
        
        try:
            # Load and preprocess image
            image = Image.open(content)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image if needed
            image = image.resize(self.config.image_size)
            
            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                embedding = outputs.cpu().numpy()[0].tolist()
            
            return embedding
        except Exception as e:
            self.logger.error(f"Error generating image embedding: {str(e)}")
            raise
    
    def get_embedding_dim(self) -> int:
        return self.config.embedding_dim


class Qwen3_Embedding_Model(EmbeddingModel):
    """Qwen3-Embedding model for text"""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.model_name = config.qwen3_embedding_path
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Qwen3-Embedding model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self.logger.info(f"Qwen3-Embedding model loaded successfully on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load Qwen3-Embedding model: {str(e)}")
            raise
    
    def embed(self, content: str) -> List[float]:
        """Generate embedding for text content"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded")
        
        try:
            # Truncate content if too long
            if len(content) > self.config.max_text_length:
                content = content[:self.config.max_text_length]
            
            # Tokenize input
            inputs = self.tokenizer(
                content,
                padding=True,
                truncation=True,
                max_length=self.config.max_text_length,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
                embedding = embeddings.cpu().numpy()[0].tolist()
            
            return embedding
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def get_embedding_dim(self) -> int:
        return self.config.embedding_dim


class EmbeddingModelRegistry:
    """Registry for embedding models"""
    
    def __init__(self, config: Config):
        self.config = config
        self.models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all available models"""
        try:
            self.models["text"] = BGE_M3_Model(self.config)
            self.models["image"] = Chinese_CLIP_Model(self.config)
            self.models["qwen3"] = Qwen3_Embedding_Model(self.config)
        except Exception as e:
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.error(f"Failed to initialize models: {str(e)}")
            # Initialize with basic models if full models fail
            self.models["text"] = BGE_M3_Model(self.config)
    
    def get_model(self, content_type: str) -> Optional[EmbeddingModel]:
        """Get the appropriate model for content type"""
        model_mapping = {
            "text": "text",      # Use BGE-M3 for text
            "image": "image",    # Use Chinese-CLIP for images
            "video": "image",    # Use Chinese-CLIP for video keyframes
            "document": "text",  # Use BGE-M3 for documents
        }
        
        model_key = model_mapping.get(content_type)
        if model_key in self.models:
            return self.models[model_key]
        
        # Default to text model
        return self.models.get("text")
    
    def register_model(self, name: str, model: EmbeddingModel):
        """Register a new model"""
        self.models[name] = model