"""
Model Manager - Handles loading and switching between different embedding models
"""
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class ModelManager:
    """Centralized model management for the recruitment system"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.available_models = self._get_available_models()
            self.current_embedding_model = None
            self.current_ner_model = None
    
    def _get_available_models(self) -> Dict[str, Any]:
        """Get available models and their status"""
        from app.core.config import settings
        
        models = {
            "embedding": {
                "bge_large": {
                    "name": "BAAI/bge-large-en-v1.5",
                    "dimension": 1024,
                    "size_mb": 1300,
                    "description": "Best semantic similarity, recommended for production"
                },
                "mpnet": {
                    "name": "sentence-transformers/all-mpnet-base-v2",
                    "dimension": 768,
                    "size_mb": 420,
                    "description": "Good balance of speed and accuracy"
                },
                "minilm": {
                    "name": "sentence-transformers/all-MiniLM-L6-v2",
                    "dimension": 384,
                    "size_mb": 80,
                    "description": "Fastest, good for development"
                },
                "e5_large": {
                    "name": "intfloat/e5-large-v2",
                    "dimension": 1024,
                    "size_mb": 1300,
                    "description": "Excellent for retrieval tasks"
                },
                "gte_large": {
                    "name": "thenlper/gte-large",
                    "dimension": 1024,
                    "size_mb": 1300,
                    "description": "Great for text matching"
                }
            },
            "ner": {
                "custom_roberta": {
                    "path": str(settings.custom_ner_model_path),
                    "description": "Your custom fine-tuned RoBERTa model"
                },
                "roberta_large": {
                    "name": "Jean-Baptiste/roberta-large-ner-english",
                    "description": "Standard RoBERTa large NER"
                },
                "bert_base": {
                    "name": "dslim/bert-base-NER",
                    "description": "Fallback BERT model"
                }
            }
        }
        
        # Check if custom model exists
        custom_path = Path(settings.custom_ner_model_path)
        models["ner"]["custom_roberta"]["available"] = custom_path.exists()
        
        return models
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about currently loaded models"""
        info = {
            "available_models": self.available_models,
            "current_embedding": None,
            "current_ner": None
        }
        
        # Try to get current embedding model info
        try:
            from app.models.embedding_model import embedding_model
            info["current_embedding"] = embedding_model.get_model_info()
        except Exception as e:
            logger.error(f"Could not get embedding model info: {e}")
        
        return info
    
    def switch_embedding_model(self, model_key: str) -> bool:
        """Switch to a different embedding model (requires restart)"""
        if model_key not in self.available_models["embedding"]:
            logger.error(f"Model {model_key} not found")
            return False
        
        from app.core.config import settings
        model_config = self.available_models["embedding"][model_key]
        settings.embedding_model = model_config["name"]
        
        logger.info(f"Switched to embedding model: {model_key}")
        logger.warning("Restart the application for changes to take effect")
        return True

# Singleton
model_manager = ModelManager()