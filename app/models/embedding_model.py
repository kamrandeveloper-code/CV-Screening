"""Embedding model using sentence-transformers"""
import logging
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from app.core.config import settings

logger = logging.getLogger(__name__)

class EmbeddingModel:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if EmbeddingModel._initialized:
            return
            
        self.device = "cpu"
        logger.info(f"Loading embedding model: {settings.embedding_model} on {self.device}")
        
        try:
            # Try to load the configured model
            self.model = SentenceTransformer(
                settings.embedding_model,
                device=self.device,
                cache_folder=str(settings.models_dir / "embedding_models")
            )
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"[OK] Embedding model loaded: {settings.embedding_model}")
            logger.info(f"     Dimension: {self.dimension}")
        except Exception as e:
            logger.warning(f"Primary model {settings.embedding_model} failed: {e}")
            logger.info("Falling back to all-MiniLM-L6-v2")
            self.model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
            self.dimension = 384
        
        EmbeddingModel._initialized = True
    
    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        return self.model.encode(
            text,
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=False
        )
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=False
        )
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        return float(cosine_similarity([vec1], [vec2])[0][0])
    
    def similarity_matrix(self, vecs1: np.ndarray, vecs2: np.ndarray):
        return cosine_similarity(vecs1, vecs2)
    
    def get_model_info(self) -> dict:
        return {
            "model_name": settings.embedding_model,
            "dimension": self.dimension,
            "device": self.device,
            "max_seq_length": getattr(self.model, "max_seq_length", 512)
        }

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model