"""
Embedding Manager Module

Handles computation and caching of semantic embeddings for assessments and queries.
Uses sentence-transformers for high-quality embeddings optimized for semantic search.
"""

import os
import pickle
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages embedding computation and caching for semantic search.

    Supports multiple embedding models with automatic caching for performance.

    Usage:
        manager = EmbeddingManager(model_name="all-mpnet-base-v2")

        # Embed catalog
        embeddings = manager.get_embeddings(texts, cache_key="catalog")

        # Embed query
        query_emb = manager.encode_query("software engineer")
    """

    # Model options (quality vs speed trade-off)
    AVAILABLE_MODELS = {
        "all-MiniLM-L6-v2": {
            "dim": 384,
            "description": "Fast, good quality",
            "speed": "fast"
        },
        "all-mpnet-base-v2": {
            "dim": 768,
            "description": "High quality, recommended for production",
            "speed": "medium"
        },
        "multi-qa-mpnet-base-dot-v1": {
            "dim": 768,
            "description": "Optimized for semantic search / QA",
            "speed": "medium"
        },
        "sentence-t5-base": {
            "dim": 768,
            "description": "T5-based, very high quality",
            "speed": "slow"
        }
    }

    DEFAULT_MODEL = "all-mpnet-base-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: str = "data/embeddings_cache",
        device: Optional[str] = None
    ):
        """
        Initialize the embedding manager.

        Args:
            model_name: Name of sentence-transformers model
            cache_dir: Directory for caching embeddings
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        self._model: Optional[SentenceTransformer] = None

        if model_name not in self.AVAILABLE_MODELS:
            logger.warning(f"Model {model_name} not in known models, proceeding anyway")

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Model loaded (dimension: {self._model.get_sentence_embedding_dimension()})")
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    def get_embeddings(
        self,
        texts: List[str],
        use_cache: bool = True,
        cache_key: Optional[str] = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Get embeddings for a list of texts, using cache if available.

        Args:
            texts: List of text strings to embed
            use_cache: Whether to use/save cache
            cache_key: Optional custom cache key
            show_progress: Show progress bar

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])

        # Generate cache key
        if cache_key is None:
            cache_key = self._generate_cache_key(texts)

        cache_path = self.cache_dir / f"{cache_key}_{self.model_name.replace('/', '_')}.pkl"

        # Try loading from cache
        if use_cache and cache_path.exists():
            cached = self._load_cache(cache_path, texts)
            if cached is not None:
                logger.info(f"Loaded embeddings from cache: {cache_path}")
                return cached

        # Compute embeddings
        logger.info(f"Computing embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalize for cosine similarity
        )

        # Save to cache
        if use_cache:
            self._save_cache(cache_path, embeddings, texts)

        return embeddings

    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Encode a single query string.

        Args:
            query: Query text to encode
            normalize: Whether to L2 normalize (recommended for cosine similarity)

        Returns:
            numpy array of shape (embedding_dim,)
        """
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        return embedding

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode a batch of texts without caching.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            normalize: Whether to L2 normalize
            show_progress: Show progress bar

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )

    def _generate_cache_key(self, texts: List[str]) -> str:
        """Generate cache key from texts."""
        content_hash = hashlib.md5(
            "\n".join(texts[:100]).encode()  # Use first 100 for hash
        ).hexdigest()[:8]
        return f"emb_{len(texts)}_{content_hash}"

    def _save_cache(self, path: Path, embeddings: np.ndarray, texts: List[str]) -> None:
        """Save embeddings to cache."""
        cache_data = {
            "embeddings": embeddings,
            "texts_hash": self._hash_texts(texts),
            "model_name": self.model_name,
            "num_texts": len(texts)
        }
        with open(path, "wb") as f:
            pickle.dump(cache_data, f)
        logger.info(f"Cached embeddings to: {path}")

    def _load_cache(self, path: Path, texts: List[str]) -> Optional[np.ndarray]:
        """Load embeddings from cache if valid."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            # Validate cache
            if data.get("model_name") != self.model_name:
                logger.info("Cache model mismatch, recomputing")
                return None
            if data.get("num_texts") != len(texts):
                logger.info("Cache size mismatch, recomputing")
                return None
            if data.get("texts_hash") != self._hash_texts(texts):
                logger.info("Cache content mismatch, recomputing")
                return None

            return data["embeddings"]
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
            return None

    def _hash_texts(self, texts: List[str]) -> str:
        """Hash texts for cache validation."""
        combined = "\n".join(texts)
        return hashlib.md5(combined.encode()).hexdigest()

    def clear_cache(self) -> int:
        """Clear all cached embeddings. Returns number of files deleted."""
        count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
            count += 1
        logger.info(f"Cleared {count} cache files")
        return count

    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and corpus.

        Since embeddings are normalized, dot product = cosine similarity.

        Args:
            query_embedding: Query embedding (embedding_dim,)
            corpus_embeddings: Corpus embeddings (n, embedding_dim)

        Returns:
            Similarity scores (n,)
        """
        # Ensure query is 1D
        if query_embedding.ndim == 2:
            query_embedding = query_embedding[0]

        # Dot product (= cosine similarity for normalized vectors)
        return corpus_embeddings @ query_embedding

