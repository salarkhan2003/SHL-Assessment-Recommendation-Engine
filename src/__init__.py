"""
SHL Assessment Recommendation Engine

A production-quality recommendation system for SHL assessments using
semantic embeddings and RAG-style retrieval.

Modules:
- data: Catalog and dataset loading
- embeddings: Sentence embedding computation
- recommender: Core recommendation engine
- evaluation: Metrics and evaluation
- scraper: SHL catalog scraping

Note: Heavy imports (sentence-transformers, sklearn) are loaded lazily
to allow using individual modules without all dependencies.
"""

__version__ = "1.0.0"

# Lazy imports - these are loaded only when accessed
def __getattr__(name):
    """Lazy import of heavy modules."""
    if name == "AssessmentRecommender":
        from .recommender import AssessmentRecommender
        return AssessmentRecommender
    elif name == "RecommendationResult":
        from .recommender import RecommendationResult
        return RecommendationResult
    elif name == "CatalogLoader":
        from .data import CatalogLoader
        return CatalogLoader
    elif name == "Assessment":
        from .data import Assessment
        return Assessment
    elif name == "DatasetLoader":
        from .data import DatasetLoader
        return DatasetLoader
    elif name == "QuerySample":
        from .data import QuerySample
        return QuerySample
    elif name == "EmbeddingManager":
        from .embeddings import EmbeddingManager
        return EmbeddingManager
    elif name == "Evaluator":
        from .evaluation import Evaluator
        return Evaluator
    elif name == "EvaluationReport":
        from .evaluation import EvaluationReport
        return EvaluationReport
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AssessmentRecommender",
    "RecommendationResult",
    "CatalogLoader",
    "Assessment",
    "DatasetLoader",
    "QuerySample",
    "EmbeddingManager",
    "Evaluator",
    "EvaluationReport",
]

