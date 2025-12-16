"""
Assessment Recommender Module

Core recommendation engine that combines:
- Semantic embeddings for query-assessment matching
- K/P type balancing when query spans multiple domains
- Relevance scoring and ranking
- Optional explanation generation

Designed to meet SHL assignment requirements for balanced recommendations.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
import re

from ..data.catalog_loader import CatalogLoader, Assessment
from ..embeddings.embedding_manager import EmbeddingManager

logger = logging.getLogger(__name__)


@dataclass
class RecommendationResult:
    """
    A single recommendation result with metadata.

    Attributes:
        assessment: The recommended Assessment object
        score: Relevance score (0-1, higher is better)
        rank: Ranking position (1-indexed)
        explanation: Why this assessment matches the query
    """
    assessment: Assessment
    score: float
    rank: int
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "rank": self.rank,
            "name": self.assessment.name,
            "url": self.assessment.url,
            "description": self.assessment.description,
            "test_type": self.assessment.test_type,
            "type_display": self.assessment.type_display,
            "remote_testing": self.assessment.remote_testing,
            "adaptive_irt": self.assessment.adaptive_irt,
            "duration": self.assessment.duration,
            "relevance_score": round(self.score, 4),
            "explanation": self.explanation
        }


class AssessmentRecommender:
    """
    Production-grade assessment recommendation engine.

    Features:
    - Semantic similarity using sentence embeddings
    - K/P type balancing for queries spanning multiple domains
    - Automatic explanation generation
    - Configurable ranking parameters

    Usage:
        recommender = AssessmentRecommender()
        recommender.load()

        results = recommender.recommend(
            "Looking for a software engineer with analytical skills",
            top_k=10
        )

        for r in results:
            print(f"{r.rank}. {r.assessment.name} ({r.assessment.test_type})")
            print(f"   Score: {r.score:.3f}")
            print(f"   Why: {r.explanation}")
    """

    # Keywords for detecting K vs P query intent
    K_QUERY_KEYWORDS = [
        "cognitive", "ability", "aptitude", "reasoning", "numerical", "verbal",
        "analytical", "problem solving", "critical thinking", "logical",
        "technical", "coding", "programming", "math", "quantitative",
        "abstract", "spatial", "mechanical", "data analysis"
    ]

    P_QUERY_KEYWORDS = [
        "personality", "behavioral", "leadership", "teamwork", "communication",
        "interpersonal", "motivation", "work style", "cultural fit", "values",
        "emotional intelligence", "customer service", "collaboration",
        "management", "soft skills", "attitude", "situational"
    ]

    def __init__(
        self,
        catalog_path: str = "data/shl_catalog.csv",
        embedding_model: str = "all-mpnet-base-v2",
        cache_dir: str = "data/embeddings_cache",
        enable_kp_balance: bool = True,
        kp_balance_ratio: float = 0.5
    ):
        """
        Initialize the recommender.

        Args:
            catalog_path: Path to scraped catalog CSV
            embedding_model: Sentence-transformers model name
            cache_dir: Directory for embedding cache
            enable_kp_balance: Balance K and P types when query spans both
            kp_balance_ratio: Target ratio of K-type (0.5 = equal)
        """
        self.catalog_path = catalog_path
        self.embedding_model = embedding_model
        self.cache_dir = cache_dir
        self.enable_kp_balance = enable_kp_balance
        self.kp_balance_ratio = kp_balance_ratio

        self._catalog_loader: Optional[CatalogLoader] = None
        self._embedding_manager: Optional[EmbeddingManager] = None
        self._assessments: Optional[List[Assessment]] = None
        self._embeddings: Optional[np.ndarray] = None
        self._is_loaded = False

    def load(self) -> "AssessmentRecommender":
        """
        Load catalog and precompute embeddings.

        Returns:
            self for method chaining
        """
        if self._is_loaded:
            return self

        # Load catalog
        logger.info("Loading assessment catalog...")
        self._catalog_loader = CatalogLoader(self.catalog_path)
        self._assessments = self._catalog_loader.load()
        logger.info(f"Loaded {len(self._assessments)} assessments")

        # Compute embeddings
        logger.info("Initializing embedding manager...")
        self._embedding_manager = EmbeddingManager(
            model_name=self.embedding_model,
            cache_dir=self.cache_dir
        )

        texts = self._catalog_loader.get_combined_texts()
        self._embeddings = self._embedding_manager.get_embeddings(
            texts,
            cache_key="shl_catalog"
        )

        self._is_loaded = True
        logger.info("Recommender ready")
        return self

    def recommend(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
        generate_explanations: bool = True
    ) -> List[RecommendationResult]:
        """
        Recommend assessments for a query.

        Args:
            query: Natural language query (job description, requirements, etc.)
            top_k: Maximum number of recommendations
            min_score: Minimum relevance score threshold
            generate_explanations: Whether to generate explanations

        Returns:
            List of RecommendationResult sorted by relevance
        """
        if not self._is_loaded:
            self.load()

        # Encode query
        query_embedding = self._embedding_manager.encode_query(query)

        # Compute similarities
        similarities = self._embedding_manager.compute_similarity(
            query_embedding,
            self._embeddings
        )

        # Determine query intent (K, P, or both)
        query_intent = self._analyze_query_intent(query)

        # Apply K/P balancing if enabled and query spans both
        if self.enable_kp_balance and query_intent == "both":
            results = self._balanced_ranking(similarities, top_k, min_score)
        else:
            results = self._simple_ranking(similarities, top_k, min_score)

        # Generate explanations
        if generate_explanations:
            for result in results:
                result.explanation = self._generate_explanation(query, result.assessment, result.score)

        return results

    def _simple_ranking(
        self,
        similarities: np.ndarray,
        top_k: int,
        min_score: float
    ) -> List[RecommendationResult]:
        """Simple ranking by similarity score."""
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(similarities[idx])
            if score < min_score:
                break

            results.append(RecommendationResult(
                assessment=self._assessments[idx],
                score=score,
                rank=rank
            ))

        return results

    def _balanced_ranking(
        self,
        similarities: np.ndarray,
        top_k: int,
        min_score: float
    ) -> List[RecommendationResult]:
        """
        Balanced ranking that ensures mix of K and P types.

        When query spans both cognitive and personality domains,
        ensure recommendations include both types proportionally.
        """
        # Sort all by similarity
        sorted_indices = np.argsort(similarities)[::-1]

        # Separate into K and P pools
        k_pool = []
        p_pool = []

        for idx in sorted_indices:
            score = float(similarities[idx])
            if score < min_score:
                continue

            assessment = self._assessments[idx]
            if assessment.is_cognitive:
                k_pool.append((idx, score))
            if assessment.is_personality:
                p_pool.append((idx, score))

        # Calculate target counts
        target_k = int(top_k * self.kp_balance_ratio)
        target_p = top_k - target_k

        # Select from each pool
        selected_indices = set()
        results = []

        # Interleave selection from K and P pools
        k_idx, p_idx = 0, 0
        while len(results) < top_k:
            # Add from K pool
            while k_idx < len(k_pool) and len([r for r in results if r.assessment.is_cognitive and not r.assessment.is_personality]) < target_k:
                idx, score = k_pool[k_idx]
                k_idx += 1
                if idx not in selected_indices:
                    selected_indices.add(idx)
                    results.append(RecommendationResult(
                        assessment=self._assessments[idx],
                        score=score,
                        rank=0  # Will be updated
                    ))
                    break

            # Add from P pool
            while p_idx < len(p_pool) and len([r for r in results if r.assessment.is_personality and not r.assessment.is_cognitive]) < target_p:
                idx, score = p_pool[p_idx]
                p_idx += 1
                if idx not in selected_indices:
                    selected_indices.add(idx)
                    results.append(RecommendationResult(
                        assessment=self._assessments[idx],
                        score=score,
                        rank=0
                    ))
                    break

            # If we've exhausted priority pools, fill from remaining
            if k_idx >= len(k_pool) and p_idx >= len(p_pool):
                break
            if len([r for r in results if r.assessment.is_cognitive and not r.assessment.is_personality]) >= target_k and \
               len([r for r in results if r.assessment.is_personality and not r.assessment.is_cognitive]) >= target_p:
                # Fill rest with highest scoring
                for idx in sorted_indices:
                    if idx not in selected_indices and similarities[idx] >= min_score:
                        selected_indices.add(idx)
                        results.append(RecommendationResult(
                            assessment=self._assessments[idx],
                            score=float(similarities[idx]),
                            rank=0
                        ))
                        if len(results) >= top_k:
                            break
                break

        # Re-sort by score and assign ranks
        results.sort(key=lambda r: r.score, reverse=True)
        for i, r in enumerate(results):
            r.rank = i + 1

        return results[:top_k]

    def _analyze_query_intent(self, query: str) -> str:
        """
        Analyze query to determine K, P, or both intent.

        Returns:
            "K" for cognitive-focused
            "P" for personality-focused
            "both" for queries spanning both domains
        """
        query_lower = query.lower()

        k_score = sum(1 for kw in self.K_QUERY_KEYWORDS if kw in query_lower)
        p_score = sum(1 for kw in self.P_QUERY_KEYWORDS if kw in query_lower)

        # Thresholds for classification
        if k_score >= 2 and p_score >= 2:
            return "both"
        elif k_score > p_score:
            return "K"
        elif p_score > k_score:
            return "P"
        else:
            # Default to "both" for general queries
            return "both"

    def _generate_explanation(self, query: str, assessment: Assessment, score: float) -> str:
        """
        Generate a brief explanation for why this assessment matches.

        Uses keyword matching and assessment metadata.
        """
        explanations = []
        query_lower = query.lower()
        name_lower = assessment.name.lower()
        desc_lower = assessment.description.lower()

        # High relevance
        if score > 0.6:
            explanations.append("Strong semantic match to query requirements")
        elif score > 0.4:
            explanations.append("Good match to query requirements")

        # Type-specific explanations
        if assessment.is_cognitive:
            k_matches = [kw for kw in self.K_QUERY_KEYWORDS if kw in query_lower]
            if k_matches:
                explanations.append(f"Measures {k_matches[0]} abilities")

        if assessment.is_personality:
            p_matches = [kw for kw in self.P_QUERY_KEYWORDS if kw in query_lower]
            if p_matches:
                explanations.append(f"Assesses {p_matches[0]} traits")

        # Feature explanations
        if "remote" in query_lower and assessment.remote_testing:
            explanations.append("Supports remote testing")

        if assessment.adaptive_irt:
            explanations.append("Adaptive assessment for accurate measurement")

        # Duration
        if assessment.duration:
            explanations.append(f"Duration: {assessment.duration}")

        return "; ".join(explanations[:3]) if explanations else "Relevant to query based on semantic similarity"

    def recommend_urls(self, query: str, top_k: int = 10) -> List[str]:
        """
        Get just URLs for top recommendations.

        Convenience method for submission generation.
        """
        results = self.recommend(query, top_k=top_k, generate_explanations=False)
        return [r.assessment.url for r in results]

    def find_assessment_by_url(self, url: str) -> Optional[Assessment]:
        """Find assessment by URL."""
        if not self._is_loaded:
            self.load()
        return self._catalog_loader.find_by_url(url)

    @property
    def catalog_size(self) -> int:
        """Number of assessments in catalog."""
        if not self._is_loaded:
            self.load()
        return len(self._assessments)

    @property
    def catalog_stats(self) -> Dict:
        """Get catalog statistics."""
        if not self._is_loaded:
            self.load()
        return self._catalog_loader.get_stats()

