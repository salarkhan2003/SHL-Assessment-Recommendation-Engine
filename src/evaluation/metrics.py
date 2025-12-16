"""
Evaluation Metrics Module

Provides comprehensive evaluation for the recommendation engine:
- Recall@K (primary metric per SHL assignment)
- Mean Average Precision (MAP@K)
- Mean Reciprocal Rank (MRR)
- Hit Rate

Generates clean reports suitable for technical documentation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# URL NORMALIZATION
# =============================================================================

def normalize_url(url: str) -> str:
    """
    Normalize URL for comparison.

    Handles variations in SHL URLs:
    - With/without trailing slash
    - Different path prefixes
    - Case differences

    Returns the slug (last path component) for comparison.
    """
    if not url:
        return ""

    # Parse URL and get path
    parsed = urlparse(url.lower().strip())
    path = parsed.path.rstrip('/')

    # Extract the slug (last meaningful path component)
    parts = [p for p in path.split('/') if p]
    if parts:
        return parts[-1]
    return url.lower().strip()


# =============================================================================
# METRIC FUNCTIONS
# =============================================================================

def recall_at_k(true_urls: List[str], predicted_urls: List[str], k: int = 10) -> float:
    """
    Calculate Recall@K.

    Recall@K = (# of relevant items in top-K) / (total # of relevant items)

    For single-label case (1 relevant item): 1.0 if found, 0.0 otherwise

    Args:
        true_urls: Ground truth URLs (can be single or multiple)
        predicted_urls: Predicted URLs in ranked order
        k: Number of top predictions to consider

    Returns:
        Recall score between 0 and 1
    """
    if not true_urls:
        return 0.0

    true_set = set(true_urls) if isinstance(true_urls, list) else {true_urls}
    pred_top_k = set(predicted_urls[:k])

    hits = len(true_set & pred_top_k)
    return hits / len(true_set)


def precision_at_k(true_urls: List[str], predicted_urls: List[str], k: int = 10) -> float:
    """
    Calculate Precision@K.

    Precision@K = (# of relevant items in top-K) / K

    Args:
        true_urls: Ground truth URLs
        predicted_urls: Predicted URLs in ranked order
        k: Number of top predictions to consider

    Returns:
        Precision score between 0 and 1
    """
    if k == 0:
        return 0.0

    true_set = set(true_urls) if isinstance(true_urls, list) else {true_urls}
    pred_top_k = predicted_urls[:k]

    hits = sum(1 for url in pred_top_k if url in true_set)
    return hits / k


def average_precision(true_urls: List[str], predicted_urls: List[str], k: int = 10) -> float:
    """
    Calculate Average Precision (AP) for a single query.

    AP = sum(P@i * rel(i)) / min(k, |relevant|)
    where rel(i) = 1 if item i is relevant, 0 otherwise

    Args:
        true_urls: Ground truth URLs
        predicted_urls: Predicted URLs in ranked order
        k: Maximum rank to consider

    Returns:
        AP score between 0 and 1
    """
    true_set = set(true_urls) if isinstance(true_urls, list) else {true_urls}

    if not true_set:
        return 0.0

    hits = 0
    sum_precisions = 0.0

    for i, url in enumerate(predicted_urls[:k], start=1):
        if url in true_set:
            hits += 1
            sum_precisions += hits / i

    return sum_precisions / min(k, len(true_set))


def mean_average_precision(
    all_true_urls: List[List[str]],
    all_predicted_urls: List[List[str]],
    k: int = 10
) -> float:
    """
    Calculate Mean Average Precision (MAP@K).

    Args:
        all_true_urls: List of ground truth URL lists (one per query)
        all_predicted_urls: List of predicted URL lists (one per query)
        k: Maximum rank to consider

    Returns:
        MAP score between 0 and 1
    """
    if len(all_true_urls) != len(all_predicted_urls):
        raise ValueError("Number of queries must match")

    if not all_true_urls:
        return 0.0

    aps = [
        average_precision(true, pred, k)
        for true, pred in zip(all_true_urls, all_predicted_urls)
    ]

    return np.mean(aps)


def reciprocal_rank(true_urls: List[str], predicted_urls: List[str]) -> float:
    """
    Calculate Reciprocal Rank with URL normalization.

    RR = 1 / (rank of first relevant item), or 0 if none found

    Args:
        true_urls: Ground truth URLs
        predicted_urls: Predicted URLs in ranked order

    Returns:
        RR score between 0 and 1
    """
    true_normalized = set(normalize_url(u) for u in true_urls)

    for i, url in enumerate(predicted_urls, start=1):
        if normalize_url(url) in true_normalized:
            return 1.0 / i

    return 0.0


def mean_reciprocal_rank(
    all_true_urls: List[List[str]],
    all_predicted_urls: List[List[str]]
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).

    Args:
        all_true_urls: List of ground truth URL lists
        all_predicted_urls: List of predicted URL lists

    Returns:
        MRR score between 0 and 1
    """
    if len(all_true_urls) != len(all_predicted_urls):
        raise ValueError("Number of queries must match")

    if not all_true_urls:
        return 0.0

    rrs = [
        reciprocal_rank(true, pred)
        for true, pred in zip(all_true_urls, all_predicted_urls)
    ]

    return np.mean(rrs)


# =============================================================================
# EVALUATION REPORT
# =============================================================================

@dataclass
class EvaluationReport:
    """
    Comprehensive evaluation report.

    Contains all metrics, per-query details, and metadata for documentation.
    """
    # Metrics at different K values
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    map_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    hit_rate_at_k: Dict[int, float] = field(default_factory=dict)

    # Summary
    num_queries: int = 0
    num_hits: Dict[int, int] = field(default_factory=dict)

    # Per-query details for analysis
    query_details: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    timestamp: str = ""
    model_name: str = ""
    catalog_size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "recall_at_k": self.recall_at_k,
            "precision_at_k": self.precision_at_k,
            "map_at_k": self.map_at_k,
            "mrr": self.mrr,
            "hit_rate_at_k": self.hit_rate_at_k,
            "num_queries": self.num_queries,
            "num_hits": self.num_hits,
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "catalog_size": self.catalog_size
        }

    def print_report(self) -> str:
        """Generate formatted report string."""
        lines = [
            "=" * 60,
            "EVALUATION REPORT",
            "=" * 60,
            f"Timestamp: {self.timestamp}",
            f"Model: {self.model_name}",
            f"Catalog Size: {self.catalog_size}",
            f"Queries Evaluated: {self.num_queries}",
            "",
            "-" * 60,
            "METRICS SUMMARY",
            "-" * 60,
        ]

        for k in sorted(self.recall_at_k.keys()):
            lines.append(f"\n[K = {k}]")
            lines.append(f"  Recall@{k}:    {self.recall_at_k[k]:.4f}")
            lines.append(f"  Precision@{k}: {self.precision_at_k.get(k, 0):.4f}")
            lines.append(f"  MAP@{k}:       {self.map_at_k.get(k, 0):.4f}")
            lines.append(f"  Hit Rate@{k}:  {self.hit_rate_at_k.get(k, 0):.4f} ({self.num_hits.get(k, 0)}/{self.num_queries})")

        lines.append(f"\n[Overall]")
        lines.append(f"  MRR: {self.mrr:.4f}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def save_to_csv(self, path: str):
        """Save query details to CSV for analysis."""
        if self.query_details:
            df = pd.DataFrame(self.query_details)
            df.to_csv(path, index=False)
            logger.info(f"Saved evaluation details to {path}")


# =============================================================================
# EVALUATOR CLASS
# =============================================================================

class Evaluator:
    """
    Comprehensive evaluator for the recommendation engine.

    Usage:
        from src.data import DatasetLoader
        from src.recommender import AssessmentRecommender

        loader = DatasetLoader("data/Gen_AI-Dataset.xlsx")
        train_samples = loader.load_train_set()

        recommender = AssessmentRecommender()
        recommender.load()

        evaluator = Evaluator(recommender)
        report = evaluator.evaluate(train_samples, k_values=[3, 5, 10])

        print(report.print_report())
    """

    def __init__(self, recommender):
        """
        Initialize evaluator.

        Args:
            recommender: AssessmentRecommender instance
        """
        self.recommender = recommender

    def evaluate(
        self,
        samples,  # List[QuerySample] - avoid circular import
        k_values: List[int] = [3, 5, 10],
        verbose: bool = True
    ) -> EvaluationReport:
        """
        Evaluate recommender on a set of samples.

        Args:
            samples: List of QuerySample with query and ground truth URLs
            k_values: K values to compute metrics for
            verbose: Print progress

        Returns:
            EvaluationReport with all metrics
        """
        if verbose:
            logger.info(f"Evaluating on {len(samples)} queries...")

        report = EvaluationReport(
            timestamp=datetime.now().isoformat(),
            model_name=getattr(self.recommender, 'embedding_model', 'unknown'),
            catalog_size=self.recommender.catalog_size,
            num_queries=len(samples)
        )

        # Initialize metric accumulators
        all_true_urls = []
        all_pred_urls = []

        max_k = max(k_values)

        # Process each query
        for i, sample in enumerate(samples):
            if verbose and (i + 1) % 10 == 0:
                logger.info(f"Processing query {i + 1}/{len(samples)}")

            query = sample.query
            true_urls = sample.assessment_urls

            # Get predictions
            pred_urls = self.recommender.recommend_urls(query, top_k=max_k)

            all_true_urls.append(true_urls)
            all_pred_urls.append(pred_urls)

            # Store per-query details
            query_detail = {
                "query": query[:100],  # Truncate
                "true_urls": true_urls,
                "num_true": len(true_urls),
                "predicted_urls_top5": pred_urls[:5]
            }

            for k in k_values:
                r = recall_at_k(true_urls, pred_urls, k)
                query_detail[f"recall@{k}"] = r
                query_detail[f"hit@{k}"] = 1 if r > 0 else 0

            query_detail["rr"] = reciprocal_rank(true_urls, pred_urls)
            report.query_details.append(query_detail)

        # Compute aggregate metrics
        for k in k_values:
            # Recall@K
            recalls = [recall_at_k(t, p, k) for t, p in zip(all_true_urls, all_pred_urls)]
            report.recall_at_k[k] = np.mean(recalls)

            # Precision@K
            precisions = [precision_at_k(t, p, k) for t, p in zip(all_true_urls, all_pred_urls)]
            report.precision_at_k[k] = np.mean(precisions)

            # MAP@K
            report.map_at_k[k] = mean_average_precision(all_true_urls, all_pred_urls, k)

            # Hit Rate@K
            hits = sum(1 for t, p in zip(all_true_urls, all_pred_urls) if recall_at_k(t, p, k) > 0)
            report.hit_rate_at_k[k] = hits / len(samples) if samples else 0
            report.num_hits[k] = hits

        # MRR
        report.mrr = mean_reciprocal_rank(all_true_urls, all_pred_urls)

        if verbose:
            print(report.print_report())

        return report

    def evaluate_from_dataframe(
        self,
        df: pd.DataFrame,
        query_col: str = "Query",
        url_col: str = "Assessment_url",
        k_values: List[int] = [3, 5, 10]
    ) -> EvaluationReport:
        """
        Evaluate from DataFrame directly.

        Args:
            df: DataFrame with queries and URLs
            query_col: Name of query column
            url_col: Name of URL column
            k_values: K values to evaluate

        Returns:
            EvaluationReport
        """
        # Group by query to handle multiple URLs
        @dataclass
        class _Sample:
            query: str
            assessment_urls: List[str]

        grouped = df.groupby(query_col)[url_col].apply(list).reset_index()
        samples = [
            _Sample(query=row[query_col], assessment_urls=row[url_col])
            for _, row in grouped.iterrows()
        ]

        return self.evaluate(samples, k_values)

