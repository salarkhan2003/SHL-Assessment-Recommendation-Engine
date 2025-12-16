"""
Tests for Evaluation Module

Run with: pytest tests/test_evaluation.py -v
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    average_precision,
    mean_average_precision,
    reciprocal_rank,
    mean_reciprocal_rank
)


class TestRecallAtK:
    """Tests for Recall@K metric."""
    
    def test_single_label_hit(self):
        """Test recall when true label is in predictions."""
        true_urls = ["url1"]
        pred_urls = ["url1", "url2", "url3"]
        
        assert recall_at_k(true_urls, pred_urls, k=3) == 1.0
        assert recall_at_k(true_urls, pred_urls, k=1) == 1.0
    
    def test_single_label_miss(self):
        """Test recall when true label is not in top-k."""
        true_urls = ["url4"]
        pred_urls = ["url1", "url2", "url3", "url4"]
        
        assert recall_at_k(true_urls, pred_urls, k=3) == 0.0
        assert recall_at_k(true_urls, pred_urls, k=4) == 1.0
    
    def test_multi_label(self):
        """Test recall with multiple true labels."""
        true_urls = ["url1", "url3", "url5"]
        pred_urls = ["url1", "url2", "url3", "url4", "url5"]
        
        # 2 out of 3 in top-3
        assert recall_at_k(true_urls, pred_urls, k=3) == 2/3
        
        # All 3 in top-5
        assert recall_at_k(true_urls, pred_urls, k=5) == 1.0
    
    def test_empty_true(self):
        """Test recall with empty ground truth."""
        assert recall_at_k([], ["url1", "url2"], k=2) == 0.0
    
    def test_empty_predictions(self):
        """Test recall with empty predictions."""
        assert recall_at_k(["url1"], [], k=5) == 0.0


class TestPrecisionAtK:
    """Tests for Precision@K metric."""
    
    def test_all_relevant(self):
        """Test precision when all predictions are relevant."""
        true_urls = ["url1", "url2", "url3"]
        pred_urls = ["url1", "url2", "url3"]
        
        assert precision_at_k(true_urls, pred_urls, k=3) == 1.0
    
    def test_none_relevant(self):
        """Test precision when no predictions are relevant."""
        true_urls = ["url4"]
        pred_urls = ["url1", "url2", "url3"]
        
        assert precision_at_k(true_urls, pred_urls, k=3) == 0.0
    
    def test_partial_relevant(self):
        """Test precision with partial relevance."""
        true_urls = ["url1", "url3"]
        pred_urls = ["url1", "url2", "url3", "url4"]
        
        # 2 relevant out of 4
        assert precision_at_k(true_urls, pred_urls, k=4) == 0.5


class TestReciprocalRank:
    """Tests for Reciprocal Rank metric."""
    
    def test_first_position(self):
        """Test RR when relevant at position 1."""
        assert reciprocal_rank(["url1"], ["url1", "url2", "url3"]) == 1.0
    
    def test_second_position(self):
        """Test RR when relevant at position 2."""
        assert reciprocal_rank(["url2"], ["url1", "url2", "url3"]) == 0.5
    
    def test_third_position(self):
        """Test RR when relevant at position 3."""
        assert reciprocal_rank(["url3"], ["url1", "url2", "url3"]) == 1/3
    
    def test_not_found(self):
        """Test RR when relevant item not found."""
        assert reciprocal_rank(["url4"], ["url1", "url2", "url3"]) == 0.0
    
    def test_multiple_relevant(self):
        """Test RR with multiple relevant items (first one counts)."""
        # url2 appears first
        assert reciprocal_rank(["url2", "url3"], ["url1", "url2", "url3"]) == 0.5


class TestAveragePrecision:
    """Tests for Average Precision metric."""
    
    def test_perfect_ranking(self):
        """Test AP with perfect ranking."""
        true_urls = ["url1", "url2"]
        pred_urls = ["url1", "url2", "url3", "url4"]
        
        # P@1 = 1, P@2 = 1 -> AP = (1 + 1) / 2 = 1.0
        assert average_precision(true_urls, pred_urls, k=4) == 1.0
    
    def test_imperfect_ranking(self):
        """Test AP with imperfect ranking."""
        true_urls = ["url2", "url4"]
        pred_urls = ["url1", "url2", "url3", "url4"]
        
        # P@2 = 0.5 (hit), P@4 = 0.5 (hit)
        # AP = (0.5 + 0.5) / 2 = 0.5
        assert average_precision(true_urls, pred_urls, k=4) == 0.5


class TestMeanMetrics:
    """Tests for mean metrics across queries."""
    
    def test_mrr(self):
        """Test Mean Reciprocal Rank."""
        all_true = [["url1"], ["url2"], ["url3"]]
        all_pred = [
            ["url1", "url2"],  # RR = 1
            ["url1", "url2"],  # RR = 0.5
            ["url1", "url2", "url3"]  # RR = 1/3
        ]
        
        expected = (1 + 0.5 + 1/3) / 3
        assert abs(mean_reciprocal_rank(all_true, all_pred) - expected) < 0.001
    
    def test_map(self):
        """Test Mean Average Precision."""
        all_true = [["url1"], ["url1", "url2"]]
        all_pred = [
            ["url1", "url2", "url3"],  # AP = 1.0
            ["url1", "url2", "url3"]   # AP = 1.0
        ]
        
        assert mean_average_precision(all_true, all_pred, k=3) == 1.0


class TestEvaluationReport:
    """Tests for EvaluationReport class."""
    
    def test_report_generation(self):
        """Test report string generation."""
        from src.evaluation.metrics import EvaluationReport
        
        report = EvaluationReport(
            recall_at_k={5: 0.75, 10: 0.85},
            precision_at_k={5: 0.15, 10: 0.085},
            map_at_k={5: 0.70, 10: 0.72},
            mrr=0.65,
            hit_rate_at_k={5: 0.80, 10: 0.90},
            num_queries=100,
            num_hits={5: 80, 10: 90},
            timestamp="2024-01-01T00:00:00",
            model_name="test-model",
            catalog_size=50
        )
        
        report_str = report.print_report()
        
        assert "EVALUATION REPORT" in report_str
        assert "Recall@5" in report_str
        assert "Recall@10" in report_str
        assert "MRR" in report_str
        assert "0.75" in report_str or "0.7500" in report_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

