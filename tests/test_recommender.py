"""
Tests for Recommender Module

Run with: pytest tests/test_recommender.py -v
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import tempfile
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRecommendationResult:
    """Tests for RecommendationResult dataclass."""

    def test_result_to_dict(self):
        """Test result serialization."""
        from src.recommender.base_recommender import RecommendationResult
        from src.data.catalog_loader import Assessment

        assessment = Assessment(
            name="Test Assessment",
            url="https://example.com/test",
            description="Test description",
            test_type="K",
            remote_testing=True,
            adaptive_irt=False,
            duration="25 minutes"
        )

        result = RecommendationResult(
            assessment=assessment,
            score=0.85,
            rank=1,
            explanation="High relevance match"
        )

        d = result.to_dict()
        assert d["name"] == "Test Assessment"
        assert d["relevance_score"] == 0.85
        assert d["rank"] == 1
        assert d["explanation"] == "High relevance match"


class TestAssessmentRecommender:
    """Tests for AssessmentRecommender class."""

    @pytest.fixture
    def sample_catalog(self, tmp_path):
        """Create a sample catalog for testing."""
        data = [
            {
                "name": "Numerical Reasoning Test",
                "url": "https://example.com/numerical",
                "description": "Assesses numerical and analytical reasoning abilities",
                "test_type": "K",
                "remote_testing": True,
                "adaptive_irt": True,
                "duration": "20 minutes"
            },
            {
                "name": "OPQ32 Personality Assessment",
                "url": "https://example.com/opq32",
                "description": "Comprehensive personality questionnaire for workplace behavior",
                "test_type": "P",
                "remote_testing": True,
                "adaptive_irt": False,
                "duration": "25 minutes"
            },
            {
                "name": "Verbal Reasoning Test",
                "url": "https://example.com/verbal",
                "description": "Evaluates verbal comprehension and critical thinking",
                "test_type": "K",
                "remote_testing": True,
                "adaptive_irt": True,
                "duration": "18 minutes"
            },
            {
                "name": "Leadership Assessment",
                "url": "https://example.com/leadership",
                "description": "Assesses leadership potential and management style",
                "test_type": "K, P",
                "remote_testing": True,
                "adaptive_irt": False,
                "duration": "45 minutes"
            }
        ]

        df = pd.DataFrame(data)
        catalog_path = tmp_path / "test_catalog.csv"
        df.to_csv(catalog_path, index=False)
        return str(catalog_path)

    def test_recommender_load(self, sample_catalog, tmp_path):
        """Test recommender loading."""
        from src.recommender import AssessmentRecommender

        recommender = AssessmentRecommender(
            catalog_path=sample_catalog,
            embedding_model="all-MiniLM-L6-v2",  # Smaller model for testing
            cache_dir=str(tmp_path / "cache")
        )
        recommender.load()

        assert recommender.catalog_size == 4
        assert recommender._is_loaded

    def test_recommend_returns_results(self, sample_catalog, tmp_path):
        """Test that recommend returns results."""
        from src.recommender import AssessmentRecommender

        recommender = AssessmentRecommender(
            catalog_path=sample_catalog,
            embedding_model="all-MiniLM-L6-v2",
            cache_dir=str(tmp_path / "cache")
        )
        recommender.load()

        results = recommender.recommend(
            "Looking for numerical reasoning assessment",
            top_k=3
        )

        assert len(results) > 0
        assert len(results) <= 3
        assert all(r.score >= 0 for r in results)
        assert results[0].rank == 1

    def test_recommend_urls(self, sample_catalog, tmp_path):
        """Test URL-only recommendations."""
        from src.recommender import AssessmentRecommender

        recommender = AssessmentRecommender(
            catalog_path=sample_catalog,
            embedding_model="all-MiniLM-L6-v2",
            cache_dir=str(tmp_path / "cache")
        )
        recommender.load()

        urls = recommender.recommend_urls("personality assessment", top_k=2)

        assert isinstance(urls, list)
        assert len(urls) <= 2
        assert all(isinstance(u, str) for u in urls)

    def test_query_intent_analysis(self, sample_catalog, tmp_path):
        """Test query intent classification."""
        from src.recommender import AssessmentRecommender

        recommender = AssessmentRecommender(
            catalog_path=sample_catalog,
            embedding_model="all-MiniLM-L6-v2",
            cache_dir=str(tmp_path / "cache")
        )

        # Cognitive query
        assert recommender._analyze_query_intent("numerical reasoning aptitude test") == "K"

        # Personality query
        assert recommender._analyze_query_intent("leadership personality behavioral") == "P"

        # Mixed query
        assert recommender._analyze_query_intent(
            "analytical reasoning and communication leadership skills"
        ) == "both"


class TestCatalogLoader:
    """Tests for CatalogLoader."""

    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create sample catalog CSV."""
        data = [
            {"name": "Test 1", "url": "url1", "description": "Desc 1",
             "test_type": "K", "remote_testing": True, "adaptive_irt": False, "duration": "20 min"},
            {"name": "Test 2", "url": "url2", "description": "Desc 2",
             "test_type": "P", "remote_testing": False, "adaptive_irt": True, "duration": "30 min"}
        ]
        path = tmp_path / "catalog.csv"
        pd.DataFrame(data).to_csv(path, index=False)
        return str(path)

    def test_load_catalog(self, sample_csv):
        """Test catalog loading."""
        from src.data import CatalogLoader

        loader = CatalogLoader(sample_csv)
        assessments = loader.load()

        assert len(assessments) == 2
        assert assessments[0].name == "Test 1"
        assert assessments[0].is_cognitive
        assert assessments[1].is_personality

    def test_get_combined_texts(self, sample_csv):
        """Test combined text generation."""
        from src.data import CatalogLoader

        loader = CatalogLoader(sample_csv)
        texts = loader.get_combined_texts()

        assert len(texts) == 2
        assert "Test 1" in texts[0]
        assert "Cognitive" in texts[0] or "knowledge" in texts[0].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

