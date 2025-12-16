"""
Tests for Scraper Module

Run with: pytest tests/test_scraper.py -v
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scraper.shl_scraper import (
    SHLAssessment,
    SHLCatalogScraper,
    create_sample_catalog
)


class TestSHLAssessment:
    """Tests for SHLAssessment dataclass."""

    def test_assessment_creation(self):
        """Test basic assessment creation."""
        assessment = SHLAssessment(
            name="Test Assessment",
            url="https://example.com/test",
            description="Test description",
            test_type=["K"]
        )

        assert assessment.name == "Test Assessment"
        assert assessment.url == "https://example.com/test"
        assert assessment.is_cognitive
        assert not assessment.is_personality

    def test_type_string(self):
        """Test type string generation."""
        # Single type
        k_only = SHLAssessment(name="K Test", url="", test_type=["K"])
        assert k_only.type_string == "K"

        # Both types
        both = SHLAssessment(name="Both", url="", test_type=["K", "P"])
        assert both.type_string == "K, P"

        # Empty
        empty = SHLAssessment(name="Empty", url="", test_type=[])
        assert empty.type_string == "Unknown"

    def test_to_dict(self):
        """Test dictionary conversion."""
        assessment = SHLAssessment(
            name="Test",
            url="https://example.com",
            description="Desc",
            test_type=["K", "P"],
            remote_testing=True,
            duration="25 minutes"
        )

        d = assessment.to_dict()
        assert d["name"] == "Test"
        assert d["test_type"] == "K, P"
        assert d["remote_testing"] == True
        assert d["duration"] == "25 minutes"


class TestSHLCatalogScraper:
    """Tests for SHLCatalogScraper class."""

    def test_scraper_initialization(self):
        """Test scraper initialization."""
        scraper = SHLCatalogScraper(delay_seconds=0.5)
        assert scraper.delay == 0.5
        assert scraper.timeout == 30

    def test_normalize_url(self):
        """Test URL normalization."""
        scraper = SHLCatalogScraper()

        # Relative URL
        assert scraper._normalize_url("/test") == "https://www.shl.com/test"

        # Absolute URL
        assert scraper._normalize_url("https://other.com/test") == "https://other.com/test"

        # Invalid URLs
        assert scraper._normalize_url("javascript:void(0)") is None
        assert scraper._normalize_url("mailto:test@test.com") is None
        assert scraper._normalize_url("#anchor") is None

    def test_is_assessment_url(self):
        """Test assessment URL detection."""
        scraper = SHLCatalogScraper()

        # Valid assessment URLs
        valid_urls = [
            "https://www.shl.com/solutions/products/assessments/verify/",
            "https://www.shl.com/solutions/products/product-catalog/view/test/"
        ]
        for url in valid_urls:
            assert scraper._is_assessment_url(url), f"Should be valid: {url}"

        # Invalid URLs
        invalid_urls = [
            "https://www.shl.com/solutions/products/product-catalog/",
            "https://www.shl.com/solutions/products/product-catalog/?page=2",
            "https://other.com/test",
            "#anchor"
        ]
        for url in invalid_urls:
            assert not scraper._is_assessment_url(url), f"Should be invalid: {url}"

    def test_is_bundle_detection(self):
        """Test bundled solution detection."""
        scraper = SHLCatalogScraper()

        bundle = SHLAssessment(
            name="Pre-packaged Solution",
            url="",
            description="Complete bundle for hiring"
        )
        assert scraper._is_bundle(bundle)

        individual = SHLAssessment(
            name="Verify G+",
            url="",
            description="Cognitive ability test"
        )
        assert not scraper._is_bundle(individual)


class TestSampleCatalog:
    """Tests for sample catalog generation."""

    def test_create_sample_catalog(self):
        """Test sample catalog creation."""
        df = create_sample_catalog()

        # Check structure
        assert len(df) > 10
        assert "name" in df.columns
        assert "url" in df.columns
        assert "test_type" in df.columns
        assert "remote_testing" in df.columns

        # Check K/P distribution
        type_counts = df["test_type"].value_counts()
        assert "K" in type_counts.index or "K, P" in type_counts.index
        assert "P" in type_counts.index or "K, P" in type_counts.index

    def test_sample_catalog_has_required_fields(self):
        """Test that sample catalog has all required fields."""
        df = create_sample_catalog()

        required_columns = ["name", "url", "description", "test_type",
                          "remote_testing", "adaptive_irt", "duration"]

        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

        # No empty names or URLs
        assert df["name"].notna().all()
        assert (df["name"] != "").all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

