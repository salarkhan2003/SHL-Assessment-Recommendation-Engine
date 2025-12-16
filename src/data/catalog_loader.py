"""
Catalog Loader Module

Loads the scraped SHL assessment catalog from CSV and prepares it
for the recommendation engine with proper K/P type classification.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class Assessment:
    """
    Data class representing an SHL assessment.

    Attributes:
        name: Assessment name/title
        url: Full URL to assessment page
        description: Brief description
        test_type: "K" (Cognitive), "P" (Personality), or "K, P" (both)
        remote_testing: Supports remote/unproctored testing
        adaptive_irt: Uses adaptive/IRT technology
        duration: Test duration (e.g., "25 minutes")
        combined_text: Pre-built text for embedding (auto-generated)
    """
    name: str
    url: str
    description: str = ""
    test_type: str = "K"  # "K", "P", or "K, P"
    remote_testing: bool = False
    adaptive_irt: bool = False
    duration: Optional[str] = None
    combined_text: str = ""

    def __post_init__(self):
        """Generate combined text if not provided."""
        if not self.combined_text:
            self.combined_text = self._build_combined_text()

    def _build_combined_text(self) -> str:
        """
        Build rich combined text for embedding generation.

        This text is used to compute semantic embeddings for matching.
        Includes all relevant metadata for better matching.
        """
        parts = [self.name]

        if self.description:
            parts.append(self.description)

        # Add type information with descriptive text
        if "K" in self.test_type:
            parts.append("Cognitive ability assessment for knowledge and reasoning")
        if "P" in self.test_type:
            parts.append("Personality behavioral assessment for workplace behavior")

        # Add feature information
        if self.remote_testing:
            parts.append("Supports remote online unproctored testing")
        if self.adaptive_irt:
            parts.append("Adaptive IRT assessment that adjusts difficulty")
        if self.duration:
            parts.append(f"Duration: {self.duration}")

        return " | ".join(parts)

    @property
    def is_cognitive(self) -> bool:
        """Check if this is a cognitive/knowledge assessment."""
        return "K" in self.test_type

    @property
    def is_personality(self) -> bool:
        """Check if this is a personality/behavioral assessment."""
        return "P" in self.test_type

    @property
    def type_display(self) -> str:
        """Get display-friendly type string."""
        types = []
        if "K" in self.test_type:
            types.append("Cognitive")
        if "P" in self.test_type:
            types.append("Personality")
        return " & ".join(types) if types else "Unknown"

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "url": self.url,
            "description": self.description,
            "test_type": self.test_type,
            "remote_testing": self.remote_testing,
            "adaptive_irt": self.adaptive_irt,
            "duration": self.duration,
        }


class CatalogLoader:
    """
    Loads and preprocesses the SHL assessment catalog.

    Usage:
        loader = CatalogLoader("data/shl_catalog.csv")
        assessments = loader.load()

        # Filter by type
        cognitive = loader.get_cognitive_assessments()
        personality = loader.get_personality_assessments()
    """

    def __init__(self, catalog_path: str = "data/shl_catalog.csv"):
        """
        Initialize the catalog loader.

        Args:
            catalog_path: Path to the scraped catalog CSV file.
        """
        self.catalog_path = Path(catalog_path)
        self._df: Optional[pd.DataFrame] = None
        self._assessments: Optional[List[Assessment]] = None

    def load(self) -> List[Assessment]:
        """
        Load the catalog and return Assessment objects.

        Returns:
            List of Assessment dataclass instances

        Raises:
            FileNotFoundError: If catalog CSV doesn't exist
        """
        if self._assessments is not None:
            return self._assessments

        if not self.catalog_path.exists():
            raise FileNotFoundError(
                f"Catalog not found at {self.catalog_path}. "
                "Please run: python scripts/scrape_catalog.py"
            )

        self._df = pd.read_csv(self.catalog_path)
        self._assessments = self._parse_assessments()

        logger.info(f"Loaded {len(self._assessments)} assessments from catalog")
        return self._assessments

    def _parse_assessments(self) -> List[Assessment]:
        """Parse DataFrame rows into Assessment objects."""
        assessments = []

        for _, row in self._df.iterrows():
            assessment = Assessment(
                name=str(row.get("name", "")).strip(),
                url=str(row.get("url", "")).strip(),
                description=str(row.get("description", "")).strip(),
                test_type=str(row.get("test_type", "K")).strip(),
                remote_testing=self._parse_bool(row.get("remote_testing", False)),
                adaptive_irt=self._parse_bool(row.get("adaptive_irt", False)),
                duration=self._parse_duration(row.get("duration")),
            )

            if assessment.name and assessment.url:
                assessments.append(assessment)

        return assessments

    @staticmethod
    def _parse_bool(value) -> bool:
        """Parse various boolean representations."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "yes", "1", "y")
        if pd.isna(value):
            return False
        return bool(value)

    @staticmethod
    def _parse_duration(value) -> Optional[str]:
        """Parse duration field."""
        if pd.isna(value) or not value:
            return None
        return str(value).strip()

    def get_dataframe(self) -> pd.DataFrame:
        """Get raw DataFrame."""
        if self._df is None:
            self.load()
        return self._df.copy()

    def get_combined_texts(self) -> List[str]:
        """Get combined text fields for embedding."""
        assessments = self.load()
        return [a.combined_text for a in assessments]

    def get_urls(self) -> List[str]:
        """Get list of all assessment URLs."""
        assessments = self.load()
        return [a.url for a in assessments]

    def get_cognitive_assessments(self) -> List[Assessment]:
        """Get only cognitive (K-type) assessments."""
        return [a for a in self.load() if a.is_cognitive]

    def get_personality_assessments(self) -> List[Assessment]:
        """Get only personality (P-type) assessments."""
        return [a for a in self.load() if a.is_personality]

    def find_by_url(self, url: str) -> Optional[Assessment]:
        """Find assessment by URL."""
        for a in self.load():
            if a.url == url:
                return a
        return None

    def get_stats(self) -> Dict:
        """Get catalog statistics."""
        assessments = self.load()
        return {
            "total": len(assessments),
            "cognitive_only": sum(1 for a in assessments if a.is_cognitive and not a.is_personality),
            "personality_only": sum(1 for a in assessments if a.is_personality and not a.is_cognitive),
            "both": sum(1 for a in assessments if a.is_cognitive and a.is_personality),
            "remote_testing": sum(1 for a in assessments if a.remote_testing),
            "adaptive_irt": sum(1 for a in assessments if a.adaptive_irt),
        }

