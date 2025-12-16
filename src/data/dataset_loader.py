"""
Dataset Loader Module

Handles loading and preprocessing of the Gen_AI-Dataset.xlsx file
containing Train-Set and Test-Set for evaluation.
"""

import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class QuerySample:
    """A single query sample from the dataset."""
    query: str
    assessment_urls: List[str]  # Can have multiple correct URLs

    @property
    def has_ground_truth(self) -> bool:
        return len(self.assessment_urls) > 0


class DatasetLoader:
    """
    Loads Train-Set and Test-Set from Gen_AI-Dataset.xlsx.

    Expected format:
    - Train-Set sheet: Query, Assessment_url columns (may have multiple URLs per query)
    - Test-Set sheet: Query column

    Usage:
        loader = DatasetLoader("data/Gen_AI-Dataset.xlsx")
        train_samples = loader.load_train_set()
        test_queries = loader.load_test_set()
    """

    def __init__(self, excel_path: str):
        """
        Initialize the dataset loader.

        Args:
            excel_path: Path to Gen_AI-Dataset.xlsx
        """
        self.excel_path = Path(excel_path)
        if not self.excel_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.excel_path}")

        self._train_df: Optional[pd.DataFrame] = None
        self._test_df: Optional[pd.DataFrame] = None

    def load_train_set(self) -> List[QuerySample]:
        """
        Load training set with queries and ground truth URLs.

        Returns:
            List of QuerySample objects
        """
        if self._train_df is None:
            self._train_df = self._load_sheet("Train-Set")

        return self._parse_samples(self._train_df, has_ground_truth=True)

    def load_test_set(self) -> List[str]:
        """
        Load test set queries (no ground truth).

        Returns:
            List of query strings
        """
        if self._test_df is None:
            self._test_df = self._load_sheet("Test-Set")

        query_col = self._find_query_column(self._test_df)
        return self._test_df[query_col].dropna().tolist()

    def load_train_dataframe(self) -> pd.DataFrame:
        """Load raw training DataFrame."""
        if self._train_df is None:
            self._train_df = self._load_sheet("Train-Set")
        return self._train_df.copy()

    def load_test_dataframe(self) -> pd.DataFrame:
        """Load raw test DataFrame."""
        if self._test_df is None:
            self._test_df = self._load_sheet("Test-Set")
        return self._test_df.copy()

    def _load_sheet(self, sheet_name: str) -> pd.DataFrame:
        """Load a specific sheet from the Excel file."""
        try:
            df = pd.read_excel(self.excel_path, sheet_name=sheet_name)
            logger.info(f"Loaded {len(df)} rows from {sheet_name}")
            return df
        except Exception as e:
            logger.error(f"Error loading sheet {sheet_name}: {e}")
            raise

    def _find_query_column(self, df: pd.DataFrame) -> str:
        """Find the query column in DataFrame."""
        # Standardize column names
        df.columns = df.columns.str.strip()

        # Look for query column
        for col in df.columns:
            if "query" in col.lower():
                return col

        # Default to first column
        logger.warning(f"No 'Query' column found, using first column: {df.columns[0]}")
        return df.columns[0]

    def _find_url_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the URL column in DataFrame."""
        df.columns = df.columns.str.strip()

        for col in df.columns:
            col_lower = col.lower()
            if "url" in col_lower or "assessment" in col_lower:
                return col

        return None

    def _parse_samples(self, df: pd.DataFrame, has_ground_truth: bool) -> List[QuerySample]:
        """Parse DataFrame into QuerySample objects."""
        samples = []

        query_col = self._find_query_column(df)
        url_col = self._find_url_column(df) if has_ground_truth else None

        # Group by query to handle multiple URLs per query
        if url_col:
            grouped = df.groupby(query_col)[url_col].apply(list).reset_index()
            for _, row in grouped.iterrows():
                query = str(row[query_col]).strip()
                urls = [str(u).strip() for u in row[url_col] if pd.notna(u) and str(u).strip()]
                samples.append(QuerySample(query=query, assessment_urls=urls))
        else:
            for _, row in df.iterrows():
                query = str(row[query_col]).strip()
                samples.append(QuerySample(query=query, assessment_urls=[]))

        return samples

    def get_stats(self) -> dict:
        """Get dataset statistics."""
        train_samples = self.load_train_set()
        test_queries = self.load_test_set()

        return {
            "train_queries": len(train_samples),
            "train_unique_urls": len(set(url for s in train_samples for url in s.assessment_urls)),
            "test_queries": len(test_queries),
            "avg_urls_per_train_query": sum(len(s.assessment_urls) for s in train_samples) / len(train_samples) if train_samples else 0
        }


def load_train_test_split(excel_path: str) -> Tuple[List[QuerySample], List[str]]:
    """
    Convenience function to load both train and test sets.

    Args:
        excel_path: Path to Gen_AI-Dataset.xlsx

    Returns:
        Tuple of (train_samples, test_queries)
    """
    loader = DatasetLoader(excel_path)
    return loader.load_train_set(), loader.load_test_set()

