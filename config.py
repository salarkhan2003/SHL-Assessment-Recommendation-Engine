"""
Central Configuration for SHL Assessment Recommendation Engine

All configurable parameters in one place for easy tuning and reproducibility.
This configuration file follows best practices for ML research projects.
"""

from pathlib import Path
from typing import Optional

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = DATA_DIR / "embeddings_cache"

# Data files - with fallback for different naming conventions
CATALOG_PATH = DATA_DIR / "shl_catalog.csv"

# Dataset path - try multiple naming conventions
_DATASET_CANDIDATES = [
    "Gen_AI Dataset.xlsx",      # With space
    "Gen_AI-Dataset.xlsx",      # With hyphen
    "GenAI_Dataset.xlsx",       # No separator
    "dataset.xlsx",             # Simple name
]

def find_dataset_path() -> Optional[Path]:
    """Find the dataset file, trying multiple naming conventions."""
    for name in _DATASET_CANDIDATES:
        path = DATA_DIR / name
        if path.exists():
            return path
    return None

DATASET_PATH = find_dataset_path() or DATA_DIR / "Gen_AI Dataset.xlsx"
SUBMISSION_PATH = OUTPUTS_DIR / "submission.csv"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# =============================================================================
# SCRAPER SETTINGS
# =============================================================================
SHL_BASE_URL = "https://www.shl.com"
SHL_CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"

SCRAPER_DELAY = 1.0  # Seconds between requests (be respectful)
SCRAPER_TIMEOUT = 30  # Request timeout in seconds
SCRAPER_MAX_RETRIES = 3

# =============================================================================
# EMBEDDING MODEL CONFIGURATION
# =============================================================================
# Available models (quality vs speed trade-off):
#
# | Model                        | Dim  | Quality | Speed  | Use Case           |
# |------------------------------|------|---------|--------|--------------------|
# | all-MiniLM-L6-v2             | 384  | Good    | Fast   | Quick prototyping  |
# | all-mpnet-base-v2            | 768  | High    | Medium | Production (default)|
# | multi-qa-mpnet-base-dot-v1   | 768  | High    | Medium | Q&A / Search       |
#
EMBEDDING_MODEL = "all-mpnet-base-v2"

# Alternative model for ablation study
EMBEDDING_MODEL_FAST = "all-MiniLM-L6-v2"

# =============================================================================
# RECOMMENDER SETTINGS
# =============================================================================
DEFAULT_TOP_K = 10              # Default number of recommendations
MIN_RELEVANCE_SCORE = 0.0       # Minimum score threshold

# K/P Balance: Ensure recommendations include both assessment types
# When a query spans cognitive (K) and personality (P) domains,
# balance the results to include both types.
KP_BALANCE_ENABLED = True
KP_BALANCE_RATIO = 0.5          # Target 50% K, 50% P when both relevant

# =============================================================================
# EVALUATION SETTINGS
# =============================================================================
# K values for Recall@K evaluation (per SHL assignment: Recall@10 is primary)
EVAL_K_VALUES = [1, 3, 5, 10]
PRIMARY_EVAL_K = 10             # Primary metric for reporting

# =============================================================================
# STREAMLIT UI SETTINGS
# =============================================================================
UI_TITLE = "SHL Assessment Recommendation Engine"
UI_PAGE_ICON = "🎯"
UI_MAX_RECOMMENDATIONS = 20
UI_DEFAULT_RECOMMENDATIONS = 10

# =============================================================================
# LOGGING
# =============================================================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

# =============================================================================
# ASSESSMENT TYPE MAPPING
# =============================================================================
# SHL Assessment Types:
# - K: Knowledge/Cognitive (reasoning, aptitude, technical skills)
# - P: Personality/Behavioral (work style, motivation, situational judgment)
#
ASSESSMENT_TYPE_MAP = {
    "K": [
        "cognitive", "ability", "aptitude", "reasoning", "numerical", "verbal",
        "abstract", "logical", "inductive", "deductive", "mechanical", "spatial",
        "knowledge", "technical", "skills", "coding", "programming", "analytical"
    ],
    "P": [
        "personality", "behavioral", "behaviour", "competency", "motivation",
        "values", "interests", "opq", "situational", "judgment", "leadership",
        "emotional", "interpersonal", "communication", "work style"
    ]
}

