"""
Submission Generation Script for SHL Assessment Recommendation Engine

Generates predictions for the Test-Set in the required submission format.
Optionally runs evaluation on Train-Set first to verify model performance.

Usage:
    python scripts/generate_submission.py
    python scripts/generate_submission.py --evaluate      # Also evaluate on Train-Set
    python scripts/generate_submission.py --top-k 10      # Specify number of predictions

Output Format (submission.csv):
    query,predictions
    "Job description text...", "url1 | url2 | url3 | ..."

Note: The predictions column contains pipe-separated URLs, ordered by relevance.
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import (
    DATASET_PATH, CATALOG_PATH, OUTPUTS_DIR, SUBMISSION_PATH,
    EMBEDDING_MODEL, DEFAULT_TOP_K, EVAL_K_VALUES,
    find_dataset_path
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def print_banner(title: str):
    """Print a formatted banner."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_prerequisites() -> tuple:
    """Check that all required files exist."""
    # Check dataset
    dataset_path = find_dataset_path()
    if not dataset_path or not dataset_path.exists():
        print("\n[ERROR] Dataset not found!")
        print("\n   Please ensure Gen_AI Dataset.xlsx is in the data/ folder.")
        print(f"   Looked for: {DATASET_PATH}")
        print("\n   Files in data/ folder:")
        for f in Path(project_root / "data").glob("*"):
            print(f"     - {f.name}")
        sys.exit(1)

    # Check catalog
    if not CATALOG_PATH.exists():
        print("\n[ERROR] Catalog not found!")
        print(f"\n   Expected: {CATALOG_PATH}")
        print("\n   Please run the scraper first:")
        print("     python scripts/scrape_catalog.py --sample")
        sys.exit(1)

    return dataset_path, CATALOG_PATH


def generate_submission(
    recommender,
    test_queries: list,
    top_k: int,
    output_path: Path
) -> None:
    """
    Generate submission CSV for Test-Set.

    Args:
        recommender: Loaded AssessmentRecommender instance
        test_queries: List of query strings from Test-Set
        top_k: Number of predictions per query
        output_path: Path to save submission CSV
    """
    import pandas as pd
    from tqdm import tqdm

    print(f"\n[GEN] Generating predictions (top-{top_k} per query)...")

    results = []

    for query in tqdm(test_queries, desc="Processing queries", unit="query"):
        # Get recommendations
        predictions = recommender.recommend_urls(query, top_k=top_k)

        # Format as pipe-separated string (required format)
        predictions_str = " | ".join(predictions)

        results.append({
            "query": query,
            "predictions": predictions_str
        })

    # Create DataFrame and save
    submission_df = pd.DataFrame(results)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)

    print(f"\n[OK] Submission saved to: {output_path}")
    print(f"   Total queries: {len(submission_df)}")
    print(f"   Predictions per query: {top_k}")

    # Show sample predictions
    print(f"\n[SAMPLE] Sample Predictions (first 3 queries):")
    print("-" * 70)

    for i, row in submission_df.head(3).iterrows():
        query_preview = row['query'][:65] + "..." if len(row['query']) > 65 else row['query']
        print(f"\nQuery {i+1}: \"{query_preview}\"")

        urls = row['predictions'].split(' | ')
        print(f"Top-3 predictions:")
        for j, url in enumerate(urls[:3], 1):
            # Extract assessment name from URL
            name = url.split('/')[-2].replace('-', ' ').title() if '/' in url else url
            print(f"  {j}. {name}")
            print(f"     {url}")


def run_evaluation(recommender, dataset_path: Path):
    """Run evaluation on Train-Set."""
    from src.data import DatasetLoader
    from src.evaluation import Evaluator

    print_banner("Evaluation on Train-Set")

    loader = DatasetLoader(str(dataset_path))
    train_samples = loader.load_train_set()
    print(f"Train queries: {len(train_samples)}")

    evaluator = Evaluator(recommender)
    report = evaluator.evaluate(train_samples, k_values=EVAL_K_VALUES, verbose=False)

    print(report.print_report())

    # Save evaluation details
    eval_path = OUTPUTS_DIR / "evaluation_with_submission.csv"
    report.save_to_csv(str(eval_path))
    print(f"[SAVE] Evaluation details saved to: {eval_path}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Generate submission.csv for SHL Assessment Test-Set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_submission.py                 # Basic generation
  python scripts/generate_submission.py --evaluate      # With Train-Set evaluation
  python scripts/generate_submission.py --top-k 5       # Top-5 predictions
  python scripts/generate_submission.py -o my_submission.csv  # Custom output

Submission Format:
  query,predictions
  "Looking for a software...", "url1 | url2 | url3 | ..."
        """
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=None,
        help="Path to Gen_AI Dataset.xlsx (auto-detected if not specified)"
    )
    parser.add_argument(
        "--catalog", "-c",
        type=str,
        default=str(CATALOG_PATH),
        help=f"Path to catalog CSV (default: {CATALOG_PATH})"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(SUBMISSION_PATH),
        help=f"Output submission CSV (default: {SUBMISSION_PATH})"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of predictions per query (default: {DEFAULT_TOP_K})"
    )
    parser.add_argument(
        "--evaluate", "-e",
        action="store_true",
        help="Also run evaluation on Train-Set before generating submission"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=EMBEDDING_MODEL,
        help=f"Embedding model (default: {EMBEDDING_MODEL})"
    )

    args = parser.parse_args()

    print_banner("SHL Assessment Recommender - Submission Generator")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check prerequisites
    if args.dataset:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            print(f"\n[ERROR] Dataset not found: {dataset_path}")
            sys.exit(1)
    else:
        dataset_path, _ = check_prerequisites()

    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        print(f"\n[ERROR] Catalog not found: {catalog_path}")
        print("   Run: python scripts/scrape_catalog.py --sample")
        sys.exit(1)

    print(f"\n[OK] Dataset: {dataset_path.name}")
    print(f"[OK] Catalog: {catalog_path.name}")

    # Import heavy dependencies
    from src.recommender import AssessmentRecommender
    from src.data import DatasetLoader

    # Load recommender
    print(f"\n[LOAD] Loading recommender...")
    print(f"   Model: {args.model}")

    recommender = AssessmentRecommender(
        catalog_path=str(catalog_path),
        embedding_model=args.model,
        cache_dir=str(project_root / "data" / "embeddings_cache")
    )
    recommender.load()

    stats = recommender.catalog_stats
    print(f"   Catalog: {stats['total']} assessments")
    print(f"   - Cognitive (K): {stats['cognitive_only'] + stats['both']}")
    print(f"   - Personality (P): {stats['personality_only'] + stats['both']}")

    # Optional: Run evaluation on Train-Set first
    if args.evaluate:
        run_evaluation(recommender, dataset_path)

    # Load Test-Set
    print_banner("Generating Submission")

    loader = DatasetLoader(str(dataset_path))
    test_queries = loader.load_test_set()
    print(f"Test queries: {len(test_queries)}")

    # Generate submission
    output_path = Path(args.output)
    generate_submission(recommender, test_queries, args.top_k, output_path)

    # Final summary
    print("\n" + "=" * 70)
    print("  [DONE] Submission Generation Complete!")
    print("=" * 70)
    print(f"\n[OUTPUT] Output file: {output_path}")
    print(f"[INFO] Queries processed: {len(test_queries)}")
    print(f"[INFO] Predictions per query: {args.top_k}")

    if args.evaluate:
        print("\n[TIP] Review evaluation results to ensure model performance is acceptable.")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

