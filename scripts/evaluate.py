"""
Evaluation Script for SHL Assessment Recommendation Engine

Evaluates the recommender on the Train-Set and generates a comprehensive report.
Primary metric: Recall@10 (as specified in SHL assignment)

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --model all-MiniLM-L6-v2  # Test different model
    python scripts/evaluate.py --compare  # Compare two embedding models

Output:
    - Console: Formatted evaluation report
    - File: outputs/evaluation_report.csv (per-query details)
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
    DATASET_PATH, CATALOG_PATH, OUTPUTS_DIR,
    EVAL_K_VALUES, EMBEDDING_MODEL, EMBEDDING_MODEL_FAST,
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
    """
    Check that all required files exist.
    Returns (dataset_path, catalog_path) or raises informative errors.
    """
    # Check dataset
    dataset_path = find_dataset_path()
    if not dataset_path or not dataset_path.exists():
        print("\n[ERROR] Dataset not found!")
        print("\n   Please ensure Gen_AI Dataset.xlsx is in the data/ folder.")
        print(f"   Expected location: {DATASET_PATH}")
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


def run_evaluation(
    dataset_path: Path,
    catalog_path: Path,
    model_name: str,
    k_values: list,
    output_dir: Path
) -> dict:
    """
    Run evaluation and return results.

    Returns:
        dict with evaluation metrics
    """
    # Import here to avoid loading heavy dependencies during arg parsing
    from src.recommender import AssessmentRecommender
    from src.data import DatasetLoader
    from src.evaluation import Evaluator

    print(f"\n[LOAD] Loading recommender...")
    print(f"   Model: {model_name}")
    print(f"   Catalog: {catalog_path}")

    recommender = AssessmentRecommender(
        catalog_path=str(catalog_path),
        embedding_model=model_name,
        cache_dir=str(project_root / "data" / "embeddings_cache")
    )
    recommender.load()

    stats = recommender.catalog_stats
    print(f"   Assessments: {stats['total']} total")
    print(f"   - Cognitive (K): {stats['cognitive_only'] + stats['both']}")
    print(f"   - Personality (P): {stats['personality_only'] + stats['both']}")
    print(f"   - Both (K+P): {stats['both']}")

    print(f"\n[DATA] Loading dataset: {dataset_path.name}")
    loader = DatasetLoader(str(dataset_path))
    train_samples = loader.load_train_set()
    print(f"   Train queries: {len(train_samples)}")

    # Show sample queries
    print(f"\n   Sample queries:")
    for sample in train_samples[:3]:
        query_preview = sample.query[:60] + "..." if len(sample.query) > 60 else sample.query
        print(f"     - \"{query_preview}\"")

    print(f"\n[EVAL] Running evaluation (K = {k_values})...")
    evaluator = Evaluator(recommender)
    report = evaluator.evaluate(train_samples, k_values=k_values, verbose=False)

    # Print results
    print(report.print_report())

    # Save detailed results
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save per-query details
    details_path = output_dir / f"evaluation_details_{timestamp}.csv"
    report.save_to_csv(str(details_path))
    print(f"\n[SAVE] Detailed results saved to: {details_path}")

    # Save summary
    summary_path = output_dir / "evaluation_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(report.print_report())
    print(f"[SAVE] Summary saved to: {summary_path}")

    return {
        "model": model_name,
        "recall_at_k": report.recall_at_k,
        "map_at_k": report.map_at_k,
        "mrr": report.mrr,
        "report": report
    }


def run_comparison(dataset_path: Path, catalog_path: Path, k_values: list):
    """
    Compare two embedding models (ablation study).
    Demonstrates research mindset by comparing approaches.
    """
    print_banner("MODEL COMPARISON (Ablation Study)")

    print("\nComparing embedding models:")
    print(f"  Model A: {EMBEDDING_MODEL} (high quality, slower)")
    print(f"  Model B: {EMBEDDING_MODEL_FAST} (faster, lower quality)")

    results = {}

    for model_name in [EMBEDDING_MODEL, EMBEDDING_MODEL_FAST]:
        print_banner(f"Evaluating: {model_name}")
        results[model_name] = run_evaluation(
            dataset_path, catalog_path, model_name, k_values,
            OUTPUTS_DIR / "comparison"
        )

    # Print comparison summary
    print_banner("COMPARISON SUMMARY")

    print("\n" + "-" * 70)
    print(f"{'Metric':<25} {'all-mpnet-base-v2':>20} {'all-MiniLM-L6-v2':>20}")
    print("-" * 70)

    for k in k_values:
        r1 = results[EMBEDDING_MODEL]["recall_at_k"].get(k, 0)
        r2 = results[EMBEDDING_MODEL_FAST]["recall_at_k"].get(k, 0)
        diff = r1 - r2
        diff_str = f"({'+' if diff > 0 else ''}{diff:.4f})"
        print(f"Recall@{k:<19} {r1:>20.4f} {r2:>20.4f} {diff_str}")

    mrr1 = results[EMBEDDING_MODEL]["mrr"]
    mrr2 = results[EMBEDDING_MODEL_FAST]["mrr"]
    print(f"{'MRR':<25} {mrr1:>20.4f} {mrr2:>20.4f}")
    print("-" * 70)

    # Recommendation
    better_model = EMBEDDING_MODEL if mrr1 >= mrr2 else EMBEDDING_MODEL_FAST
    print(f"\n[RESULT] Recommended model: {better_model}")

    if mrr1 > mrr2:
        improvement = ((mrr1 - mrr2) / mrr2) * 100 if mrr2 > 0 else 0
        print(f"   {EMBEDDING_MODEL} achieves {improvement:.1f}% higher MRR")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SHL Assessment Recommender on Train-Set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluate.py                    # Standard evaluation
  python scripts/evaluate.py --model all-MiniLM-L6-v2  # Different model
  python scripts/evaluate.py --compare          # Compare two models
  python scripts/evaluate.py --k 1 3 5 10       # Custom K values
        """
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=EMBEDDING_MODEL,
        help=f"Embedding model (default: {EMBEDDING_MODEL})"
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=EVAL_K_VALUES,
        help=f"K values for Recall@K (default: {EVAL_K_VALUES})"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare two embedding models (ablation study)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(OUTPUTS_DIR),
        help=f"Output directory (default: {OUTPUTS_DIR})"
    )

    args = parser.parse_args()

    print_banner("SHL Assessment Recommender - Evaluation")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check prerequisites
    dataset_path, catalog_path = check_prerequisites()
    print(f"\n[OK] Dataset found: {dataset_path.name}")
    print(f"[OK] Catalog found: {catalog_path.name}")

    output_dir = Path(args.output)

    if args.compare:
        run_comparison(dataset_path, catalog_path, args.k)
    else:
        run_evaluation(dataset_path, catalog_path, args.model, args.k, output_dir)

    print("\n" + "=" * 70)
    print("  [DONE] Evaluation Complete!")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
