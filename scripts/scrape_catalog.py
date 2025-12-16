"""
Script to scrape SHL product catalog.

Usage:
    python scripts/scrape_catalog.py
    python scripts/scrape_catalog.py --output data/shl_catalog.csv
    python scripts/scrape_catalog.py --sample  # Create sample catalog for testing
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import directly from the module file to avoid loading heavy dependencies
# when only creating sample data
from src.scraper.shl_scraper import SHLCatalogScraper, create_sample_catalog


def main():
    parser = argparse.ArgumentParser(
        description="Scrape SHL product catalog or create sample data"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/shl_catalog.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=20,
        help="Maximum number of catalog pages to scrape"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between requests in seconds"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Create sample catalog instead of scraping (for testing)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SHL Product Catalog Scraper")
    print("=" * 60)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.sample:
        print("Creating sample catalog for testing...")
        df = create_sample_catalog()
        df.to_csv(output_path, index=False)
        print(f"\nSaved {len(df)} sample assessments to {output_path}")
        print(f"\nTest types:")
        print(df["test_type"].value_counts())
    else:
        print(f"Output: {args.output}")
        print(f"Max pages: {args.max_pages}")
        print(f"Delay: {args.delay}s")
        print("=" * 60)

        scraper = SHLCatalogScraper(delay_seconds=args.delay)
        assessments = scraper.scrape_all(max_pages=args.max_pages)

        if assessments:
            scraper.save_to_csv(args.output)
        else:
            print("\n⚠️  No assessments scraped. Creating sample catalog instead...")
            df = create_sample_catalog()
            df.to_csv(output_path, index=False)
            print(f"Saved {len(df)} sample assessments to {output_path}")

    print("\n[DONE] Scraping complete!")


if __name__ == "__main__":
    main()

