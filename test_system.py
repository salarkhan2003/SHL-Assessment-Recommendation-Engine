"""Quick test to verify the system works with the new catalog."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

# Check catalog
df = pd.read_csv("data/shl_catalog.csv")
print(f"Catalog size: {len(df)} assessments")
print(f"\nType distribution:")
print(df['test_type'].value_counts())

# Test recommender
print("\n" + "="*50)
print("Testing Recommender...")
print("="*50)

from src.recommender import AssessmentRecommender

recommender = AssessmentRecommender(
    catalog_path="data/shl_catalog.csv",
    embedding_model="all-mpnet-base-v2",
    cache_dir="data/embeddings_cache"
)
recommender.load()

print(f"\nLoaded {recommender.catalog_size} assessments")

# Test query
query = "Looking for a software engineer with Python and Java skills who can collaborate well"
results = recommender.recommend(query, top_k=5)

print(f"\nQuery: {query[:60]}...")
print("\nTop 5 Recommendations:")
for r in results:
    print(f"  {r.rank}. {r.assessment.name} ({r.assessment.test_type}) - {r.score:.3f}")

# Test LLM explainer
print("\n" + "="*50)
print("Testing LLM Explainer...")
print("="*50)

from src.recommender import LLMExplainer

explainer = LLMExplainer()
print(f"LLM Available: {explainer.is_available}")

assessment_data = [
    {"name": r.assessment.name, "test_type": r.assessment.test_type, "description": r.assessment.description}
    for r in results[:3]
]

explanation = explainer.generate_explanation(query, assessment_data)
print(f"\nExplanation:\n{explanation}")

print("\n" + "="*50)
print("All tests passed!")
print("="*50)

