# SHL Assessment Recommendation Engine

A production-quality semantic search recommendation system for SHL assessments. Given a job description or natural language query, the system recommends the most relevant assessments from the SHL product catalog using RAG-style retrieval with sentence embeddings.

## 🎯 Project Overview

This project was built for the **SHL Research Intern, AI** assessment. It demonstrates:

- **Semantic Search**: Uses sentence-transformers for understanding natural language queries
- **RAG-style Retrieval**: Embeds both assessments and queries for similarity matching
- **K/P Type Balancing**: Automatically balances cognitive (K) and personality (P) assessments
- **AI-Powered Explanations**: Uses Google Gemini LLM to explain why assessments fit a job description
- **Production-Ready**: Clean architecture, tests, evaluation metrics, and documentation

## 📁 Project Structure

```
ASSESSMENT RECOMMENDATION ENGINE/
├── app/
│   └── streamlit_app.py         # Production Streamlit UI with LLM integration
├── src/
│   ├── data/
│   │   ├── catalog_loader.py    # Load SHL catalog with K/P types
│   │   └── dataset_loader.py    # Load Gen_AI-Dataset.xlsx
│   ├── embeddings/
│   │   └── embedding_manager.py # Sentence embedding with caching
│   ├── recommender/
│   │   ├── base_recommender.py  # Core recommendation engine
│   │   └── llm_explainer.py     # Gemini LLM for explanations
│   ├── evaluation/
│   │   └── metrics.py           # Recall@K, MAP, MRR metrics
│   └── scraper/
│       └── shl_scraper.py       # SHL catalog scraper
├── scripts/
│   ├── scrape_catalog.py        # Scrape SHL catalog
│   ├── augment_catalog.py       # Expand catalog with synthetic data
│   ├── evaluate.py              # Run evaluation on Train-Set
│   └── generate_submission.py   # Generate submission.csv
├── tests/
│   ├── test_scraper.py
│   ├── test_recommender.py
│   └── test_evaluation.py
├── data/
│   ├── Gen_AI-Dataset.xlsx      # Dataset (Train-Set, Test-Set)
│   └── shl_catalog.csv          # Assessment catalog (100 assessments)
├── outputs/
│   └── submission.csv           # Final predictions
├── docs/
│   └── tech_report.md           # Technical report
├── .env.example                 # Environment variables template
├── config.py                    # Central configuration
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys (Optional)

For AI-powered explanations, set your Gemini API key:

```bash
# Option 1: Create .env file
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# Option 2: Set environment variable directly (Windows)
set GEMINI_API_KEY=your_api_key_here

# Option 2: Set environment variable directly (Linux/Mac)
export GEMINI_API_KEY=your_api_key_here
```

### 3. Prepare Data

Place `Gen_AI-Dataset.xlsx` in the `data/` folder.

### 4. Create/Expand Assessment Catalog

```bash
# If scraper times out, augment the catalog with synthetic data
python scripts/augment_catalog.py --target 100

# Or create sample catalog
python scripts/scrape_catalog.py --sample
```

### 5. Run Evaluation

```bash
python scripts/evaluate.py

# Compare embedding models (ablation study)
python scripts/evaluate.py --compare
```

### 6. Generate Submission

```bash
python scripts/generate_submission.py --evaluate
```

### 7. Launch Web UI

```bash
streamlit run app.py
```

## 📊 Evaluation Metrics

The system is evaluated using:

| Metric | Description |
|--------|-------------|
| **Recall@K** | Fraction of queries where true assessment is in top-K (primary metric) |
| **MAP@K** | Mean Average Precision at K |
| **MRR** | Mean Reciprocal Rank |
| **Hit Rate** | Fraction of queries with at least one correct prediction |

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Embedding model (quality vs speed)
EMBEDDING_MODEL = "all-mpnet-base-v2"  # High quality
# EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Faster

# K/P Balance settings
KP_BALANCE_ENABLED = True
KP_BALANCE_RATIO = 0.5  # 50% K, 50% P when query spans both
```

## 🤖 AI-Powered Explanations

The system uses Google Gemini to generate natural language explanations for recommendations:

```python
from src.recommender import LLMExplainer

explainer = LLMExplainer()  # Reads GEMINI_API_KEY from environment
explanation = explainer.generate_explanation(
    query="Looking for a software engineer...",
    assessments=[{"name": "Python Test", "test_type": "K", ...}]
)
```

If the API key is not set, the system falls back to template-based explanations.

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_recommender.py -v
```

## 📝 API Usage

```python
from src.recommender import AssessmentRecommender

# Initialize
recommender = AssessmentRecommender(
    catalog_path="data/shl_catalog.csv",
    embedding_model="all-mpnet-base-v2"
)
recommender.load()

# Get recommendations
results = recommender.recommend(
    query="Software engineer with analytical skills",
    top_k=10
)

for r in results:
    print(f"{r.rank}. {r.assessment.name}")
    print(f"   Type: {r.assessment.test_type}")
    print(f"   Score: {r.score:.3f}")
```

## 📄 Submission Format

The `submission.csv` follows the required format:

| query | predictions |
|-------|-------------|
| "Job description..." | "url1 \| url2 \| url3 \| ..." |

## 🏗️ Architecture

```
┌─────────────────┐
│  Job Description│
│     (Query)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Query Embedding │ ◄── Sentence Transformers
│  (768-dim vec)  │     (all-mpnet-base-v2)
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ Cosine          │ ◄───│ Catalog         │
│ Similarity      │     │ Embeddings      │
└────────┬────────┘     │ (cached)        │
         │              └─────────────────┘
         ▼
┌─────────────────┐
│ K/P Balance     │ ◄── Ensure mix of cognitive
│ Reranking       │     and personality types
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ Top-K Results   │────►│ Gemini LLM      │
│                 │     │ (Explanations)  │
└─────────────────┘     └─────────────────┘
```

## 📚 Documentation

- **[Technical Report](docs/tech_report.md)**: Detailed system design, choices, and limitations
- **[Config Reference](config.py)**: All configurable parameters

## 🔮 Future Improvements

1. **Hybrid Search**: Combine semantic + keyword matching
2. **LLM Reranking**: Use GPT/Claude for final reranking
3. **Query Expansion**: Expand queries with synonyms
4. **Fine-tuning**: Fine-tune embeddings on SHL-specific data
5. **Multi-modal**: Support JD images/PDFs

## 📜 License

This project is for the SHL Research Intern, AI assessment.

---

Built with ❤️ using Python, Streamlit, Sentence Transformers, and Google Gemini

