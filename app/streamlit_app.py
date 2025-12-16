"""
SHL Assessment Recommendation Engine - Streamlit UI

Production-quality web interface for the recommendation engine.

Features:
- Clean, spacious layout with full-width results
- Results table with Name, Type (K/P), URL, and Relevance score
- AI-powered "Why these assessments?" explanations using Gemini LLM
- Export functionality

Usage:
    streamlit run app/streamlit_app.py
    OR
    streamlit run app.py (from project root)
    
Environment Variables:
    GEMINI_API_KEY - Optional. Enables AI-powered explanations.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env if available
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars

import streamlit as st
import pandas as pd

from src.recommender import AssessmentRecommender, LLMExplainer


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling with better spacing
st.markdown("""
<style>
    /* Main container - more padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Main header styling */
    .main-header {
        background: linear-gradient(90deg, #0066cc 0%, #004499 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 1.8rem;
    }
    .main-header p {
        color: #e0e0e0;
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
    }
    
    /* Assessment card styling */
    .assessment-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: all 0.2s ease;
    }
    .assessment-card:hover {
        border-color: #0066cc;
        box-shadow: 0 4px 12px rgba(0,102,204,0.15);
    }
    
    /* Type badges */
    .badge-k {
        background: #e3f2fd;
        color: #1565c0;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 5px;
    }
    .badge-p {
        background: #f3e5f5;
        color: #7b1fa2;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 5px;
    }
    .badge-kp {
        background: #fff3e0;
        color: #e65100;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 5px;
    }
    
    /* Score styling */
    .score-badge {
        background: #e8f5e9;
        color: #2e7d32;
        padding: 4px 10px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    
    /* Metrics row */
    .metrics-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1.5rem;
    }
    
    /* AI explanation box */
    .ai-explanation {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.25rem;
        border-radius: 10px;
        margin: 1.5rem 0;
    }
    .ai-explanation h4 {
        color: white;
        margin: 0 0 0.5rem 0;
    }
    
    /* Results section header */
    .results-header {
        background: #f0f4f8;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid #0066cc;
    }
    
    /* Table styling */
    .dataframe {
        font-size: 0.95rem !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #666;
        padding: 2rem;
        border-top: 1px solid #e0e0e0;
        margin-top: 3rem;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CACHED RESOURCE LOADING
# =============================================================================

@st.cache_resource
def load_recommender():
    """Load and cache the recommender model."""
    recommender = AssessmentRecommender(
        catalog_path=str(project_root / "data" / "shl_catalog.csv"),
        embedding_model="all-mpnet-base-v2",
        cache_dir=str(project_root / "data" / "embeddings_cache"),
        enable_kp_balance=True
    )
    recommender.load()
    return recommender


@st.cache_resource
def load_llm_explainer():
    """Load and cache the LLM explainer."""
    return LLMExplainer()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_type_badge(test_type: str) -> str:
    """Get badge class for assessment type."""
    if "K" in test_type and "P" in test_type:
        return "badge-kp"
    elif "K" in test_type:
        return "badge-k"
    else:
        return "badge-p"


def format_type_label(test_type: str) -> str:
    """Format type for display."""
    if "K" in test_type and "P" in test_type:
        return "Cognitive + Personality"
    elif "K" in test_type:
        return "Cognitive"
    else:
        return "Personality"


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 SHL Assessment Recommendation Engine</h1>
        <p>Find the right assessments for your job requirements using AI-powered semantic search</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load recommender
    try:
        with st.spinner("Loading recommendation engine..."):
            recommender = load_recommender()
    except FileNotFoundError:
        st.error("""
        **Catalog not found!**
        
        Please run the setup first:
        ```bash
        python scripts/scrape_catalog.py --sample
        ```
        """)
        return
    except Exception as e:
        st.error(f"Error loading recommender: {e}")
        return
    
    # Load LLM explainer
    explainer = load_llm_explainer()
    
    # Sidebar settings
    with st.sidebar:
        st.header("📊 System Info")
        stats = recommender.catalog_stats
        st.metric("Total Assessments", stats["total"])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Cognitive (K)", stats["cognitive_only"] + stats["both"])
        with col2:
            st.metric("Personality (P)", stats["personality_only"] + stats["both"])
        
        st.markdown("---")
        
        # LLM Status
        st.header("🤖 AI Explainer")
        if explainer.is_available:
            st.success("✅ Gemini Connected")
        else:
            st.warning("⚠️ Not configured")
            st.caption("Set GEMINI_API_KEY to enable")
        
        st.markdown("---")
        st.header("⚙️ Settings")
        top_k = st.slider("Number of results", 1, 20, 10)
        min_score = st.slider("Min relevance score", 0.0, 0.5, 0.0, 0.05)
        show_explanations = st.checkbox("Show explanations", value=True)
        use_llm = st.checkbox("Use AI explanations", value=explainer.is_available, disabled=not explainer.is_available)

    # ==========================================================================
    # SEARCH SECTION (Full Width)
    # ==========================================================================
    
    st.subheader("📝 Enter Job Description or Query")
    
    # Example queries dropdown
    example_queries = {
        "Select an example...": "",
        "1. AI Research Intern": "AI Research Intern working on NLP, speech processing, computer vision, Python, TensorFlow, PyTorch, and Generative AI models like LLMs and RAG",
        "2. Graduate Software Engineer": "Graduate Software Engineer proficient in Python, Java, SQL, with strong logical reasoning and team collaboration skills",
        "3. Sales Manager": "Sales Manager who leads teams, influences external stakeholders, handles customer negotiations, and drives revenue growth",
        "4. Data Analyst": "Data Analyst requiring advanced Excel, SQL, numerical reasoning, attention to detail, and stakeholder communication",
        "5. HR Business Partner": "HR Business Partner focused on talent acquisition, employee relations, stakeholder management, and people leadership",
        "6. Presales Specialist": "Presales Specialist building client demos, responding to RFPs, presenting solutions, and scoping technical requirements",
        "7. Contact Center Agent": "Contact Center Agent handling high-volume customer queries with empathy, problem-solving, and communication skills",
        "8. Financial Analyst": "Financial Analyst with strong numerical reasoning, Excel modeling, financial statement analysis, and attention to detail",
        "9. UX Designer": "UX Designer requiring creativity, user research skills, prototyping, collaboration with developers, and design thinking",
        "10. Project Manager": "Project Manager with leadership, stakeholder management, planning, risk assessment, and team coordination skills",
        "11. Content Writer": "Content Writer expert in English grammar, SEO optimization, creative writing, and deadline management",
        "12. Machine Learning Engineer": "Machine Learning Engineer with Python, ML frameworks, data processing, model deployment, and problem-solving",
        "13. Customer Success Manager": "Customer Success Manager focused on client relationship building, retention, upselling, and satisfaction metrics",
        "14. Operations Manager": "Operations Manager requiring process optimization, strategic thinking, team leadership, and performance analysis",
        "15. Marketing Manager": "Marketing Manager with campaign strategy, data analysis, creativity, stakeholder influence, and ROI measurement",
    }
    
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_example = st.selectbox("Load example:", list(example_queries.keys()))
    
    query = st.text_area(
        "Enter job description or requirements:",
        value=example_queries.get(selected_example, ""),
        height=150,
        placeholder="Paste your job description here... Include required skills, responsibilities, and desired competencies."
    )
    
    search_btn = st.button("🔍 Find Matching Assessments", type="primary", use_container_width=True)
    
    # ==========================================================================
    # RESULTS SECTION (Full Width with proper spacing)
    # ==========================================================================
    
    if search_btn and query.strip():
        st.markdown("---")
        
        with st.spinner("Finding best matching assessments..."):
            results = recommender.recommend(
                query=query.strip(),
                top_k=top_k,
                min_score=min_score,
                generate_explanations=show_explanations
            )
        
        if not results:
            st.warning("No assessments found matching your criteria. Try lowering the minimum score.")
        else:
            # Results header
            st.markdown(f"""
            <div class="results-header">
                <h3 style="margin:0;">🎯 Found {len(results)} Matching Assessments</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Summary metrics in columns
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("📊 Total Results", len(results))
            with metric_cols[1]:
                k_count = sum(1 for r in results if r.assessment.is_cognitive)
                st.metric("🧠 Cognitive (K)", k_count)
            with metric_cols[2]:
                p_count = sum(1 for r in results if r.assessment.is_personality)
                st.metric("👤 Personality (P)", p_count)
            with metric_cols[3]:
                avg_score = sum(r.score for r in results) / len(results)
                st.metric("⭐ Avg Relevance", f"{avg_score:.1%}")
            
            # AI-powered explanation section
            if show_explanations and use_llm:
                st.markdown("####")
                with st.spinner("🤖 Generating AI explanation..."):
                    assessment_data = [
                        {
                            "name": r.assessment.name,
                            "test_type": r.assessment.test_type,
                            "description": r.assessment.description or ""
                        }
                        for r in results[:5]
                    ]
                    ai_explanation = explainer.generate_explanation(
                        query=query.strip(),
                        assessments=assessment_data
                    )
                
                st.markdown(f"""
                <div class="ai-explanation">
                    <h4>🤖 AI Analysis - Why These Assessments?</h4>
                    <p>{ai_explanation}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("####")
            
            # Results Table (cleaner view)
            st.markdown("### 📋 Assessment Results")
            
            table_data = []
            for r in results:
                table_data.append({
                    "#": r.rank,
                    "Assessment Name": r.assessment.name,
                    "Type": r.assessment.test_type,
                    "Duration": r.assessment.duration or "N/A",
                    "Relevance": f"{r.score:.1%}",
                    "Remote": "✅" if r.assessment.remote_testing else "❌",
                    "Adaptive": "✅" if r.assessment.adaptive_irt else "❌",
                })
            
            df = pd.DataFrame(table_data)
            
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                height=min(400, 50 + len(results) * 35)  # Dynamic height
            )
            
            st.markdown("####")
            
            # Detailed Assessment Cards
            st.markdown("### 📝 Detailed Assessment Information")
            
            # Display in 2-column grid for better use of space
            for i in range(0, len(results), 2):
                cols = st.columns(2)
                
                for j, col in enumerate(cols):
                    if i + j < len(results):
                        r = results[i + j]
                        with col:
                            with st.container():
                                # Card header
                                type_badge = "🧠" if r.assessment.is_cognitive else "👤"
                                if r.assessment.is_cognitive and r.assessment.is_personality:
                                    type_badge = "🧠👤"
                                
                                st.markdown(f"**{r.rank}. {r.assessment.name}** {type_badge}")
                                
                                # Info row
                                info_cols = st.columns(3)
                                with info_cols[0]:
                                    st.caption(f"**Type:** {r.assessment.test_type}")
                                with info_cols[1]:
                                    st.caption(f"**Score:** {r.score:.1%}")
                                with info_cols[2]:
                                    st.caption(f"**Duration:** {r.assessment.duration or 'N/A'}")
                                
                                # Description
                                st.markdown(f"_{r.assessment.description or 'No description available.'}_")
                                
                                # Features
                                features = []
                                if r.assessment.remote_testing:
                                    features.append("✅ Remote")
                                if r.assessment.adaptive_irt:
                                    features.append("🔄 Adaptive")
                                if r.assessment.duration:
                                    features.append(f"⏱️ {r.assessment.duration}")

                                if features:
                                    st.caption(" | ".join(features))
                                
                                # Link
                                st.markdown(f"[🔗 View Details]({r.assessment.url})")
                                
                                st.markdown("---")
            
            # Export section
            st.markdown("### 📥 Export Results")
            
            export_cols = st.columns([2, 1, 1])
            with export_cols[0]:
                # Full export with URLs
                full_export = []
                for r in results:
                    full_export.append({
                        "Rank": r.rank,
                        "Assessment": r.assessment.name,
                        "Type": r.assessment.test_type,
                        "Relevance": f"{r.score:.3f}",
                        "Duration": r.assessment.duration or "",
                        "Remote Testing": r.assessment.remote_testing,
                        "URL": r.assessment.url,
                        "Description": r.assessment.description or ""
                    })
                export_df = pd.DataFrame(full_export)
                csv = export_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Full Results (CSV)",
                    data=csv,
                    file_name="shl_recommendations.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    elif search_btn:
        st.warning("⚠️ Please enter a job description or query to search.")
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p><strong>SHL Assessment Recommendation Engine</strong> v1.0</p>
        <p>Powered by Sentence Transformers, Semantic Search & Google Gemini</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

