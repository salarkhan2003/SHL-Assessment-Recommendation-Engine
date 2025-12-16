"""
LLM Explainer Module - Gemini Integration

Generates natural language explanations for why recommended assessments
fit a given job description using Google's Gemini API.

Configuration:
    Set environment variable GEMINI_API_KEY with your API key.
    The key is NEVER hardcoded or logged.

Usage:
    from src.recommender.llm_explainer import LLMExplainer

    explainer = LLMExplainer()
    explanation = explainer.generate_explanation(
        query="Software engineer with Python skills",
        assessments=[{"name": "Python Test", "type": "K", ...}]
    )
"""

import os
import json
import logging
from typing import List, Dict, Optional
import requests

logger = logging.getLogger(__name__)

# Gemini API endpoint
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Default timeout for API calls
DEFAULT_TIMEOUT = 30


class LLMExplainer:
    """
    LLM-powered explanation generator using Google Gemini.

    Generates natural language explanations for why recommended
    assessments match a given job description or query.

    The API key is read from the GEMINI_API_KEY environment variable
    and is never logged or stored in code.
    """

    def __init__(self, api_key: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT):
        """
        Initialize the LLM explainer.

        Args:
            api_key: Optional API key. If not provided, reads from GEMINI_API_KEY env var.
            timeout: Request timeout in seconds.
        """
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.timeout = timeout
        self._is_available = self._api_key is not None and len(self._api_key) > 0

        if not self._is_available:
            logger.warning(
                "GEMINI_API_KEY not set. LLM explanations will use fallback templates. "
                "Set the environment variable to enable AI-powered explanations."
            )

    @property
    def is_available(self) -> bool:
        """Check if the LLM service is available (API key is set)."""
        return self._is_available

    def generate_explanation(
        self,
        query: str,
        assessments: List[Dict],
        max_length: int = 300
    ) -> str:
        """
        Generate an explanation for why the assessments fit the query.

        Args:
            query: The job description or user query
            assessments: List of assessment dicts with 'name', 'test_type', 'description'
            max_length: Maximum length of explanation

        Returns:
            Natural language explanation string
        """
        if not assessments:
            return "No assessments selected."

        # Try LLM first, fall back to template
        if self._is_available:
            try:
                return self._call_gemini(query, assessments, max_length)
            except Exception as e:
                logger.warning(f"Gemini API call failed: {e}. Using fallback.")
                return self._generate_fallback(query, assessments)
        else:
            return self._generate_fallback(query, assessments)

    def _call_gemini(
        self,
        query: str,
        assessments: List[Dict],
        max_length: int
    ) -> str:
        """
        Call the Gemini API to generate an explanation.

        Args:
            query: The job description or user query
            assessments: List of assessment dicts
            max_length: Maximum length of explanation

        Returns:
            Generated explanation from Gemini

        Raises:
            Exception: If API call fails
        """
        # Format assessments for the prompt
        assessment_text = "\n".join([
            f"- {a.get('name', 'Unknown')} ({a.get('test_type', 'Unknown')} type): {a.get('description', 'No description')[:100]}"
            for a in assessments[:5]  # Limit to top 5 for prompt size
        ])

        # Craft the prompt
        prompt = f"""You are an expert HR consultant helping explain assessment recommendations.

Given this job requirement or query:
"{query[:500]}"

And these recommended SHL assessments:
{assessment_text}

Write a brief, professional explanation (2-3 sentences, max {max_length} characters) of why these assessments are relevant for evaluating candidates for this role. Focus on the connection between the job requirements and what each assessment type measures.

Be concise and helpful. Do not use bullet points - write flowing prose."""

        # Prepare request
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self._api_key
        }

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 256,
                "temperature": 0.7
            }
        }

        # Make API call
        response = requests.post(
            GEMINI_API_URL,
            headers=headers,
            json=payload,
            timeout=self.timeout
        )

        if response.status_code != 200:
            raise Exception(f"Gemini API error: {response.status_code} - {response.text[:200]}")

        # Parse response
        result = response.json()

        try:
            explanation = result["candidates"][0]["content"]["parts"][0]["text"]
            # Clean up and truncate if needed
            explanation = explanation.strip()
            if len(explanation) > max_length:
                explanation = explanation[:max_length-3] + "..."
            return explanation
        except (KeyError, IndexError) as e:
            raise Exception(f"Unexpected Gemini response format: {e}")

    def _generate_fallback(self, query: str, assessments: List[Dict]) -> str:
        """
        Generate a template-based explanation when LLM is unavailable.

        Args:
            query: The job description or user query
            assessments: List of assessment dicts

        Returns:
            Template-based explanation string
        """
        if not assessments:
            return "No assessments to explain."

        # Count types
        k_count = sum(1 for a in assessments if "K" in a.get("test_type", ""))
        p_count = sum(1 for a in assessments if "P" in a.get("test_type", ""))

        # Build explanation
        parts = []

        if k_count > 0 and p_count > 0:
            parts.append(
                f"These recommendations include {k_count} cognitive/skills assessments and "
                f"{p_count} personality/behavioral assessments to provide a comprehensive "
                "evaluation of candidates."
            )
        elif k_count > 0:
            parts.append(
                f"These {k_count} cognitive and skills-based assessments will help evaluate "
                "technical abilities, reasoning skills, and job-relevant knowledge."
            )
        elif p_count > 0:
            parts.append(
                f"These {p_count} personality and behavioral assessments will help understand "
                "work style, interpersonal skills, and cultural fit."
            )

        # Add specific mentions
        names = [a.get("name", "Unknown") for a in assessments[:3]]
        if names:
            parts.append(
                f"Key assessments include {', '.join(names)}, which align with the "
                "requirements described in the job description."
            )

        return " ".join(parts)

    def explain_single_assessment(
        self,
        query: str,
        assessment: Dict
    ) -> str:
        """
        Generate a brief explanation for a single assessment.

        Args:
            query: The job description or user query
            assessment: Single assessment dict

        Returns:
            Brief explanation string
        """
        name = assessment.get("name", "This assessment")
        test_type = assessment.get("test_type", "")
        description = assessment.get("description", "")

        if "K" in test_type:
            type_desc = "cognitive and skills-based"
        elif "P" in test_type:
            type_desc = "personality and behavioral"
        else:
            type_desc = "comprehensive"

        return f"{name} is a {type_desc} assessment that {description[:150].lower()}"


# Singleton instance for convenience
_default_explainer: Optional[LLMExplainer] = None


def get_explainer() -> LLMExplainer:
    """Get the default LLM explainer instance."""
    global _default_explainer
    if _default_explainer is None:
        _default_explainer = LLMExplainer()
    return _default_explainer


def generate_explanation(query: str, assessments: List[Dict]) -> str:
    """
    Convenience function to generate an explanation.

    Args:
        query: The job description or user query
        assessments: List of assessment dicts

    Returns:
        Explanation string
    """
    return get_explainer().generate_explanation(query, assessments)

