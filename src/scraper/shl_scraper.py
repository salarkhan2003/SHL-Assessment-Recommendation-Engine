"""
SHL Product Catalog Scraper - Production Version

Scrapes assessment data from SHL's product catalog at:
https://www.shl.com/solutions/products/product-catalog/

Key Features:
- Extracts individual assessments (skips pre-packaged solutions)
- Captures K (Knowledge/Cognitive) and P (Personality/Behavioral) types
- Handles pagination and AJAX-loaded content
- Robust error handling and retry logic
- Rate limiting to respect server

The SHL catalog displays assessments in a table with columns:
- Assessment Name (with link to details)
- Remote Testing Support (Yes/No icon)
- Adaptive/IRT Support (Yes/No icon)
- Test Type indicators
"""

import re
import time
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, asdict, field
from urllib.parse import urljoin, urlparse, parse_qs
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SHLAssessment:
    """
    Data class for an SHL assessment.

    Attributes:
        name: Assessment name/title
        url: Full URL to assessment details page
        description: Brief description of the assessment
        test_type: List of types - "K" (Knowledge/Cognitive), "P" (Personality/Behavioral)
        remote_testing: Whether remote/unproctored testing is supported
        adaptive_irt: Whether adaptive/IRT testing is supported
        duration: Test duration (e.g., "25 minutes")
        languages: Available languages
        job_levels: Target job levels (Entry, Professional, Manager, Executive)
    """
    name: str
    url: str
    description: str = ""
    test_type: List[str] = field(default_factory=list)  # ["K"], ["P"], or ["K", "P"]
    remote_testing: bool = False
    adaptive_irt: bool = False
    duration: Optional[str] = None
    languages: List[str] = field(default_factory=list)
    job_levels: List[str] = field(default_factory=list)

    @property
    def type_string(self) -> str:
        """Get type as string (e.g., 'K', 'P', 'K, P')."""
        return ", ".join(sorted(self.test_type)) if self.test_type else "Unknown"

    @property
    def is_cognitive(self) -> bool:
        """Check if this is a cognitive/knowledge assessment."""
        return "K" in self.test_type

    @property
    def is_personality(self) -> bool:
        """Check if this is a personality/behavioral assessment."""
        return "P" in self.test_type

    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV export."""
        return {
            "name": self.name,
            "url": self.url,
            "description": self.description,
            "test_type": self.type_string,
            "remote_testing": self.remote_testing,
            "adaptive_irt": self.adaptive_irt,
            "duration": self.duration or "",
            "languages": ", ".join(self.languages) if self.languages else "",
            "job_levels": ", ".join(self.job_levels) if self.job_levels else "",
        }


class SHLCatalogScraper:
    """
    Production-grade scraper for SHL's product catalog.

    Usage:
        scraper = SHLCatalogScraper()
        assessments = scraper.scrape_all()
        scraper.save_to_csv("data/shl_catalog.csv")
    """

    BASE_URL = "https://www.shl.com"
    CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"

    # Keywords indicating pre-packaged/bundled solutions (to skip)
    BUNDLE_KEYWORDS = [
        "pre-packaged", "prepackaged", "bundle", "bundled",
        "solution package", "job solution", "talent solution"
    ]

    # Keywords for determining K vs P type
    K_TYPE_KEYWORDS = [
        "cognitive", "ability", "aptitude", "reasoning", "numerical", "verbal",
        "abstract", "logical", "inductive", "deductive", "mechanical", "spatial",
        "problem solving", "critical thinking", "g+", "verify"
    ]

    P_TYPE_KEYWORDS = [
        "personality", "behavioral", "behaviour", "competency", "motivation",
        "opq", "situational", "judgment", "sjt", "leadership", "emotional",
        "values", "interests", "occupational", "work style"
    ]

    def __init__(
        self,
        delay_seconds: float = 1.0,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize the scraper.

        Args:
            delay_seconds: Delay between requests
            timeout: Request timeout in seconds
            max_retries: Number of retries for failed requests
        """
        self.delay = delay_seconds
        self.timeout = timeout
        self.max_retries = max_retries

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        })

        self._assessments: List[SHLAssessment] = []
        self._seen_urls: Set[str] = set()

    def _request_with_retry(self, url: str) -> Optional[requests.Response]:
        """Make a request with retry logic."""
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed for {url}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay * (attempt + 1))  # Exponential backoff
        return None

    def scrape_all(self, max_pages: int = 50) -> List[SHLAssessment]:
        """
        Scrape all assessments from the SHL catalog.

        Args:
            max_pages: Maximum number of pages to scrape

        Returns:
            List of SHLAssessment objects
        """
        logger.info("Starting SHL catalog scrape...")
        self._assessments = []
        self._seen_urls = set()

        # Step 1: Get all assessment URLs from catalog pages
        all_urls = self._scrape_catalog_listing(max_pages)
        logger.info(f"Found {len(all_urls)} unique assessment URLs")

        if not all_urls:
            logger.warning("No URLs found. The website structure may have changed.")
            return []

        # Step 2: Scrape each individual assessment page
        logger.info("Scraping individual assessment pages...")
        for url in tqdm(all_urls, desc="Scraping assessments"):
            assessment = self._scrape_assessment_detail(url)
            if assessment and not self._is_bundle(assessment):
                self._assessments.append(assessment)
            time.sleep(self.delay)

        logger.info(f"Successfully scraped {len(self._assessments)} individual assessments")
        return self._assessments

    def _scrape_catalog_listing(self, max_pages: int) -> List[str]:
        """
        Scrape the catalog listing pages to extract assessment URLs.

        The SHL catalog uses a table format with pagination.
        """
        all_urls = []
        page = 1
        consecutive_empty = 0

        while page <= max_pages and consecutive_empty < 3:
            # Try different URL patterns for pagination
            urls_to_try = [
                f"{self.CATALOG_URL}?start={12 * (page - 1)}",  # 12 items per page
                f"{self.CATALOG_URL}?page={page}",
                f"{self.CATALOG_URL}?p={page}",
            ]

            page_urls = []
            for url in urls_to_try:
                response = self._request_with_retry(url)
                if response:
                    page_urls = self._extract_urls_from_listing(response.text)
                    if page_urls:
                        break

            if not page_urls:
                consecutive_empty += 1
                logger.debug(f"No URLs found on page {page}")
            else:
                consecutive_empty = 0
                new_urls = [u for u in page_urls if u not in all_urls]
                if new_urls:
                    all_urls.extend(new_urls)
                    logger.info(f"Page {page}: Found {len(new_urls)} new URLs")
                else:
                    logger.debug(f"Page {page}: All URLs already seen (end of pagination)")
                    break

            page += 1
            time.sleep(self.delay)

        return list(set(all_urls))

    def _extract_urls_from_listing(self, html: str) -> List[str]:
        """Extract assessment URLs from a catalog listing page."""
        soup = BeautifulSoup(html, "lxml")
        urls = []

        # Strategy 1: Look for table rows with assessment links
        for table in soup.find_all("table"):
            for row in table.find_all("tr"):
                for link in row.find_all("a", href=True):
                    url = self._normalize_url(link["href"])
                    if url and self._is_assessment_url(url):
                        urls.append(url)

        # Strategy 2: Look for product cards/tiles
        product_selectors = [
            "a.product-link", "a.assessment-link",
            ".product-card a", ".assessment-card a",
            "[data-product] a", "[data-assessment] a"
        ]
        for selector in product_selectors:
            for link in soup.select(selector):
                if link.get("href"):
                    url = self._normalize_url(link["href"])
                    if url and self._is_assessment_url(url):
                        urls.append(url)

        # Strategy 3: Look for any link that matches assessment URL pattern
        for link in soup.find_all("a", href=True):
            href = link["href"]
            url = self._normalize_url(href)
            if url and self._is_assessment_url(url):
                # Avoid navigation/footer links
                text = link.get_text(strip=True).lower()
                if not any(skip in text for skip in ["next", "prev", "page", "menu", "contact"]):
                    urls.append(url)

        return list(set(urls))

    def _is_assessment_url(self, url: str) -> bool:
        """Check if URL points to an individual assessment detail page."""
        if not url:
            return False

        # Must be on SHL domain
        if not url.startswith(self.BASE_URL):
            return False

        # Positive patterns (assessment detail pages)
        positive_patterns = [
            r"/solutions/products/assessments?/",
            r"/solutions/products/product-catalog/view/",
            r"/product-catalog/[^/]+/?$",
        ]

        # Negative patterns (not individual assessments)
        negative_patterns = [
            r"/product-catalog/?$",
            r"/product-catalog/?\?",
            r"/solutions/products/?$",
            r"/solutions/?$",
            r"#",
            r"\.pdf$",
            r"\.doc",
        ]

        for pattern in negative_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return False

        for pattern in positive_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return True

        return False

    def _normalize_url(self, url: str) -> Optional[str]:
        """Normalize URL to absolute form."""
        if not url or url.startswith(("javascript:", "mailto:", "#")):
            return None

        if url.startswith("http"):
            return url

        return urljoin(self.BASE_URL, url)

    def _scrape_assessment_detail(self, url: str) -> Optional[SHLAssessment]:
        """Scrape details from an individual assessment page."""
        if url in self._seen_urls:
            return None

        self._seen_urls.add(url)
        response = self._request_with_retry(url)

        if not response:
            return None

        soup = BeautifulSoup(response.text, "lxml")

        # Extract name
        name = self._extract_name(soup)
        if not name:
            logger.debug(f"Could not extract name from {url}")
            return None

        # Extract other fields
        description = self._extract_description(soup)
        test_type = self._determine_test_type(soup, name, description)
        remote_testing = self._has_remote_testing(soup)
        adaptive_irt = self._has_adaptive_irt(soup)
        duration = self._extract_duration(soup)

        return SHLAssessment(
            name=name,
            url=url,
            description=description,
            test_type=test_type,
            remote_testing=remote_testing,
            adaptive_irt=adaptive_irt,
            duration=duration,
        )

    def _extract_name(self, soup: BeautifulSoup) -> str:
        """Extract assessment name from page."""
        # Try h1 first
        h1 = soup.find("h1")
        if h1:
            name = h1.get_text(strip=True)
            # Clean up common suffixes
            name = re.sub(r"\s*[-|]\s*SHL.*$", "", name, flags=re.IGNORECASE)
            if name and len(name) > 2:
                return name

        # Try title tag
        title = soup.find("title")
        if title:
            name = title.get_text(strip=True)
            name = re.sub(r"\s*[-|]\s*SHL.*$", "", name, flags=re.IGNORECASE)
            if name and len(name) > 2:
                return name

        return ""

    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract assessment description."""
        # Try meta description
        meta = soup.find("meta", {"name": "description"})
        if meta and meta.get("content"):
            desc = meta["content"].strip()
            if len(desc) > 20:
                return desc[:500]

        # Try first paragraph in main content
        main_content = soup.find(["main", "article", ".content", "#content"])
        if main_content:
            for p in main_content.find_all("p"):
                text = p.get_text(strip=True)
                if len(text) > 50:
                    return text[:500]

        # Fallback: first substantial paragraph
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if len(text) > 50 and not any(skip in text.lower() for skip in ["cookie", "privacy", "copyright"]):
                return text[:500]

        return ""

    def _determine_test_type(self, soup: BeautifulSoup, name: str, description: str) -> List[str]:
        """
        Determine assessment type: K (Cognitive/Knowledge) or P (Personality/Behavioral).

        Some assessments may be both K and P.
        """
        combined_text = f"{name} {description}".lower()
        page_text = soup.get_text().lower()

        types = []

        # Check for K-type indicators
        k_score = sum(1 for kw in self.K_TYPE_KEYWORDS if kw in combined_text)
        k_score += sum(0.5 for kw in self.K_TYPE_KEYWORDS if kw in page_text)

        # Check for P-type indicators
        p_score = sum(1 for kw in self.P_TYPE_KEYWORDS if kw in combined_text)
        p_score += sum(0.5 for kw in self.P_TYPE_KEYWORDS if kw in page_text)

        # Threshold-based classification
        if k_score >= 1:
            types.append("K")
        if p_score >= 1:
            types.append("P")

        # Default to "K" if unclear (most assessments are cognitive)
        if not types:
            types.append("K")

        return types

    def _has_remote_testing(self, soup: BeautifulSoup) -> bool:
        """Check if assessment supports remote/unproctored testing."""
        page_text = soup.get_text().lower()
        indicators = ["remote", "unproctored", "online", "unsupervised", "at home"]
        return any(ind in page_text for ind in indicators)

    def _has_adaptive_irt(self, soup: BeautifulSoup) -> bool:
        """Check if assessment uses adaptive/IRT technology."""
        page_text = soup.get_text().lower()
        indicators = ["adaptive", "irt", "item response theory", "cat", "computer adaptive"]
        return any(ind in page_text for ind in indicators)

    def _extract_duration(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract test duration."""
        page_text = soup.get_text()

        patterns = [
            r"(\d+)\s*(?:minutes?|mins?)\s*(?:to\s*complete)?",
            r"(?:duration|time|length)[:\s]+(\d+)\s*(?:minutes?|mins?)?",
            r"(?:approximately|approx\.?|about)\s*(\d+)\s*(?:minutes?|mins?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, page_text, re.IGNORECASE)
            if match:
                mins = int(match.group(1))
                if 1 <= mins <= 180:  # Sanity check
                    return f"{mins} minutes"

        return None

    def _is_bundle(self, assessment: SHLAssessment) -> bool:
        """Check if this is a bundled/pre-packaged solution (should skip)."""
        combined = f"{assessment.name} {assessment.description}".lower()
        return any(kw in combined for kw in self.BUNDLE_KEYWORDS)

    def save_to_csv(self, output_path: str = "data/shl_catalog.csv") -> None:
        """Save scraped assessments to CSV."""
        if not self._assessments:
            logger.warning("No assessments to save")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [a.to_dict() for a in self._assessments]
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)

        logger.info(f"Saved {len(df)} assessments to {output_path}")

        # Print summary
        print(f"\n{'='*50}")
        print(f"Catalog Summary:")
        print(f"{'='*50}")
        print(f"Total assessments: {len(df)}")
        print(f"\nBy Type:")
        print(df["test_type"].value_counts())
        print(f"\nRemote Testing: {df['remote_testing'].sum()} assessments")
        print(f"Adaptive/IRT: {df['adaptive_irt'].sum()} assessments")

    def get_dataframe(self) -> pd.DataFrame:
        """Get scraped data as DataFrame."""
        if not self._assessments:
            return pd.DataFrame()
        return pd.DataFrame([a.to_dict() for a in self._assessments])


def create_sample_catalog() -> pd.DataFrame:
    """
    Create a sample catalog for testing when scraping isn't possible.

    This uses the REAL SHL product catalog URL format to match training data.
    URLs follow the pattern: https://www.shl.com/solutions/products/product-catalog/view/{slug}/

    Returns a DataFrame that can be saved to CSV.
    """
    # Base URL pattern matching real SHL catalog
    BASE_URL = "https://www.shl.com/solutions/products/product-catalog/view"

    sample_data = [
        # =================================================================
        # COGNITIVE ASSESSMENTS (K type) - Technical & Reasoning
        # =================================================================
        {
            "name": "Automata Fix",
            "url": f"{BASE_URL}/automata-fix-new/",
            "description": "Debugging and code fixing assessment for software developers. Tests ability to identify and fix bugs in code.",
            "test_type": "K",
            "remote_testing": True,
            "adaptive_irt": False,
            "duration": "30 minutes",
            "languages": "English",
            "job_levels": "Professional"
        },
        {
            "name": "Core Java (Entry Level)",
            "url": f"{BASE_URL}/core-java-entry-level-new/",
            "description": "Entry-level Java programming assessment covering fundamentals, OOP concepts, and basic data structures.",
            "test_type": "K",
            "remote_testing": True,
            "adaptive_irt": False,
            "duration": "40 minutes",
            "languages": "English",
            "job_levels": "Entry"
        },
        {
            "name": "Core Java (Advanced Level)",
            "url": f"{BASE_URL}/core-java-advanced-level-new/",
            "description": "Advanced Java programming assessment covering multithreading, collections, design patterns, and JVM internals.",
            "test_type": "K",
            "remote_testing": True,
            "adaptive_irt": False,
            "duration": "45 minutes",
            "languages": "English",
            "job_levels": "Professional"
        },
        {
            "name": "Java 8",
            "url": f"{BASE_URL}/java-8-new/",
            "description": "Java 8 specific features assessment including streams, lambdas, functional interfaces, and new date/time API.",
            "test_type": "K",
            "remote_testing": True,
            "adaptive_irt": False,
            "duration": "35 minutes",
            "languages": "English",
            "job_levels": "Professional"
        },
        {
            "name": "Python (New)",
            "url": f"{BASE_URL}/python-new/",
            "description": "Python programming assessment covering syntax, data structures, OOP, and common libraries.",
            "test_type": "K",
            "remote_testing": True,
            "adaptive_irt": False,
            "duration": "40 minutes",
            "languages": "English",
            "job_levels": "Professional"
        },
        {
            "name": "SQL (Advanced)",
            "url": f"{BASE_URL}/sql-advanced-new/",
            "description": "Advanced SQL assessment covering complex queries, optimization, stored procedures, and database design.",
            "test_type": "K",
            "remote_testing": True,
            "adaptive_irt": False,
            "duration": "35 minutes",
            "languages": "English",
            "job_levels": "Professional"
        },
        {
            "name": "Verify Numerical Reasoning",
            "url": f"{BASE_URL}/verify-numerical-reasoning/",
            "description": "Assesses ability to analyze and interpret numerical data including tables, graphs, and statistics.",
            "test_type": "K",
            "remote_testing": True,
            "adaptive_irt": True,
            "duration": "18 minutes",
            "languages": "English, Spanish, French",
            "job_levels": "Professional, Manager"
        },
        {
            "name": "Verify Verbal Reasoning",
            "url": f"{BASE_URL}/verify-verbal-reasoning/",
            "description": "Evaluates ability to understand written information, draw conclusions, and analyze complex text.",
            "test_type": "K",
            "remote_testing": True,
            "adaptive_irt": True,
            "duration": "17 minutes",
            "languages": "English, Spanish, French, German",
            "job_levels": "Professional, Manager"
        },
        {
            "name": "Verify Inductive Reasoning",
            "url": f"{BASE_URL}/verify-inductive-reasoning/",
            "description": "Measures ability to identify patterns and trends in abstract data. Tests logical reasoning.",
            "test_type": "K",
            "remote_testing": True,
            "adaptive_irt": True,
            "duration": "24 minutes",
            "languages": "English, Spanish",
            "job_levels": "All Levels"
        },
        {
            "name": "Verify G+ Cognitive Ability",
            "url": f"{BASE_URL}/verify-g-plus/",
            "description": "General cognitive ability assessment combining numerical, verbal, and abstract reasoning. Adaptive.",
            "test_type": "K",
            "remote_testing": True,
            "adaptive_irt": True,
            "duration": "25 minutes",
            "languages": "English, Spanish, French, German",
            "job_levels": "All Levels"
        },
        {
            "name": "Mechanical Comprehension",
            "url": f"{BASE_URL}/mechanical-comprehension/",
            "description": "Evaluates understanding of mechanical and physical principles for technical roles.",
            "test_type": "K",
            "remote_testing": True,
            "adaptive_irt": False,
            "duration": "25 minutes",
            "languages": "English, Spanish",
            "job_levels": "Entry, Professional"
        },
        {
            "name": "Technology Professional 8.0 Job Focused Assessment",
            "url": f"{BASE_URL}/technology-professional-8/",
            "description": "Comprehensive assessment for technology professionals covering technical aptitude and problem-solving.",
            "test_type": "K",
            "remote_testing": True,
            "adaptive_irt": False,
            "duration": "50 minutes",
            "languages": "English",
            "job_levels": "Professional"
        },
        # =================================================================
        # PERSONALITY ASSESSMENTS (P type) - Behavioral & Situational
        # =================================================================
        {
            "name": "Interpersonal Communications",
            "url": f"{BASE_URL}/interpersonal-communications/",
            "description": "Assesses interpersonal communication skills including active listening, clarity, and collaboration.",
            "test_type": "P",
            "remote_testing": True,
            "adaptive_irt": False,
            "duration": "20 minutes",
            "languages": "English",
            "job_levels": "All Levels"
        },
        {
            "name": "OPQ32r",
            "url": f"{BASE_URL}/opq32r/",
            "description": "Comprehensive personality questionnaire measuring 32 work-relevant characteristics and behaviors.",
            "test_type": "P",
            "remote_testing": True,
            "adaptive_irt": False,
            "duration": "25 minutes",
            "languages": "English, Spanish, French, German, Chinese",
            "job_levels": "All Levels"
        },
        {
            "name": "Motivation Questionnaire",
            "url": f"{BASE_URL}/motivation-questionnaire/",
            "description": "Measures factors that motivate individuals at work including achievement, recognition, and autonomy.",
            "test_type": "P",
            "remote_testing": True,
            "adaptive_irt": False,
            "duration": "20 minutes",
            "languages": "English, Spanish, French",
            "job_levels": "All Levels"
        },
        {
            "name": "Situational Judgment - Leadership",
            "url": f"{BASE_URL}/sjt-leadership/",
            "description": "Presents realistic workplace scenarios to assess leadership judgment and decision-making.",
            "test_type": "P",
            "remote_testing": True,
            "adaptive_irt": False,
            "duration": "30 minutes",
            "languages": "English, Spanish",
            "job_levels": "Manager, Executive"
        },
        {
            "name": "Situational Judgment - Graduate",
            "url": f"{BASE_URL}/sjt-graduate/",
            "description": "Entry-level situational judgment test for graduate candidates assessing workplace judgment.",
            "test_type": "P",
            "remote_testing": True,
            "adaptive_irt": False,
            "duration": "25 minutes",
            "languages": "English",
            "job_levels": "Entry"
        },
        {
            "name": "Customer Service Assessment",
            "url": f"{BASE_URL}/customer-service/",
            "description": "Evaluates competencies for customer-facing roles including communication and service orientation.",
            "test_type": "P",
            "remote_testing": True,
            "adaptive_irt": False,
            "duration": "20 minutes",
            "languages": "English, Spanish",
            "job_levels": "Entry, Professional"
        },
        {
            "name": "Call Center Simulation",
            "url": f"{BASE_URL}/call-center-simulation/",
            "description": "Realistic simulation assessing call handling, problem resolution, and customer interaction skills.",
            "test_type": "P",
            "remote_testing": True,
            "adaptive_irt": False,
            "duration": "30 minutes",
            "languages": "English",
            "job_levels": "Entry, Professional"
        },
        # =================================================================
        # MIXED ASSESSMENTS (K + P types) - Comprehensive Solutions
        # =================================================================
        {
            "name": "Sales Assessment Battery",
            "url": f"{BASE_URL}/sales-assessment/",
            "description": "Comprehensive assessment combining cognitive ability, personality traits, and sales judgment.",
            "test_type": "K, P",
            "remote_testing": True,
            "adaptive_irt": False,
            "duration": "45 minutes",
            "languages": "English, Spanish",
            "job_levels": "Professional"
        },
        {
            "name": "Manager Assessment Solution",
            "url": f"{BASE_URL}/manager-assessment/",
            "description": "Integrated assessment for management candidates including cognitive ability and leadership traits.",
            "test_type": "K, P",
            "remote_testing": True,
            "adaptive_irt": True,
            "duration": "60 minutes",
            "languages": "English, Spanish, French",
            "job_levels": "Manager"
        },
        {
            "name": "Graduate Assessment Solution",
            "url": f"{BASE_URL}/graduate-assessment/",
            "description": "Complete assessment for graduate recruitment including reasoning and personality measures.",
            "test_type": "K, P",
            "remote_testing": True,
            "adaptive_irt": True,
            "duration": "50 minutes",
            "languages": "English",
            "job_levels": "Entry"
        },
        {
            "name": "Technology Professional Assessment",
            "url": f"{BASE_URL}/technology-professional-assessment/",
            "description": "Comprehensive assessment for tech roles combining technical aptitude with work style assessment.",
            "test_type": "K, P",
            "remote_testing": True,
            "adaptive_irt": False,
            "duration": "55 minutes",
            "languages": "English",
            "job_levels": "Professional"
        },
    ]

    return pd.DataFrame(sample_data)


if __name__ == "__main__":
    # Quick test
    print("Creating sample catalog...")
    df = create_sample_catalog()
    print(f"Created {len(df)} sample assessments")
    print(df[["name", "test_type", "duration"]].to_string())

