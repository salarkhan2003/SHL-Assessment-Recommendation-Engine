"""
Catalog Augmentation Script for SHL Assessment Recommendation Engine

When the live SHL scraper times out, use this script to expand the catalog
with synthetic but realistic assessments for experimentation.

Usage:
    python scripts/augment_catalog.py
    python scripts/augment_catalog.py --target 150  # Expand to 150 assessments

This script:
- Loads existing data/shl_catalog.csv
- Adds realistic synthetic assessments
- Maintains proper K/P type distribution (60-70% K, 30-40% P)
- Preserves the exact schema and URL patterns
"""

import sys
import argparse
import random
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd


# =============================================================================
# ASSESSMENT TEMPLATES FOR REALISTIC GENERATION
# =============================================================================

# Cognitive/Knowledge (K) assessments
K_ASSESSMENTS = [
    # Programming & Technical
    ("JavaScript (Modern ES6+)", "Modern JavaScript assessment covering ES6+ features, async/await, promises, modules, and DOM manipulation.", "40 minutes", "Professional"),
    ("React.js Development", "React.js framework assessment covering components, hooks, state management, and best practices.", "45 minutes", "Professional"),
    ("Angular Development", "Angular framework assessment including TypeScript, components, services, and RxJS.", "45 minutes", "Professional"),
    ("Node.js Backend", "Node.js backend development assessment covering Express, APIs, middleware, and async patterns.", "40 minutes", "Professional"),
    ("Full Stack Development", "Comprehensive full-stack assessment covering frontend, backend, databases, and deployment.", "60 minutes", "Professional"),
    ("C++ Programming", "C++ programming assessment covering memory management, OOP, STL, and modern C++ features.", "45 minutes", "Professional"),
    ("C# .NET Development", "C# and .NET framework assessment covering language features, LINQ, async programming, and patterns.", "40 minutes", "Professional"),
    ("Go Programming", "Go programming assessment covering goroutines, channels, interfaces, and idiomatic Go patterns.", "35 minutes", "Professional"),
    ("Rust Programming", "Rust programming assessment covering ownership, borrowing, lifetimes, and safe concurrency.", "45 minutes", "Professional"),
    ("TypeScript Advanced", "Advanced TypeScript assessment covering type system, generics, decorators, and utility types.", "35 minutes", "Professional"),
    ("Data Structures & Algorithms", "Fundamental assessment covering arrays, trees, graphs, sorting, searching, and complexity analysis.", "50 minutes", "Professional"),
    ("System Design", "System design assessment covering scalability, distributed systems, and architecture patterns.", "60 minutes", "Manager"),
    ("API Design & REST", "RESTful API design assessment covering endpoints, authentication, versioning, and best practices.", "30 minutes", "Professional"),
    ("GraphQL Development", "GraphQL assessment covering schemas, resolvers, queries, mutations, and subscriptions.", "35 minutes", "Professional"),
    ("Microservices Architecture", "Microservices assessment covering service design, communication patterns, and containerization.", "45 minutes", "Manager"),
    ("DevOps Fundamentals", "DevOps assessment covering CI/CD, containerization, orchestration, and infrastructure as code.", "40 minutes", "Professional"),
    ("AWS Cloud Practitioner", "AWS cloud assessment covering core services, architecture, security, and pricing models.", "45 minutes", "Professional"),
    ("Azure Fundamentals", "Microsoft Azure assessment covering cloud concepts, core services, and management tools.", "40 minutes", "Professional"),
    ("GCP Cloud Engineering", "Google Cloud Platform assessment covering compute, storage, networking, and data services.", "45 minutes", "Professional"),
    ("Docker & Kubernetes", "Container orchestration assessment covering Docker, Kubernetes, deployments, and scaling.", "40 minutes", "Professional"),
    ("Machine Learning Basics", "Machine learning assessment covering supervised/unsupervised learning, evaluation, and common algorithms.", "45 minutes", "Professional"),
    ("Data Science Python", "Data science assessment covering pandas, numpy, visualization, and statistical analysis in Python.", "50 minutes", "Professional"),
    ("Database Administration", "Database administration assessment covering optimization, replication, backup, and security.", "40 minutes", "Professional"),
    ("MongoDB NoSQL", "MongoDB assessment covering document modeling, queries, aggregation, and indexing.", "35 minutes", "Professional"),
    ("PostgreSQL Advanced", "Advanced PostgreSQL assessment covering complex queries, performance tuning, and extensions.", "40 minutes", "Professional"),
    ("Redis & Caching", "Redis assessment covering data structures, caching strategies, pub/sub, and persistence.", "30 minutes", "Professional"),
    ("Network Security Basics", "Network security assessment covering protocols, encryption, vulnerabilities, and best practices.", "35 minutes", "Professional"),
    ("Cybersecurity Fundamentals", "Cybersecurity assessment covering threat modeling, risk assessment, and security controls.", "40 minutes", "Professional"),
    ("Linux System Administration", "Linux administration assessment covering shell scripting, system management, and troubleshooting.", "40 minutes", "Professional"),
    ("Git Version Control", "Git assessment covering branching strategies, merging, rebasing, and collaborative workflows.", "25 minutes", "Entry"),
    ("Agile & Scrum", "Agile methodology assessment covering Scrum framework, ceremonies, and agile principles.", "30 minutes", "Professional"),
    ("Software Testing & QA", "Software testing assessment covering test strategies, automation, and quality assurance practices.", "35 minutes", "Professional"),
    ("Selenium Automation", "Selenium test automation assessment covering WebDriver, page objects, and test frameworks.", "40 minutes", "Professional"),
    ("Mobile Development iOS", "iOS development assessment covering Swift, UIKit, SwiftUI, and App Store guidelines.", "45 minutes", "Professional"),
    ("Mobile Development Android", "Android development assessment covering Kotlin, Jetpack, and Android architecture components.", "45 minutes", "Professional"),
    ("React Native Development", "React Native assessment covering cross-platform development, navigation, and native modules.", "40 minutes", "Professional"),
    ("Flutter Development", "Flutter and Dart assessment covering widgets, state management, and cross-platform development.", "40 minutes", "Professional"),
    ("Blockchain Fundamentals", "Blockchain assessment covering distributed ledger concepts, consensus mechanisms, and smart contracts.", "35 minutes", "Professional"),
    ("Embedded Systems", "Embedded systems assessment covering microcontrollers, real-time systems, and hardware interfaces.", "45 minutes", "Professional"),
    ("MATLAB Engineering", "MATLAB assessment covering numerical computing, signal processing, and engineering applications.", "40 minutes", "Professional"),

    # Reasoning & Cognitive
    ("Deductive Reasoning", "Assesses ability to apply general rules to specific situations and draw logical conclusions.", "20 minutes", "All Levels"),
    ("Abstract Reasoning", "Measures ability to identify patterns, relationships, and logical rules in abstract information.", "22 minutes", "All Levels"),
    ("Spatial Reasoning", "Evaluates ability to visualize and manipulate objects in three-dimensional space.", "25 minutes", "Professional"),
    ("Critical Thinking", "Assesses ability to analyze arguments, identify assumptions, and evaluate evidence objectively.", "30 minutes", "Professional"),
    ("Problem Solving Advanced", "Advanced problem-solving assessment measuring analytical and strategic thinking skills.", "35 minutes", "Manager"),
    ("Logical Reasoning", "Measures ability to analyze complex information and identify logical relationships.", "25 minutes", "Professional"),
    ("Quantitative Analysis", "Assesses numerical analysis skills including financial data, statistics, and business metrics.", "30 minutes", "Manager"),
    ("Data Interpretation", "Evaluates ability to interpret complex data sets, charts, and statistical information.", "25 minutes", "Professional"),
    ("Business Acumen", "Assesses understanding of business concepts, financial literacy, and commercial awareness.", "35 minutes", "Manager"),
    ("Financial Analysis", "Financial analysis assessment covering ratios, statements, valuation, and financial modeling.", "40 minutes", "Manager"),
    ("Statistical Reasoning", "Statistical reasoning assessment covering probability, distributions, and hypothesis testing.", "30 minutes", "Professional"),
    ("Attention to Detail", "Assesses accuracy and thoroughness in identifying errors and inconsistencies in information.", "20 minutes", "Entry"),
    ("Error Checking", "Measures ability to identify mistakes and inconsistencies in text and numerical data.", "18 minutes", "Entry"),
    ("Checking & Calculation", "Assesses speed and accuracy in checking information and performing calculations.", "15 minutes", "Entry"),
    ("Reading Comprehension", "Evaluates ability to understand and analyze written passages from various sources.", "20 minutes", "Entry"),
    ("Written Communication", "Assesses written communication skills including grammar, clarity, and professional tone.", "25 minutes", "Professional"),
    ("Technical Writing", "Technical writing assessment covering documentation, clarity, and audience-appropriate communication.", "30 minutes", "Professional"),
]

# Personality/Behavioral (P) assessments
P_ASSESSMENTS = [
    ("Work Personality Index", "Comprehensive personality assessment measuring traits relevant to workplace success.", "30 minutes", "All Levels"),
    ("Emotional Intelligence", "Assesses emotional awareness, empathy, and ability to manage emotions in workplace settings.", "25 minutes", "All Levels"),
    ("Stress Tolerance", "Evaluates ability to maintain performance under pressure and manage workplace stress.", "20 minutes", "Professional"),
    ("Resilience Assessment", "Measures adaptability, perseverance, and ability to recover from setbacks.", "22 minutes", "Professional"),
    ("Team Collaboration", "Assesses teamwork skills including cooperation, communication, and conflict resolution.", "25 minutes", "All Levels"),
    ("Leadership Potential", "Identifies leadership qualities including vision, influence, and decision-making style.", "30 minutes", "Manager"),
    ("Executive Presence", "Evaluates executive qualities including gravitas, communication, and appearance.", "35 minutes", "Executive"),
    ("Strategic Thinking", "Assesses ability to think strategically and align actions with organizational goals.", "30 minutes", "Manager"),
    ("Change Management", "Evaluates ability to lead and adapt to organizational change initiatives.", "25 minutes", "Manager"),
    ("Innovation Mindset", "Measures creativity, openness to new ideas, and ability to drive innovation.", "25 minutes", "Professional"),
    ("Cultural Fit Assessment", "Evaluates alignment with organizational values, culture, and work environment.", "20 minutes", "All Levels"),
    ("Remote Work Readiness", "Assesses self-motivation, communication, and effectiveness in remote work settings.", "22 minutes", "All Levels"),
    ("Ethical Decision Making", "Evaluates integrity, ethical reasoning, and values-based decision making.", "25 minutes", "Professional"),
    ("Conflict Management Style", "Identifies preferred approach to handling workplace conflicts and disagreements.", "20 minutes", "Professional"),
    ("Communication Styles", "Assesses communication preferences and effectiveness across different contexts.", "22 minutes", "All Levels"),
    ("Negotiation Skills", "Evaluates ability to negotiate effectively and reach mutually beneficial agreements.", "25 minutes", "Professional"),
    ("Influencing Others", "Assesses ability to persuade, influence, and gain buy-in from others.", "22 minutes", "Professional"),
    ("Time Management", "Evaluates prioritization, planning, and time management effectiveness.", "20 minutes", "All Levels"),
    ("Decision Making Style", "Identifies decision-making approach including analytical, intuitive, and collaborative styles.", "25 minutes", "Professional"),
    ("Risk Tolerance", "Assesses comfort with uncertainty and approach to risk in business decisions.", "20 minutes", "Manager"),
    ("Achievement Orientation", "Measures drive for results, goal-setting behavior, and performance motivation.", "22 minutes", "Professional"),
    ("Service Orientation", "Evaluates customer focus, helpfulness, and dedication to serving others.", "20 minutes", "Entry"),
    ("Attention & Focus", "Assesses ability to maintain concentration and focus on tasks over time.", "18 minutes", "Entry"),
    ("Adaptability Scale", "Measures flexibility, openness to change, and ability to adjust to new situations.", "20 minutes", "All Levels"),
    ("Social Intelligence", "Evaluates ability to understand social dynamics and navigate interpersonal situations.", "25 minutes", "Professional"),
    ("Managerial Judgment", "Assesses judgment in management scenarios including people, tasks, and resources.", "30 minutes", "Manager"),
    ("Sales Personality", "Identifies personality traits associated with sales success including drive and resilience.", "25 minutes", "Professional"),
    ("Coaching Orientation", "Evaluates interest and ability in developing and coaching others.", "22 minutes", "Manager"),
    ("Project Management Style", "Assesses project management approach including planning, execution, and stakeholder management.", "28 minutes", "Professional"),
    ("Work Values Inventory", "Identifies work-related values and what motivates engagement and satisfaction.", "20 minutes", "All Levels"),
]

# Mixed (K, P) assessments
KP_ASSESSMENTS = [
    ("Software Engineer Complete", "Comprehensive software engineer assessment combining coding skills with collaboration and problem-solving style.", "70 minutes", "Professional"),
    ("Data Analyst Complete", "Complete data analyst assessment covering technical skills, analytical thinking, and communication.", "65 minutes", "Professional"),
    ("Product Manager Assessment", "Integrated product management assessment combining analytical ability with leadership and influence.", "60 minutes", "Manager"),
    ("UX Designer Assessment", "UX design assessment combining creative thinking, user empathy, and technical execution.", "55 minutes", "Professional"),
    ("Business Analyst Complete", "Comprehensive business analyst assessment covering analysis skills and stakeholder management.", "60 minutes", "Professional"),
    ("DevOps Engineer Complete", "Integrated DevOps assessment combining technical skills with collaboration and problem-solving.", "65 minutes", "Professional"),
    ("Team Lead Assessment", "Team lead assessment measuring technical capability, leadership, and team management.", "70 minutes", "Manager"),
    ("Technical Manager Battery", "Comprehensive technical manager assessment covering technical depth and people leadership.", "75 minutes", "Manager"),
    ("Consulting Analyst", "Consulting assessment combining analytical reasoning with client communication and influence.", "60 minutes", "Professional"),
    ("Finance Professional", "Finance professional assessment covering quantitative skills and business judgment.", "65 minutes", "Professional"),
    ("Marketing Manager Complete", "Marketing manager assessment combining analytical skills with creativity and leadership.", "60 minutes", "Manager"),
    ("HR Business Partner", "HR assessment combining analytical ability with interpersonal skills and business acumen.", "55 minutes", "Professional"),
    ("Operations Manager", "Operations management assessment covering analytical skills and leadership capability.", "65 minutes", "Manager"),
    ("Customer Success Manager", "Customer success assessment combining analytical skills with relationship management.", "55 minutes", "Professional"),
    ("Account Executive Battery", "Account executive assessment combining sales aptitude with relationship building.", "60 minutes", "Professional"),
]

LANGUAGES_OPTIONS = [
    "English",
    "English, Spanish",
    "English, Spanish, French",
    "English, Spanish, French, German",
    "English, Spanish, French, German, Chinese",
    "English, French",
    "English, German",
    "English, Chinese, Japanese",
]


def generate_slug(name: str) -> str:
    """Generate URL slug from assessment name."""
    slug = name.lower()
    slug = slug.replace(" & ", "-")
    slug = slug.replace(".", "")
    slug = slug.replace("+", "plus")
    slug = slug.replace("#", "sharp")
    slug = "".join(c if c.isalnum() or c == "-" else "-" for c in slug)
    slug = "-".join(part for part in slug.split("-") if part)
    return slug


def create_assessment_row(name: str, description: str, test_type: str, duration: str, job_level: str) -> dict:
    """Create a single assessment row matching the catalog schema."""
    slug = generate_slug(name)
    url = f"https://www.shl.com/solutions/products/product-catalog/view/{slug}/"

    # Randomize some fields for variety
    remote_testing = random.choice([True, True, True, False])  # 75% True
    adaptive_irt = random.choice([True, False, False])  # 33% True for K types
    languages = random.choice(LANGUAGES_OPTIONS)

    return {
        "name": name,
        "url": url,
        "description": description,
        "test_type": test_type,
        "remote_testing": remote_testing,
        "adaptive_irt": adaptive_irt if "K" in test_type else False,
        "duration": duration,
        "languages": languages,
        "job_levels": job_level,
    }


def augment_catalog(existing_df: pd.DataFrame, target_count: int = 100) -> pd.DataFrame:
    """
    Augment the catalog with synthetic assessments.

    Args:
        existing_df: Current catalog DataFrame
        target_count: Target number of total assessments

    Returns:
        Augmented DataFrame
    """
    current_count = len(existing_df)
    needed = target_count - current_count

    if needed <= 0:
        print(f"Catalog already has {current_count} assessments (target: {target_count})")
        return existing_df

    print(f"Current: {current_count} assessments")
    print(f"Target: {target_count} assessments")
    print(f"Adding: {needed} new assessments")

    # Get existing URLs to avoid duplicates
    existing_urls = set(existing_df["url"].tolist())
    existing_names = set(existing_df["name"].tolist())

    new_rows = []

    # Calculate distribution: ~65% K, ~25% P, ~10% K,P
    k_count = int(needed * 0.65)
    p_count = int(needed * 0.25)
    kp_count = needed - k_count - p_count

    # Add K assessments
    k_pool = [a for a in K_ASSESSMENTS if a[0] not in existing_names]
    random.shuffle(k_pool)
    for i, (name, desc, duration, level) in enumerate(k_pool[:k_count]):
        row = create_assessment_row(name, desc, "K", duration, level)
        if row["url"] not in existing_urls:
            new_rows.append(row)
            existing_urls.add(row["url"])

    # Add P assessments
    p_pool = [a for a in P_ASSESSMENTS if a[0] not in existing_names]
    random.shuffle(p_pool)
    for i, (name, desc, duration, level) in enumerate(p_pool[:p_count]):
        row = create_assessment_row(name, desc, "P", duration, level)
        if row["url"] not in existing_urls:
            new_rows.append(row)
            existing_urls.add(row["url"])

    # Add K,P assessments
    kp_pool = [a for a in KP_ASSESSMENTS if a[0] not in existing_names]
    random.shuffle(kp_pool)
    for i, (name, desc, duration, level) in enumerate(kp_pool[:kp_count]):
        row = create_assessment_row(name, desc, "K, P", duration, level)
        if row["url"] not in existing_urls:
            new_rows.append(row)
            existing_urls.add(row["url"])

    # Create new DataFrame and combine
    new_df = pd.DataFrame(new_rows)
    augmented_df = pd.concat([existing_df, new_df], ignore_index=True)

    return augmented_df


def main():
    parser = argparse.ArgumentParser(
        description="Augment SHL catalog with synthetic assessments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/augment_catalog.py              # Expand to 100 assessments
    python scripts/augment_catalog.py --target 150 # Expand to 150 assessments
    python scripts/augment_catalog.py --backup     # Create backup before augmenting
        """
    )
    parser.add_argument(
        "--target", "-t",
        type=int,
        default=100,
        help="Target number of assessments (default: 100)"
    )
    parser.add_argument(
        "--backup", "-b",
        action="store_true",
        help="Create backup of existing catalog before augmenting"
    )
    parser.add_argument(
        "--catalog", "-c",
        type=str,
        default="data/shl_catalog.csv",
        help="Path to catalog CSV (default: data/shl_catalog.csv)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  SHL Catalog Augmentation")
    print("=" * 60)

    catalog_path = project_root / args.catalog

    if not catalog_path.exists():
        print(f"\n[ERROR] Catalog not found: {catalog_path}")
        sys.exit(1)

    # Load existing catalog
    print(f"\n[LOAD] Loading catalog from: {catalog_path}")
    df = pd.read_csv(catalog_path)
    print(f"   Existing assessments: {len(df)}")

    # Backup if requested
    if args.backup:
        backup_path = catalog_path.with_suffix(".backup.csv")
        df.to_csv(backup_path, index=False)
        print(f"[BACKUP] Saved backup to: {backup_path}")

    # Augment
    print(f"\n[AUGMENT] Expanding catalog to {args.target} assessments...")
    augmented_df = augment_catalog(df, target_count=args.target)

    # Save
    augmented_df.to_csv(catalog_path, index=False)
    print(f"\n[SAVE] Saved augmented catalog to: {catalog_path}")

    # Summary
    print("\n" + "-" * 60)
    print("CATALOG SUMMARY")
    print("-" * 60)
    print(f"Total assessments: {len(augmented_df)}")
    print(f"\nBy Type:")
    print(augmented_df["test_type"].value_counts().to_string())
    print(f"\nBy Job Level:")
    print(augmented_df["job_levels"].value_counts().to_string())

    print("\n" + "=" * 60)
    print("  [DONE] Catalog augmentation complete!")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

