"""
Feature extraction utilities for ML features
"""

import re
from datetime import datetime
from typing import List, Dict, Any, Optional

def extract_sector_from_text(text: str) -> str:
    """Extract industry sector from RFP text"""
    text_lower = text.lower()
    
    sector_keywords = {
        "healthcare": ["healthcare", "medical", "hospital", "hipaa", "patient", "clinic", "health"],
        "finance": ["finance", "banking", "bank", "financial", "insurance", "investment", "fintech"],
        "government": ["government", "federal", "state", "municipal", "public sector", "agency"],
        "technology": ["technology", "tech", "software", "digital", "it", "cloud", "saas"],
        "education": ["education", "school", "university", "college", "academic", "student"],
        "retail": ["retail", "ecommerce", "store", "shop", "consumer", "merchant"],
        "manufacturing": ["manufacturing", "industrial", "factory", "production", "supply chain"],
        "energy": ["energy", "utility", "power", "renewable", "oil", "gas", "solar"]
    }
    
    for sector, keywords in sector_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            return sector
    
    return "general"


def parse_budget_amount(budget_str: str) -> float:
    """Parse budget string to numeric value in USD"""
    if not budget_str or budget_str == "Not specified":
        return 500000  # Default $500k
    
    budget_lower = budget_str.lower()
    
    # Find all numbers in the string
    numbers = re.findall(r'[\d,]+(?:\.\d+)?', budget_str)
    
    if not numbers:
        return 500000
    
    # Get first number
    value = float(numbers[0].replace(',', ''))
    
    # Check for multipliers
    if 'million' in budget_lower or 'm' in budget_lower:
        value = value * 1000000
    elif 'k' in budget_lower or 'thousand' in budget_lower:
        value = value * 1000
    
    # Check if it's a range (e.g., $500k - $1M)
    if '-' in budget_str and len(numbers) > 1:
        second_value = float(numbers[1].replace(',', ''))
        if 'million' in budget_lower or 'm' in budget_lower:
            second_value = second_value * 1000000
        elif 'k' in budget_lower:
            second_value = second_value * 1000
        value = (value + second_value) / 2  # Average of range
    
    return max(value, 0)


def calculate_response_time_days(deadline_str: str) -> int:
    """Calculate number of days until deadline"""
    if not deadline_str or deadline_str == "Not specified":
        return 14
    
    date_patterns = [
        r'(\d{1,2}/\d{1,2}/\d{4})',           # MM/DD/YYYY
        r'(\d{4}-\d{1,2}-\d{1,2})',            # YYYY-MM-DD
        r'([A-Za-z]+ \d{1,2},? \d{4})',        # Month DD, YYYY
        r'(\d{1,2} [A-Za-z]+ \d{4})'           # DD Month YYYY
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, deadline_str, re.IGNORECASE)
        if match:
            date_str = match.group(1)
            for fmt in ['%m/%d/%Y', '%Y-%m-%d', '%B %d, %Y', '%b %d, %Y', '%d %B %Y', '%d %b %Y']:
                try:
                    deadline = datetime.strptime(date_str, fmt)
                    days = (deadline - datetime.now()).days
                    return max(days, 1)
                except ValueError:
                    continue
    
    return 14


def estimate_document_pages(text: str) -> int:
    """Estimate number of pages from text length"""
    word_count = len(text.split())
    pages = max(1, word_count // 500)
    return min(pages, 100)


def count_gaps_from_matches(matches: List[Dict]) -> int:
    """Count number of gaps (requirements with low match score)"""
    if not matches:
        return 5
    
    gaps = sum(1 for m in matches if m.get('score', 0) < 50)
    return gaps


def normalize_budget(budget: float) -> float:
    """Normalize budget to 0-1 scale (cap at $5M)"""
    max_budget = 5_000_000
    normalized = min(budget / max_budget, 1.0)
    return normalized


def calculate_compliance_score(matches: List[Dict], requirements: List[str]) -> float:
    """Calculate compliance score based on matches"""
    if not matches or not requirements:
        return 50.0
    
    scores = [m.get('score', 0) for m in matches]
    avg_score = sum(scores) / len(scores) if scores else 50
    
    high_scores = sum(1 for s in scores if s >= 70)
    high_score_bonus = (high_scores / len(scores)) * 10
    
    return min(avg_score + high_score_bonus, 100)


def calculate_efficiency_score(compliance: float, gaps: int, pages: int) -> float:
    """Calculate efficiency score based on compliance, gaps, and pages"""
    gap_penalty = gaps * 5
    page_penalty = pages / 10
    efficiency = compliance - gap_penalty - page_penalty
    return max(min(efficiency, 100), 0)


def calculate_pages_per_gap(pages: int, gaps: int) -> float:
    """Calculate pages per gap ratio"""
    if gaps == 0:
        return pages * 10
    ratio = pages / gaps
    return min(ratio, 100)


def calculate_compliance_budget_ratio(compliance: float, budget: float) -> float:
    """Calculate compliance to budget ratio"""
    norm_budget = normalize_budget(budget)
    
    if norm_budget == 0:
        return compliance / 100
    
    ratio = (compliance / 100) / norm_budget
    return min(ratio, 2.0)


def encode_sector_to_numeric(sector: str) -> int:
    """Encode sector string to numeric value"""
    sector_map = {
        "government": 1,
        "healthcare": 2,
        "finance": 3,
        "education": 4,
        "technology": 5,
        "manufacturing": 6,
        "retail": 7,
        "energy": 8,
        "general": 0
    }
    return sector_map.get(sector, 0)


def create_feature_vector(features: Dict[str, Any]) -> List[float]:
    """Create feature vector in correct order for ML model"""
    sector_encoded = encode_sector_to_numeric(features.get("sector", "general"))
    
    feature_vector = [
        float(sector_encoded),                    # 1. sector (encoded)
        float(features.get("budget", 500000)),    # 2. budget
        float(features.get("compliance", 50)),    # 3. compliance
        float(features.get("score", 50)),         # 4. score
        float(features.get("response_time", 14)), # 5. response_time
        float(features.get("doc_pages", 20)),     # 6. doc_pages
        float(features.get("gaps_found", 5)),     # 7. gaps_found
        float(features.get("bid_manager", 3)),    # 8. bid_manager
        float(features.get("compliance_score_norm", 0.5)),  # 9. compliance_norm
        float(features.get("budget_norm", 0.1)),           # 10. budget_norm
        float(features.get("efficiency_score", 50)),       # 11. efficiency_score
        float(features.get("pages_per_gap", 4)),           # 12. pages_per_gap
        float(features.get("compliance_budget_ratio", 1))  # 13. ratio
    ]
    
    return feature_vector