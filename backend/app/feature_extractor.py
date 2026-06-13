"""
13 ML Features extraction - FIXED VERSION
Better sector detection for logistics/transportation
"""

from typing import List, Dict, Any
import re

# Expanded sector keywords with better logistics detection
SECTOR_KEYWORDS = {
    "healthcare": ["healthcare", "medical", "hospital", "hipaa", "patient", "clinical environment", "medical records", 
                   "ehr", "pharmaceutical", "clinical drug", "hospital ward", "nursing care"],
    
    "finance": ["finance", "banking", "bank", "financial", "insurance", "investment", "fintech",
                "wealth", "trading", "payment", "credit", "loan", "mortgage", "accounting"],
    
    "government": ["government", "federal", "state", "municipal", "public sector", "agency",
                   "defense", "military", "civil", "public works", "regulatory", "compliance"],
    
    "technology": ["technology", "tech", "software", "digital", "it", "cloud", "saas",
                   "platform", "infrastructure", "devops", "cybersecurity", "ai", "ml"],
    
    "education": ["education", "school", "university", "college", "academic", "student",
                  "learning", "campus", "educational", "k-12", "higher education"],
    
    "retail": ["retail", "ecommerce", "store", "shop", "consumer", "merchant",
               "inventory", "point of sale", "pos", "customer service"],
    
    "logistics": ["logistics", "supply chain", "warehouse", "transportation", "freight",
                   "shipping", "delivery", "fleet", "courier", "cargo", "port", "harbor",
                   "trucking", "rail", "air freight", "sea freight", "inventory management",
                   "distribution", "fulfillment", "3pl", "cold chain", "parcel", "last mile",
                   "supply chain management", "freight forwarding", "transport", "carrier"],
    
    "manufacturing": ["manufacturing", "industrial", "factory", "production", "assembly",
                      "plant", "machinery", "automation", "quality control", "lean"],
    
    "energy": ["energy", "utility", "power", "renewable", "solar", "wind", "oil", "gas",
               "petroleum", "electric", "grid", "battery", "storage"],
    
    "telecommunications": ["telecom", "telecommunications", "network", "5g", "broadband",
                           "fiber", "mobile", "cellular", "communication", "connectivity"],
    
    "real estate": ["real estate", "property", "construction", "building", "infrastructure",
                    "architecture", "engineering", "civil", "housing", "commercial property"],
    
    "hospitality": ["hospitality", "hotel", "restaurant", "tourism", "travel", "resort",
                    "catering", "food service", "lodging"],
    
    "agriculture": ["agriculture", "farming", "agri", "crop", "livestock", "irrigation",
                    "food production", "harvest", "rural"]
}

# Anti-patterns - if these appear, DON'T match certain sectors
ANTI_PATTERNS = {
    "healthcare": ["logistics", "supply chain", "warehouse", "transportation", "shipping"],
    "logistics": ["healthcare", "medical", "hospital", "patient", "clinical"]
}


def extract_sector_from_text(text: str) -> str:
    """
    Extract industry sector from RFP text with improved detection
    Fixes the issue where logistics RFP was showing as healthcare
    """
    text_lower = text.lower()
    
    # Score each sector based on keyword matches
    scores = {}
    for sector, keywords in SECTOR_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in text_lower:
                # Weight longer/phrase keywords more
                weight = 2 if ' ' in keyword else 1
                score += weight
        if score > 0:
            scores[sector] = score
    
    if not scores:
        return "general"
    
    # Get top candidate
    top_sector = max(scores, key=scores.get)
    
    # Check anti-patterns
    anti_patterns = ANTI_PATTERNS.get(top_sector, [])
    for anti in anti_patterns:
        if anti in text_lower and scores.get(top_sector, 0) < 3:
            # If anti-pattern found and score is low, return general
            return "general"
    
    # Also check if another sector has similar score
    sorted_sectors = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_sectors) > 1:
        top_score = sorted_sectors[0][1]
        second_score = sorted_sectors[1][1]
        
        # If scores are close, check which one is more specific
        if top_score - second_score <= 2:
            # Could be ambiguous, return general
            return "general"
    
    return top_sector


def parse_budget_amount(budget_str: str) -> float:
    """Parse budget string to numeric value in USD - FIXED for NLA logistics RFP"""
    if not budget_str or budget_str == "Not specified":
        return 500000
    
    budget_lower = budget_str.lower()
    
    # Handle PKR currency with range (e.g., PKR 250M – PKR 400M)
    if 'pkr' in budget_lower or 'rs' in budget_lower:
        # Extract all numbers that look like millions
        mil_matches = re.findall(r'(\d+(?:\.\d+)?)\s*[m|million]', budget_lower)
        if mil_matches:
            values = [float(m) * 1000000 for m in mil_matches]
            avg_val = sum(values) / len(values)
            return max(avg_val / 280, 0) # Convert to approximate USD
            
        numbers = re.findall(r'[\d,]+(?:\.\d+)?', budget_str)
        if numbers:
            value = float(numbers[0].replace(',', ''))
            if 'million' in budget_lower or 'm' in budget_lower:
                value = value * 1000000 / 280
            elif 'crore' in budget_lower:
                value = value * 10000000 / 280
            else:
                value = value / 280
            return max(value, 0)
    
    numbers = re.findall(r'[\d,]+(?:\.\d+)?', budget_str)
    if not numbers:
        return 500000
    
    value = float(numbers[0].replace(',', ''))
    
    if 'million' in budget_lower or 'm' in budget_lower:
        value = value * 1000000
    elif 'billion' in budget_lower or 'b' in budget_lower:
        value = value * 1000000000
    elif 'k' in budget_lower or 'thousand' in budget_lower:
        value = value * 1000
    
    if ('-' in budget_str or '–' in budget_str) and len(numbers) > 1:
        try:
            second_value = float(numbers[1].replace(',', ''))
            if 'million' in budget_lower or 'm' in budget_lower:
                second_value = second_value * 1000000
            elif 'k' in budget_lower:
                second_value = second_value * 1000
            value = (value + second_value) / 2
        except:
            pass
    
    return max(value, 0)


def calculate_response_time_days(deadline_str: str) -> int:
    """Calculate number of days until deadline"""
    if not deadline_str or deadline_str == "Not specified":
        return 14
    
    import re
    from datetime import datetime
    
    date_patterns = [
        (r'(\d{1,2}/\d{1,2}/\d{4})', ['%m/%d/%Y', '%d/%m/%Y']),
        (r'(\d{4}-\d{1,2}-\d{1,2})', ['%Y-%m-%d']),
        (r'([A-Za-z]+ \d{1,2},? \d{4})', ['%B %d, %Y', '%b %d, %Y']),
        (r'(\d{1,2} [A-Za-z]+ \d{4})', ['%d %B %Y', '%d %b %Y'])
    ]
    
    for pattern, formats in date_patterns:
        match = re.search(pattern, deadline_str, re.IGNORECASE)
        if match:
            date_str = match.group(1)
            for fmt in formats:
                try:
                    deadline = datetime.strptime(date_str, fmt)
                    days = (deadline - datetime.now()).days
                    return max(days, 1)
                except ValueError:
                    continue
    
    return 14


def estimate_document_pages(text: str) -> int:
    """Estimate number of pages from text length - FIXED for long RFPs"""
    if not text:
        return 1
    word_count = len(text.split())
    # Standard page is ~400 words
    pages = max(1, (word_count // 400) + 1)
    return min(pages, 100)


def normalize_budget(budget: float) -> float:
    """Normalize budget to 0-1 scale (cap at $5M)"""
    max_budget = 5_000_000
    return min(budget / max_budget, 1.0)


def count_gaps_from_matches(matches: List[Dict]) -> int:
    """Count number of gaps (requirements with low match score)"""
    if not matches:
        return 5
    return sum(1 for m in matches if m.get('score', 0) < 50)


def calculate_compliance_score(matches: List[Dict], requirements: List[str]) -> float:
    """Calculate compliance score based on matches"""
    if not matches or not requirements:
        return 50.0
    
    scores = [m.get('score', 0) for m in matches]
    avg_score = sum(scores) / len(scores) if scores else 50
    
    high_scores = sum(1 for s in scores if s >= 70)
    high_score_bonus = (high_scores / len(scores)) * 10 if scores else 0
    
    return min(avg_score + high_score_bonus, 100)


def calculate_efficiency_score(compliance: float, gaps: int, pages: int) -> float:
    """Calculate efficiency score"""
    gap_penalty = min(gaps * 5, 50)
    page_penalty = min(pages / 10, 20)
    efficiency = compliance - gap_penalty - page_penalty
    return max(min(efficiency, 100), 0)


def calculate_pages_per_gap(pages: int, gaps: int) -> float:
    """Calculate pages per gap ratio"""
    if gaps == 0:
        return pages * 10
    return min(pages / gaps, 100)


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
        "government": 1, "healthcare": 2, "finance": 3,
        "education": 4, "technology": 5, "manufacturing": 6,
        "retail": 7, "energy": 8, "logistics": 9,
        "telecommunications": 10, "real estate": 11,
        "hospitality": 12, "agriculture": 13, "general": 0
    }
    return sector_map.get(sector, 0)


def create_feature_vector(features: Dict[str, Any]) -> list:
    """Create feature vector in correct order for ML model"""
    sector_encoded = encode_sector_to_numeric(features.get("sector", "general"))
    
    return [
        float(sector_encoded),
        float(features.get("budget", 500000)),
        float(features.get("compliance", 50)),
        float(features.get("score", 50)),
        float(features.get("response_time", 14)),
        float(features.get("doc_pages", 20)),
        float(features.get("gaps_found", 5)),
        float(features.get("bid_manager", 3)),
        float(features.get("compliance_score_norm", 0.5)),
        float(features.get("budget_norm", 0.1)),
        float(features.get("efficiency_score", 50)),
        float(features.get("pages_per_gap", 4)),
        float(features.get("compliance_budget_ratio", 1))
    ]