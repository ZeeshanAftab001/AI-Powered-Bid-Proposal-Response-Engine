"""
Utility functions for text extraction
"""

import io
import re
from typing import Optional
import PyPDF2
from docx import Document


def extract_text(file_bytes: bytes, url_str: str) -> Optional[str]:
    """
    Extract text from file bytes based on file type
    
    Args:
        file_bytes: File content as bytes
        url_str: URL string to detect file type
    
    Returns:
        Extracted text or None if unsupported
    """
    
    if url_str.endswith('.pdf'):
        # PDF extraction
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text.strip()
        except Exception as e:
            print(f"PDF extraction error: {e}")
            return None
            
    elif url_str.endswith('.docx'):
        # DOCX extraction
        try:
            doc = Document(io.BytesIO(file_bytes))
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
            return text.strip()
        except Exception as e:
            print(f"DOCX extraction error: {e}")
            return None
            
    elif url_str.endswith('.txt'):
        # TXT extraction
        try:
            return file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            try:
                return file_bytes.decode('latin-1')
            except:
                return None
    
    return None


def extract_sector_from_text(text: str) -> str:
    """Extract industry sector from RFP text"""
    text_lower = text.lower()
    
    sector_keywords = {
        "healthcare": ["healthcare", "medical", "hospital", "hipaa", "patient", "clinic"],
        "finance": ["finance", "banking", "bank", "financial", "insurance", "investment"],
        "government": ["government", "federal", "state", "municipal", "public sector"],
        "technology": ["technology", "tech", "software", "digital", "it", "cloud"],
        "education": ["education", "school", "university", "college", "academic"],
        "retail": ["retail", "ecommerce", "store", "shop", "consumer", "logistics"],
        "manufacturing": ["manufacturing", "industrial", "factory", "production"],
        "energy": ["energy", "utility", "power", "renewable", "solar", "wind"]
    }
    
    scores = {}
    for sector, keywords in sector_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            scores[sector] = score
    
    if scores:
        return max(scores, key=scores.get)
    return "general"


def parse_budget_amount(budget_str: str) -> float:
    """Parse budget string to numeric value"""
    if not budget_str or budget_str == "Not specified":
        return 500000
    
    numbers = re.findall(r'[\d,]+(?:\.\d+)?', budget_str)
    if not numbers:
        return 500000
    
    value = float(numbers[0].replace(',', ''))
    
    if 'million' in budget_str.lower() or 'm' in budget_str.lower():
        value = value * 1000000
    elif 'k' in budget_str.lower() or 'thousand' in budget_str.lower():
        value = value * 1000
    
    return max(value, 0)