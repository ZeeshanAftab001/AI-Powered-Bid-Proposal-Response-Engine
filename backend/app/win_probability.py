"""
Win probability calculation utilities
"""

import re
from typing import List, Dict, Any, Tuple

class WinProbabilityScorer:
    """Comprehensive win probability scoring using heuristics"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def calculate(self, 
                  rfp_text: str,
                  requirements: List[str],
                  matched_capabilities: Dict[str, List[str]],
                  match_scores: Dict[str, float]) -> Dict[str, Any]:
        """Calculate win probability using multiple heuristics"""
        
        components = {}
        strengths = []
        weaknesses = []
        action_items = []
        
        # 1. CAPABILITY MATCH SCORE (Weight: 25%)
        cap_score = self._score_capability_match(matched_capabilities, match_scores)
        components["Capability Match"] = {"score": cap_score, "weight": 0.25}
        if cap_score >= 80:
            strengths.append("✓ Strong capability match with multiple evidence points")
        elif cap_score < 50:
            weaknesses.append("✗ Weak capability match, significant gaps identified")
            action_items.append("→ Develop or acquire missing capabilities")
        
        # 2. PAST PERFORMANCE & EXPERIENCE (Weight: 20%)
        past_score = self._score_past_performance(matched_capabilities)
        components["Past Performance"] = {"score": past_score, "weight": 0.20}
        if past_score >= 75:
            strengths.append("✓ Extensive past performance evidence available")
        elif past_score < 50:
            weaknesses.append("✗ Limited past performance evidence")
            action_items.append("→ Gather more case studies and references")
        
        # 3. COMPETITIVE LANDSCAPE (Weight: 15%)
        comp_score = self._score_competitive_landscape(rfp_text)
        components["Competitive Position"] = {"score": comp_score, "weight": 0.15}
        if comp_score >= 70:
            strengths.append("✓ Favorable competitive landscape")
        elif comp_score < 50:
            weaknesses.append("✗ Highly competitive with strong incumbents")
            action_items.append("→ Develop unique differentiators")
        
        # 4. BUDGET & PRICE ALIGNMENT (Weight: 12%)
        budget_score = self._score_budget_alignment(rfp_text)
        components["Budget Alignment"] = {"score": budget_score, "weight": 0.12}
        if budget_score < 50:
            weaknesses.append("✗ Budget unclear or misaligned")
            action_items.append("→ Clarify budget or adjust pricing strategy")
        
        # 5. COMPLIANCE STATUS (Weight: 10%)
        compliance_score = self._score_compliance(requirements, matched_capabilities)
        components["Compliance"] = {"score": compliance_score, "weight": 0.10}
        if compliance_score >= 80:
            strengths.append("✓ All mandatory requirements met")
        elif compliance_score < 60:
            weaknesses.append("✗ Compliance gaps in requirements")
            action_items.append("→ Address compliance gaps urgently")
        
        # 6. RESOURCE AVAILABILITY (Weight: 10%)
        resource_score = self._score_resources(requirements, matched_capabilities)
        components["Resource Availability"] = {"score": resource_score, "weight": 0.10}
        if resource_score < 50:
            weaknesses.append("✗ Resource constraints may impact delivery")
            action_items.append("→ Assess resource availability and plan allocation")
        
        # 7. CLIENT RELATIONSHIP (Weight: 5%)
        relationship_score = self._score_relationship(rfp_text)
        components["Client Relationship"] = {"score": relationship_score, "weight": 0.05}
        if relationship_score >= 60:
            strengths.append("✓ Existing relationship provides advantage")
        
        # 8. TIMELINE FEASIBILITY (Weight: 3%)
        timeline_score = self._score_timeline(rfp_text)
        components["Timeline Feasibility"] = {"score": timeline_score, "weight": 0.03}
        if timeline_score < 50:
            weaknesses.append("✗ Timeline is aggressive")
            action_items.append("→ Develop detailed project schedule")
        
        # Calculate weighted total
        total_score = sum(data["score"] * data["weight"] for data in components.values())
        
        # Get recommendation
        recommendation, rec_details = self._get_recommendation(total_score, components)
        
        # Estimate win chance
        win_chance = self._estimate_win_chance(total_score, components)
        
        return {
            "total_score": round(total_score, 1),
            "recommendation": recommendation,
            "estimated_win_chance": win_chance,
            "component_scores": {k: round(v["score"], 1) for k, v in components.items()},
            "strengths": strengths[:5],
            "weaknesses": weaknesses[:5],
            "action_items": action_items[:5],
            "recommendation_details": rec_details
        }
    
    def _score_capability_match(self, matches: Dict, scores: Dict) -> float:
        if not matches:
            return 0.0
        total_requirements = len(matches)
        if total_requirements == 0:
            return 0.0
        
        avg_match_score = sum(scores.values()) / len(scores) if scores else 0
        strong_matches = sum(1 for caps in matches.values() if len(caps) >= 3)
        strong_match_percentage = (strong_matches / total_requirements) * 100
        
        return min(100, (avg_match_score * 0.6) + (strong_match_percentage * 0.4))
    
    def _score_past_performance(self, matches: Dict) -> float:
        total_evidence = sum(len(caps) for caps in matches.values())
        
        if total_evidence == 0:
            return 20.0
        
        has_relevant = any(
            any(kw in str(cap).lower() for kw in ['project', 'implementation'])
            for caps in matches.values() for cap in caps
        )
        
        has_certs = any(
            any(kw in str(cap).lower() for kw in ['iso', 'certified', 'cmmi'])
            for caps in matches.values() for cap in caps
        )
        
        score = 50.0
        score += min(30, total_evidence * 5)
        if has_relevant:
            score += 10
        if has_certs:
            score += 10
        
        return min(100, score)
    
    def _score_competitive_landscape(self, rfp_text: str) -> float:
        competitor_indicators = [
            'incumbent', 'existing vendor', 'current supplier',
            'qualified vendors', 'shortlisted', 'preferred'
        ]
        
        text_lower = rfp_text.lower()
        competitor_mentions = sum(1 for ind in competitor_indicators if ind in text_lower)
        
        if competitor_mentions >= 3:
            return 30.0
        elif competitor_mentions >= 1:
            return 50.0
        else:
            return 70.0
    
    def _score_budget_alignment(self, rfp_text: str) -> float:
        budget_patterns = [
            r'budget[:\s]*([\d,]+(?:\.\d+)?)\s*(?:M|Million)',
            r'\$[\d,]+(?:\.\d+)?\s*(?:M|Million)?',
        ]
        
        budget_found = any(re.search(p, rfp_text, re.IGNORECASE) for p in budget_patterns)
        return 65.0 if budget_found else 45.0
    
    def _score_compliance(self, requirements: List[str], matches: Dict) -> float:
        if not requirements:
            return 50.0
        
        mandatory_keywords = ['must', 'mandatory', 'required', 'shall']
        mandatory_reqs = [
            req for req in requirements 
            if any(kw in req.lower() for kw in mandatory_keywords)
        ]
        
        if not mandatory_reqs:
            return 80.0
        
        matched_count = sum(1 for req in mandatory_reqs if req in matches and matches[req])
        return (matched_count / len(mandatory_reqs)) * 100 if mandatory_reqs else 50.0
    
    def _score_resources(self, requirements: List[str], matches: Dict) -> float:
        complexity_indicators = [
            'multiple', 'integration', 'custom', 'complex',
            'enterprise', 'large-scale', 'nationwide'
        ]
        
        text = ' '.join(requirements).lower()
        complexity_score = sum(1 for ind in complexity_indicators if ind in text)
        
        if complexity_score >= 5:
            resource_need = 80
        elif complexity_score >= 3:
            resource_need = 60
        else:
            resource_need = 40
        
        return max(0, 100 - resource_need)
    
    def _score_relationship(self, rfp_text: str) -> float:
        relationship_indicators = {
            'repeat': 20, 'existing': 15, 'current': 10,
            'long-term': 15, 'partnership': 10
        }
        
        text_lower = rfp_text.lower()
        relationship_score = 30
        
        for indicator, points in relationship_indicators.items():
            if indicator in text_lower:
                relationship_score += points
        
        return min(100, relationship_score)
    
    def _score_timeline(self, rfp_text: str) -> float:
        urgency_indicators = ['urgent', 'immediate', 'asap', 'short timeline']
        text_lower = rfp_text.lower()
        
        if any(ind in text_lower for ind in urgency_indicators):
            return 40.0
        else:
            return 70.0
    
    def _get_recommendation(self, total_score: float, components: Dict) -> Tuple[str, str]:
        critical_components = [
            name for name, data in components.items() 
            if data["score"] < 30 and data["weight"] > 0.10
        ]
        
        if critical_components or total_score < 40:
            return "STRONG NO-GO 🔴🔴", f"Critical gaps in: {', '.join(critical_components)}"
        elif total_score < 55:
            return "NO-GO 🔴", "Overall capability match below threshold"
        elif total_score < 65:
            return "CONSIDER 🤔", "Marginal fit, need risk mitigation"
        elif total_score < 80:
            return "GO 🟢", "Good fit with manageable gaps"
        else:
            return "STRONG GO 🟢🟢", "Excellent fit with strong win probability"
    
    def _estimate_win_chance(self, total_score: float, components: Dict) -> str:
        if total_score >= 80:
            return "75-85%"
        elif total_score >= 70:
            return "65-75%"
        elif total_score >= 60:
            return "55-65%"
        elif total_score >= 50:
            return "45-55%"
        elif total_score >= 40:
            return "35-45%"
        else:
            return "<35%"