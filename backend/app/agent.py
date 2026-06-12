import os
from typing import List, Optional, TypedDict, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from app.rag import capability_rag
from dotenv import load_dotenv

from app.feature_extractor import (
    extract_sector_from_text,
    parse_budget_amount,
    calculate_response_time_days,
    estimate_document_pages,
    count_gaps_from_matches,
    normalize_budget,
    calculate_compliance_score,
    calculate_efficiency_score,
    calculate_pages_per_gap,
    calculate_compliance_budget_ratio,
    encode_sector_to_numeric,
    create_feature_vector
)
from app.win_probability import WinProbabilityScorer
from app.ml_predictor import get_ml_predictor

load_dotenv()

class ParsedRFP(BaseModel):
    deadline: Optional[str] = Field(description="RFP deadline date")
    budget: Optional[str] = Field(description="Budget amount")
    mandatory_requirements: List[str] = Field(description="List of mandatory requirements")
    evaluation_criteria: List[str] = Field(default=[], description="Evaluation criteria")

class RequirementMatch(BaseModel):
    requirement: str = Field(description="The requirement text")
    score: int = Field(description="Match score 0-100")
    evidence: List[str] = Field(description="Supporting evidence from RAG")
    reasoning: str = Field(description="Match reasoning")

class MatchResults(BaseModel):
    matches: List[RequirementMatch] = Field(description="All requirement matches")
    average_score: float = Field(description="Average match score")

class ProposalSection(BaseModel):
    section_title: str = Field(description="Section title")
    content: str = Field(description="Proposal content for this section")
    mapped_requirements: List[str] = Field(description="Requirements addressed")

class GeneratedProposal(BaseModel):
    executive_summary: str = Field(description="Executive summary")
    company_overview: str = Field(description="Company capabilities overview")
    response_sections: List[ProposalSection] = Field(description="Detailed responses")
    conclusion: str = Field(description="Conclusion")

# ============== Initialize LLM ==============

try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    print("✓ Gemini model initialized")
except Exception as e:
    print(f"⚠️ Gemini not available: {e}")
    llm = None

structured_llm_parser = llm.with_structured_output(ParsedRFP) if llm else None
structured_llm_matcher = llm.with_structured_output(MatchResults) if llm else None
structured_llm_proposal = llm.with_structured_output(GeneratedProposal) if llm else None

# ============== Define State ==============

class RFPState(TypedDict):
    rfp_text: str
    deadline: str
    budget: str
    mandatory_requirements: List[str]
    evaluation_criteria: List[str]
    retrieved_capabilities: Dict[str, List[str]]
    company_capabilities: List[str]
    match_results: Dict[str, Any]
    proposal: Dict[str, Any]
    final_score: float
    recommendation: str
    ml_features: Dict[str, Any]
    feature_vector: List[float]
    ml_prediction: Dict[str, Any]

# ============== Node Functions ==============

def parse_rfp(state: RFPState) -> RFPState:
    print("\n📄 Step 1: Parsing RFP document...")
    
    if not structured_llm_parser:
        raise Exception("LLM not available")
    
    prompt = ChatPromptTemplate.from_template("""
    Parse this RFP document and extract the required information.
    
    RFP Text:
    {text}
    
    Extract:
    - Deadline date
    - Budget amount
    - All mandatory requirements
    - Evaluation criteria
    """)
    
    chain = prompt | structured_llm_parser
    
    try:
        result = chain.invoke({"text": state.get("rfp_text", "")[:4000]})
        
        state["deadline"] = result.deadline or "Not specified"
        state["budget"] = result.budget or "Not specified"
        state["mandatory_requirements"] = result.mandatory_requirements
        state["evaluation_criteria"] = result.evaluation_criteria
        
        print(f"   ✓ Found {len(result.mandatory_requirements)} mandatory requirements")
        return state
    except Exception as e:
        print(f"   ✗ Error: {e}")
        state["mandatory_requirements"] = []
        return state


def retrieve_from_rag(state: RFPState) -> RFPState:
    print("\n🔍 Step 2: Retrieving capabilities from RAG...")
    
    requirements = state.get("mandatory_requirements", [])
    
    if not requirements:
        return state
    
    retrieved = {}
    for req in requirements[:10]:
        capabilities = capability_rag.retrieve_capabilities(req, k=3)
        retrieved[req] = capabilities
        print(f"   ✓ Retrieved {len(capabilities)} capabilities for: {req[:50]}...")
    
    state["retrieved_capabilities"] = retrieved
    
    all_capabilities = []
    for caps in retrieved.values():
        all_capabilities.extend(caps)
    state["company_capabilities"] = list(set(all_capabilities))
    
    return state


def match_requirements(state: RFPState) -> RFPState:
    print("\n🎯 Step 3: Matching requirements...")
    
    requirements = state.get("mandatory_requirements", [])
    retrieved_caps = state.get("retrieved_capabilities", {})
    
    if not requirements:
        state["match_results"] = {"matches": [], "average_score": 0}
        return state
    
    matches = []
    total_score = 0
    
    for req in requirements[:10]:
        caps = retrieved_caps.get(req, [])
        if caps:
            score = min(100, len(caps) * 25)
            evidence = caps[:3]
            reasoning = f"Found {len(caps)} matching capabilities"
        else:
            score = 0
            evidence = []
            reasoning = "No matching capabilities found"
        
        matches.append({
            "requirement": req,
            "score": score,
            "evidence": evidence,
            "reasoning": reasoning
        })
        total_score += score
    
    avg_score = total_score / len(matches) if matches else 0
    
    state["match_results"] = {
        "matches": matches,
        "average_score": avg_score
    }
    
    print(f"   ✓ Average match score: {avg_score:.1f}%")
    return state


def generate_ml_features(state: RFPState) -> RFPState:
    """Generate all 13 ML features using utils"""
    
    print("\n📊 Step: Generating ML Features...")
    
    rfp_text = state.get("rfp_text", "")
    deadline = state.get("deadline", "Not specified")
    budget_str = state.get("budget", "Not specified")
    match_results = state.get("match_results", {})
    matches = match_results.get("matches", [])
    avg_match_score = match_results.get("average_score", 50.0)
    
    # Extract features using utils
    sector = extract_sector_from_text(rfp_text)
    budget = parse_budget_amount(budget_str)
    compliance = calculate_compliance_score(matches, state.get("mandatory_requirements", []))
    score = avg_match_score
    response_time = calculate_response_time_days(deadline)
    doc_pages = estimate_document_pages(rfp_text)
    gaps_found = count_gaps_from_matches(matches)
    bid_manager_level = state.get("bid_manager_level", 3)
    
    compliance_score_norm = compliance / 100
    budget_norm = normalize_budget(budget)
    efficiency_score = calculate_efficiency_score(compliance, gaps_found, doc_pages)
    pages_per_gap = calculate_pages_per_gap(doc_pages, gaps_found)
    compliance_budget_ratio = calculate_compliance_budget_ratio(compliance, budget)
    
    # Store features
    state["ml_features"] = {
        "sector": sector,
        "budget": budget,
        "compliance": compliance,
        "score": score,
        "response_time": response_time,
        "doc_pages": doc_pages,
        "gaps_found": gaps_found,
        "bid_manager": bid_manager_level,
        "compliance_score_norm": compliance_score_norm,
        "budget_norm": budget_norm,
        "efficiency_score": efficiency_score,
        "pages_per_gap": pages_per_gap,
        "compliance_budget_ratio": compliance_budget_ratio
    }
    
    # Create feature vector
    state["feature_vector"] = create_feature_vector(state["ml_features"])
    
    print(f"   ✓ All 13 ML features generated")
    return state


def predict_with_ml(state: RFPState) -> RFPState:
    """Make ML prediction using utils"""
    
    print("\n🎯 Step: Running ML Prediction...")
    
    feature_vector = state.get("feature_vector", [])
    
    if not feature_vector:
        state = generate_ml_features(state)
        feature_vector = state.get("feature_vector", [])
    
    # Use ML predictor
    predictor = get_ml_predictor()
    prediction = predictor.predict(feature_vector)
    
    state["ml_prediction"] = prediction
    
    print(f"   ✓ Win Probability: {prediction['win_probability']*100:.1f}%")
    print(f"   ✓ Outcome: {prediction['outcome']}")
    
    return state


def generate_proposal(state: RFPState) -> RFPState:
    print("\n✍️ Step 4: Generating proposal...")
    
    requirements = state.get("mandatory_requirements", [])
    retrieved_caps = state.get("retrieved_capabilities", {})
    avg_score = state.get("match_results", {}).get("average_score", 0)
    
    sections = []
    for req in requirements[:5]:
        caps = retrieved_caps.get(req, [])
        if caps:
            content = "We have relevant experience:\n" + "\n".join([f"- {c}" for c in caps[:3]])
        else:
            content = "We will develop the necessary capabilities."
        
        sections.append({
            "section_title": f"Response to: {req[:60]}",
            "content": content,
            "mapped_requirements": [req]
        })
    
    state["proposal"] = {
        "executive_summary": f"We are responding with {len(requirements)} requirements addressed. Match score: {avg_score:.1f}%",
        "company_overview": "Our company has demonstrated capabilities in relevant areas.",
        "response_sections": sections,
        "conclusion": "We are confident in our ability to deliver."
    }
    
    print(f"   ✓ Generated proposal with {len(sections)} sections")
    return state


def calculate_win_score(state: RFPState) -> RFPState:
    print("\n📊 Step: Calculating win score...")
    
    scorer = WinProbabilityScorer(verbose=True)
    
    win_analysis = scorer.calculate(
        rfp_text=state.get("rfp_text", ""),
        requirements=state.get("mandatory_requirements", []),
        matched_capabilities=state.get("retrieved_capabilities", {}),
        match_scores={m["requirement"]: m["score"] for m in state.get("match_results", {}).get("matches", [])}
    )
    
    state["final_score"] = win_analysis["total_score"]
    state["recommendation"] = win_analysis["recommendation"]
    state["win_analysis"] = win_analysis
    
    return state


# ============== Create Workflow ==============

def create_rfp_workflow():
    workflow = StateGraph(RFPState)
    
    workflow.add_node("parse_rfp", parse_rfp)
    workflow.add_node("retrieve_from_rag", retrieve_from_rag)
    workflow.add_node("match_requirements", match_requirements)
    workflow.add_node("generate_ml_features", generate_ml_features)
    workflow.add_node("predict_with_ml", predict_with_ml)
    workflow.add_node("generate_proposal", generate_proposal)
    workflow.add_node("calculate_win_score", calculate_win_score)
    
    workflow.set_entry_point("parse_rfp")
    workflow.add_edge("parse_rfp", "retrieve_from_rag")
    workflow.add_edge("retrieve_from_rag", "match_requirements")
    workflow.add_edge("match_requirements", "generate_ml_features")
    workflow.add_edge("generate_ml_features", "predict_with_ml")
    workflow.add_edge("predict_with_ml", "generate_proposal")
    workflow.add_edge("generate_proposal", "calculate_win_score")
    workflow.add_edge("calculate_win_score", END)
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# Create agent instance
agent = create_rfp_workflow()
print("\n✅ RFP Agent initialized with utils!")