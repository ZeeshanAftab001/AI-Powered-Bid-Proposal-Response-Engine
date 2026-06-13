"""
Main LangGraph Agent - FULLY FIXED VERSION
- No flat 75 scores
- Workspace separation ready
- Proper sector detection
"""

import os
from typing import List, Optional, TypedDict, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Import modules
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
    create_feature_vector
)
from app.matcher import match_requirements
from app.win_probability import WinProbabilityScorer
from app.ml_predictor import get_ml_predictor
from app.rag import capability_rag
from app.workspace import WorkspaceManager

load_dotenv()

# ============== Pydantic Models ==============

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

class GeneratedProposal(BaseModel):
    executive_summary: str = Field(description="Executive summary")
    company_overview: str = Field(description="Company capabilities overview")
    response_sections: List[Dict] = Field(description="Detailed responses")
    conclusion: str = Field(description="Conclusion")

# ============== Initialize LLM ==============

try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    print("✓ Gemini model initialized")
except Exception as e:
    print(f"⚠️ Gemini not available: {e}")
    llm = None

structured_llm_parser = llm.with_structured_output(ParsedRFP) if llm else None
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
    workspace_id: str

# ============== Node Functions ==============

def parse_rfp(state: RFPState) -> RFPState:
    """Parse RFP document with structured output"""
    
    print("\n📄 Step 1: Parsing RFP document...")
    
    if not structured_llm_parser:
        # Fallback parsing
        text = state.get("rfp_text", "")
        state["deadline"] = "Not specified"
        state["budget"] = "Not specified"
        state["mandatory_requirements"] = []
        state["evaluation_criteria"] = []
        return state
    
    prompt = ChatPromptTemplate.from_template("""
    You are an expert RFP Analyst. Parse the following RFP document and extract structured data.
    
    RFP TEXT:
    {text}
    
    INSTRUCTIONS:
    1. DEADLINE: Look for 'Submission Deadline' or 'Due Date'. Extract the full date/time string.
    2. BUDGET: Look for 'Budget Range' or 'Estimated Value'. Specifically look for PKR or USD amounts.
    3. MANDATORY REQUIREMENTS: These are often in Section 2 or a table with headers like 'ID', 'Requirement', 'Evidence'.
       - Look for IDs like M1, M2, M3...
       - Extract every single requirement that is marked as 'Mandatory' or 'Non-compliance = automatic disqualification'.
    4. EVALUATION CRITERIA: Look for Section 3 or 'Weightage' tables. Extract the main criteria names and weights.
    
    Return the data in the requested structured format.
    """)
    
    chain = prompt | structured_llm_parser
    
    try:
        result = chain.invoke({"text": state.get("rfp_text", "")[:4000]})
        
        state["deadline"] = result.deadline or "Not specified"
        state["budget"] = result.budget or "Not specified"
        state["mandatory_requirements"] = result.mandatory_requirements
        state["evaluation_criteria"] = result.evaluation_criteria
        
        print(f"   ✓ Found {len(result.mandatory_requirements)} mandatory requirements")
        print(f"   ✓ Deadline: {state['deadline']}")
        print(f"   ✓ Budget: {state['budget']}")
        
        return state
    except Exception as e:
        print(f"   ✗ Error parsing RFP: {e}")
        return state


def retrieve_from_rag(state: RFPState) -> RFPState:
    """Retrieve relevant capabilities from RAG for each requirement"""
    
    print("\n🔍 Step 2: Retrieving capabilities from RAG...")
    
    requirements = state.get("mandatory_requirements", [])
    
    if not requirements:
        print("   ✗ No requirements to retrieve")
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
    
    print(f"\n   📚 Total unique capabilities retrieved: {len(state['company_capabilities'])}")
    
    return state


def match_requirements_node(state: RFPState) -> RFPState:
    """Match requirements using RAG capabilities - FIXED: no flat 75 scores"""
    
    print("\n🎯 Step 3: Matching requirements with RAG capabilities...")
    
    requirements = state.get("mandatory_requirements", [])
    retrieved_caps = state.get("retrieved_capabilities", {})
    
    # Use the fixed matcher that calculates real scores
    match_results = match_requirements(requirements, retrieved_caps)
    
    state["match_results"] = match_results
    
    print(f"\n   ✓ Average match score: {match_results['average_score']:.1f}%")
    print(f"   ✓ Score distribution: {[m['score'] for m in match_results['matches'][:5]]}")
    
    return state


def generate_ml_features(state: RFPState) -> RFPState:
    """Generate all 13 ML features - FIXED sector detection"""
    
    print("\n📊 Step 4: Generating ML Features...")
    
    rfp_text = state.get("rfp_text", "")
    deadline = state.get("deadline", "Not specified")
    budget_str = state.get("budget", "Not specified")
    match_results = state.get("match_results", {})
    matches = match_results.get("matches", [])
    avg_match_score = match_results.get("average_score", 50.0)
    
    # Extract features using improved functions
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
    
    print(f"   ✓ Sector detected: {sector}")
    print(f"   ✓ Budget: ${budget:,.0f}")
    print(f"   ✓ Compliance: {compliance:.1f}%")
    print(f"   ✓ Match Score: {score:.1f}%")
    print(f"   ✓ Gaps Found: {gaps_found}")
    print(f"   ✓ All 13 ML features generated")
    
    return state


def predict_with_ml(state: RFPState) -> RFPState:
    """Make ML prediction using utils"""
    
    print("\n🎯 Step 5: Running ML Prediction...")
    
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
    print(f"   ✓ Method: {prediction.get('method', 'unknown')}")
    
    return state


def generate_proposal(state: RFPState) -> RFPState:
    """Generate proposal using RAG-retrieved capabilities"""
    
    print("\n✍️ Step 6: Generating proposal...")
    
    requirements = state.get("mandatory_requirements", [])
    retrieved_caps = state.get("retrieved_capabilities", {})
    avg_score = state.get("match_results", {}).get("average_score", 0)
    
    sections = []
    for req in requirements[:5]:
        caps = retrieved_caps.get(req, [])
        if caps:
            content = "We have relevant experience:\n" + "\n".join([f"- {c}" for c in caps[:3]])
        else:
            content = "We will develop the necessary capabilities to meet this requirement."
        
        sections.append({
            "section_title": f"Response to: {req[:60]}",
            "content": content,
            "mapped_requirements": [req]
        })
    
    state["proposal"] = {
        "executive_summary": f"We are responding to this RFP with strong capabilities. Our overall match score is {avg_score:.1f}%, demonstrating our readiness.",
        "company_overview": "Our company has demonstrated capabilities as documented in our capability database.",
        "response_sections": sections,
        "conclusion": "We are confident in our ability to deliver this project successfully."
    }
    
    print(f"   ✓ Generated proposal with {len(sections)} sections")
    return state


def calculate_win_score(state: RFPState) -> RFPState:
    """Calculate final win probability using comprehensive scorer"""
    
    print("\n📊 Step 7: Calculating win score...")
    
    scorer = WinProbabilityScorer(verbose=True)
    
    match_scores = {}
    for m in state.get("match_results", {}).get("matches", []):
        match_scores[m["requirement"]] = m["score"]
    
    win_analysis = scorer.calculate(
        rfp_text=state.get("rfp_text", ""),
        requirements=state.get("mandatory_requirements", []),
        matched_capabilities=state.get("retrieved_capabilities", {}),
        match_scores=match_scores
    )
    
    state["final_score"] = win_analysis["total_score"]
    state["recommendation"] = win_analysis["recommendation"]
    state["win_analysis"] = win_analysis
    
    print(f"   ✓ Final Win Score: {win_analysis['total_score']:.1f}/100")
    print(f"   ✓ Recommendation: {win_analysis['recommendation']}")
    
    return state


# ============== Create Workflow ==============

def create_rfp_workflow():
    """Create LangGraph workflow with all nodes"""
    
    workflow = StateGraph(RFPState)
    
    # Add nodes
    workflow.add_node("parse_rfp", parse_rfp)
    workflow.add_node("retrieve_from_rag", retrieve_from_rag)
    workflow.add_node("match_requirements", match_requirements_node)
    workflow.add_node("generate_ml_features", generate_ml_features)
    workflow.add_node("predict_with_ml", predict_with_ml)
    workflow.add_node("generate_proposal", generate_proposal)
    workflow.add_node("calculate_win_score", calculate_win_score)
    
    # Add edges
    workflow.set_entry_point("parse_rfp")
    workflow.add_edge("parse_rfp", "retrieve_from_rag")
    workflow.add_edge("retrieve_from_rag", "match_requirements")
    workflow.add_edge("match_requirements", "generate_ml_features")
    workflow.add_edge("generate_ml_features", "predict_with_ml")
    workflow.add_edge("predict_with_ml", "generate_proposal")
    workflow.add_edge("generate_proposal", "calculate_win_score")
    workflow.add_edge("calculate_win_score", END)
    
    return workflow.compile(checkpointer=WorkspaceManager.get_memory_saver())


# Create agent instance
agent = create_rfp_workflow()
print("\n✅ RFP Agent initialized successfully!")
print("   - RAG integration: Active")
print("   - Fixed: Real match scores (no flat 75)")
print("   - Fixed: Better sector detection")
print("   - Workspace separation: Ready")