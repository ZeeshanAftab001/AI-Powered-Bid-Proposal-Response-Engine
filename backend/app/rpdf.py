from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, Any, List
import requests
import uuid
from app.utils import extract_text
from app.agent import agent
from app.workspace import WorkspaceManager

rpdf_router = APIRouter(prefix="/api", tags=["Process File"])


class CloudinaryFileRequest(BaseModel):
    cloudinary_url: HttpUrl
    rfp_id: Optional[str] = None
    bid_manager_level: Optional[int] = 3
    metadata: Optional[Dict[str, Any]] = None


# ============== Response Models ==============

class MatchEvidence(BaseModel):
    requirement: str
    score: int
    evidence: List[str]
    reasoning: str


class MatchResultsResponse(BaseModel):
    matches: List[MatchEvidence]
    average_score: float


class ProposalSectionResponse(BaseModel):
    section_title: str
    content: str
    mapped_requirements: List[str]


class ProposalResponse(BaseModel):
    executive_summary: str
    company_overview: str
    response_sections: List[ProposalSectionResponse]
    conclusion: str


class MLFeaturesResponse(BaseModel):
    sector: str
    budget: float
    compliance: float
    score: float
    response_time: int
    doc_pages: int
    gaps_found: int
    bid_manager: int
    compliance_score_norm: float
    budget_norm: float
    efficiency_score: float
    pages_per_gap: float
    compliance_budget_ratio: float


class MLPredictionResponse(BaseModel):
    outcome: str
    win_probability: float
    prediction_class: int
    method: str


class ProcessResponse(BaseModel):
    status: str
    result: Dict[str, Any]  # This will contain the complete agent output


@rpdf_router.post("/process-cloudinary-file")
async def process_cloudinary_file(request: CloudinaryFileRequest):
    """
    Receive a Cloudinary file link, download and process the document
    Returns COMPLETE agent output including all analysis
    """
    try:
        # Step 1: Download file from Cloudinary
        print(f"\n📥 Downloading from Cloudinary: {request.cloudinary_url}")
        response = requests.get(str(request.cloudinary_url), timeout=30)
        response.raise_for_status()
        file_bytes = response.content
        
        # Step 2: Detect file type from URL
        url_str = str(request.cloudinary_url).lower()
        
        # Step 3: Extract text based on file type
        extracted_text = extract_text(file_bytes, url_str)
        
        if not extracted_text:
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file type or empty content. Use PDF, DOCX, or TXT"
            )
        
        print(f"   ✓ Extracted {len(extracted_text)} characters")
        
        # Step 4: Generate or use provided RFP ID
        rfp_id = request.rfp_id or f"rfp_{uuid.uuid4().hex[:8]}"
        
        # Step 5: Create workspace for this RFP
        workspace_id = WorkspaceManager.create_workspace(
            rfp_id=rfp_id,
            rfp_text=extracted_text,
            metadata=request.metadata
        )
        
        # Step 6: Prepare initial state for agent
        initial_state = {
            "rfp_text": extracted_text,
            "deadline": "",
            "budget": "",
            "mandatory_requirements": [],
            "evaluation_criteria": [],
            "retrieved_capabilities": {},
            "company_capabilities": [],
            "match_results": {},
            "proposal": {},
            "final_score": 0,
            "recommendation": "",
            "ml_features": {},
            "feature_vector": [],
            "ml_prediction": {},
            "workspace_id": workspace_id,
            "bid_manager_level": request.bid_manager_level
        }
        
        # Step 7: Run agent with thread isolation
        print(f"\n🤖 Running agent for RFP: {rfp_id}")
        config = {"configurable": {"thread_id": f"workspace_{rfp_id}"}}
        results = agent.invoke(initial_state, config=config)
        
        # Step 8: Update workspace with results
        WorkspaceManager.update_workspace(rfp_id, results)
        
        # Step 9: Format the response exactly like your JSON
        response_data = {
            "status": "success",
            "result": {
                "rfp_text": results.get("rfp_text", ""),
                "deadline": results.get("deadline", "Not specified"),
                "budget": results.get("budget", "Not specified"),
                "mandatory_requirements": results.get("mandatory_requirements", []),
                "evaluation_criteria": results.get("evaluation_criteria", []),
                "retrieved_capabilities": results.get("retrieved_capabilities", {}),
                "company_capabilities": results.get("company_capabilities", []),
                "match_results": results.get("match_results", {}),
                "proposal": results.get("proposal", {}),
                "final_score": results.get("final_score", 0),
                "recommendation": results.get("recommendation", "Unknown"),
                "ml_features": results.get("ml_features", {}),
                "feature_vector": results.get("feature_vector", []),
                "ml_prediction": results.get("ml_prediction", {})
            }
        }
        
        print(f"\n✅ Processing complete!")
        print(f"   📊 Match Score: {results.get('match_results', {}).get('average_score', 0):.1f}%")
        print(f"   🏆 Win Score: {results.get('final_score', 0):.1f}%")
        print(f"   💡 Recommendation: {results.get('recommendation', 'Unknown')}")
        
        return response_data
        
    except requests.RequestException as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Failed to download from Cloudinary: {str(e)}"
        )
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400, 
            detail="Failed to decode file. Check file encoding (use UTF-8 for TXT files)"
        )
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Processing failed: {str(e)}"
        )


@rpdf_router.get("/workspace/{rfp_id}")
async def get_workspace_status(rfp_id: str):
    """Get the complete workspace state for an RFP"""
    
    from app.workspace import get_workspace_state
    
    state = get_workspace_state(rfp_id)
    
    if not state:
        raise HTTPException(status_code=404, detail=f"Workspace {rfp_id} not found")
    
    return {
        "status": "success",
        "result": {
            "rfp_text": state.get("rfp_text", ""),
            "deadline": state.get("deadline", "Not specified"),
            "budget": state.get("budget", "Not specified"),
            "mandatory_requirements": state.get("mandatory_requirements", []),
            "evaluation_criteria": state.get("evaluation_criteria", []),
            "retrieved_capabilities": state.get("retrieved_capabilities", {}),
            "company_capabilities": state.get("company_capabilities", []),
            "match_results": state.get("match_results", {}),
            "proposal": state.get("proposal", {}),
            "final_score": state.get("final_score", 0),
            "recommendation": state.get("recommendation", "Unknown"),
            "ml_features": state.get("ml_features", {}),
            "feature_vector": state.get("feature_vector", []),
            "ml_prediction": state.get("ml_prediction", {})
        }
    }


@rpdf_router.get("/workspaces")
async def list_all_workspaces():
    """List all processed workspaces"""
    
    from app.workspace import get_all_workspaces
    
    workspaces = get_all_workspaces()
    
    return {
        "total": len(workspaces),
        "workspaces": workspaces
    }


@rpdf_router.delete("/workspace/{rfp_id}")
async def delete_workspace(rfp_id: str):
    """Delete a workspace and its data"""
    
    from app.workspace import WorkspaceManager
    
    if WorkspaceManager.delete_workspace(rfp_id):
        return {"message": f"Workspace {rfp_id} deleted successfully", "rfp_id": rfp_id}
    
    raise HTTPException(status_code=404, detail=f"Workspace {rfp_id} not found")