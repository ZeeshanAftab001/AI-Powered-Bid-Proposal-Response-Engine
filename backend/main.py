from fastapi import FastAPI
from app.rpdf import rpdf_router
from app.rag import rag_router
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="AI Proposal Response Engine",
    description="Automated RFP/RFQ/Tender analysis and proposal generation",
    version="1.0.0"
)
#uv run python -m uvicorn main:app --reload
# Enable CORS for frontend (React, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rpdf_router)
app.include_router(rag_router)


@app.get("/")
async def root():
    return {
        "message": "RFP Agent API",
        "version": "2.0.0",
        "endpoints": {
            "process": "POST /api/process-file",
            "workspace": "GET /api/workspace/{rfp_id}",
            "workspaces": "GET /api/workspaces",
            "delete": "DELETE /api/workspace/{rfp_id}"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "RFP Agent"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)