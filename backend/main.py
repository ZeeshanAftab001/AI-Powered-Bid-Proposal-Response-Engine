from fastapi import FastAPI
from app.rpdf import rpdf_router
from app.rag import rag_router
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="AI Proposal Response Engine",
    description="Automated RFP/RFQ/Tender analysis and proposal generation",
    version="1.0.0"
)
#uv run python -m uvicorn app.main:app --reload

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
