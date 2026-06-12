from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, Any
import requests
from app.utils import extract_text
from app.agent import agent


rpdf_router = APIRouter(prefix="/api")

class CloudinaryFileRequest(BaseModel):
    cloudinary_url: HttpUrl
    rfp_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@rpdf_router.post("/process-cloudinary-file")
async def process_cloudinary_file(request: CloudinaryFileRequest):
    """
    Receive a Cloudinary file link, download and process the document
    Supports PDF, DOCX, and TXT files
    """
    try:
        # Download file from Cloudinary
        response = requests.get(str(request.cloudinary_url), timeout=30)
        response.raise_for_status()
        file_bytes = response.content
        
        # Detect file type from URL
        url_str = str(request.cloudinary_url).lower()
        
        # Extract text based on file type
        extracted_text = extract_text(file_bytes,url_str)
        if not extracted_text : 
            raise HTTPException(
                    status_code=400, 
                    detail="Unsupported file type. Use PDF, DOCX, or TXT"
                )
        print(extracted_text)
            # Method 1: Pass config as separate argument
        results = agent.invoke(
            {"rfp_text": extracted_text},
            config={"configurable": {"thread_id": "2"}}
        )

        return {
            "status": "success",
            "result": results
        }
        
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download from Cloudinary: {str(e)}")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Failed to decode TXT file. Check file encoding.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")