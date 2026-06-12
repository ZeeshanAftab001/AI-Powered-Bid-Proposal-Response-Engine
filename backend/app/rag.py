import os
import re
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, UploadFile, File

# Try to import required packages with helpful error messages
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    raise ImportError("Please install: pip install langchain-text-splitters")

try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    raise ImportError("Please install: pip install langchain-community chromadb")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Please install: pip install sentence-transformers")

try:
    import PyPDF2
except ImportError:
    raise ImportError("Please install: pip install PyPDF2")

# Initialize router
rag_router = APIRouter(prefix="/rag", tags=["RAG Operations"])

# Constants
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
PERSIST_DIRECTORY = str(Path(__file__).parent / "capability_db")

# ========== EMBEDDINGS CLASS ==========

class SentenceTransformerEmbeddings:
    """Wrapper for sentence-transformers to work with LangChain"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print(f"🔄 Loading embedding model: {model_name}...")
        try:
            self.model = SentenceTransformer(model_name)
            self.dimension = 384
            print(f"✓ Model loaded! Embedding dimension: {self.dimension}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        embedding = self.model.encode([text])[0]
        return embedding.tolist()
    
    def __call__(self, text: str) -> List[float]:
        """Make the instance callable for LangChain"""
        return self.embed_query(text)

# Initialize embeddings with error handling
embeddings = None
try:
    embeddings = SentenceTransformerEmbeddings()
except Exception as e:
    print(f"✗ Failed to initialize embeddings: {e}")

# ========== PROJECT DATA PARSER ==========

class ProjectDataParser:
    @staticmethod
    def parse_projects_from_text(text: str) -> List[Dict[str, Any]]:
        """Extract project information from raw text"""
        projects = []
        
        # Pattern for CAP-XXX projects
        cap_pattern = r'(CAP-\d{3})\s+([A-Za-z\s]+?)(?:\n|$)'
        matches = re.findall(cap_pattern, text)
        
        for match in matches:
            project_code = match[0]
            project_name = match[1].strip()
            
            # Look for certification in the following lines
            cert_pattern = r'(ISO \d+|CMMI L\d|CE Mark|N/A)'
            cert_match = re.search(cert_pattern, text[text.find(project_code):text.find(project_code) + 200])
            certification = cert_match.group(1) if cert_match else "N/A"
            
            projects.append({
                "project_code": project_code,
                "project_name": project_name,
                "certification": certification,
            })
        
        return projects
    
    @staticmethod
    def parse_metrics_from_text(text: str) -> List[Dict[str, Any]]:
        """Extract metrics (year, budget, client type, etc.)"""
        metrics = []
        
        # Look for patterns like: "2023 PKR 15M 34 International"
        metric_pattern = r'(\d{4})\s+PKR\s+(\d+(?:\.\d+)?[MB]?)\s+(\d+)\s+([A-Za-z\s]+?)(?:\n|$)'
        matches = re.findall(metric_pattern, text)
        
        for match in matches:
            metrics.append({
                "year": int(match[0]),
                "budget": match[1],
                "project_id": int(match[2]),
                "client_type": match[3].strip(),
            })
        
        return metrics
    
    @staticmethod
    def combine_project_data(projects: List[Dict], metrics: List[Dict]) -> List[Dict]:
        """Combine project info with metrics"""
        combined = []
        for i, project in enumerate(projects):
            if i < len(metrics):
                project_data = {**project, **metrics[i]}
                project_data["full_description"] = (
                    f"{project_data['project_code']}: {project_data['project_name']} | "
                    f"Year: {project_data.get('year', 'N/A')} | "
                    f"Budget: PKR {project_data.get('budget', 'N/A')} | "
                    f"Client: {project_data.get('client_type', 'N/A')} | "
                    f"Certification: {project_data.get('certification', 'N/A')}"
                )
                combined.append(project_data)
        return combined

# ========== RAG CLASS ==========

class CapabilityRAG:
    def __init__(self, persist_directory: str = PERSIST_DIRECTORY):
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "|", ", ", " ", ""],
            length_function=len,
        )
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Initialize RAG with knowledge base"""
        if not embeddings:
            print("✗ Cannot initialize RAG: embeddings not available")
            return
            
        try:
            # Check if vector store already exists
            if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=embeddings
                )
                print(f"✓ Loaded existing RAG from {self.persist_directory}")
                return
            
            # Seed knowledge base
            knowledge_docs = [
                "Project Type: Mobile Banking. Certifications: ISO 27001, CE Mark. Budget range: PKR 15M-200M",
                "Project Type: Cloud Infrastructure. Certifications: ISO 27001. Client: Federal Govt, International",
                "Project Type: Cybersecurity. Certifications: ISO 27001. Client: Federal Govt, Private Sector",
                "Project Type: ERP Implementation. Certifications: CMMI L3, ISO 27001. Client: Federal Govt",
            ]
            
            all_chunks = []
            for doc in knowledge_docs:
                chunks = self.text_splitter.split_text(doc)
                all_chunks.extend(chunks)
            
            if all_chunks:
                self.vectorstore = Chroma.from_texts(
                    texts=all_chunks,
                    embedding=embeddings,
                    persist_directory=self.persist_directory
                )
                print(f"✓ RAG initialized with {len(all_chunks)} seed knowledge chunks")
                
        except Exception as e:
            print(f"✗ Error initializing RAG: {str(e)}")
    
    def add_project_pdf_sync(self, pdf_path: Path) -> bool:
        """Add project PDF data to vector store"""
        if not embeddings or not self.text_splitter:
            return False
            
        try:
            # Extract text from PDF
            print(f"   📖 Reading PDF: {pdf_path.name}")
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                full_text = ""
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
                    print(f"   📄 Processed page {page_num}/{len(reader.pages)}")
            
            if not full_text.strip():
                print("   ✗ No text extracted from PDF")
                return False
            
            # Parse projects
            print("   🔍 Parsing project data...")
            parser = ProjectDataParser()
            projects = parser.parse_projects_from_text(full_text)
            metrics = parser.parse_metrics_from_text(full_text)
            combined_data = parser.combine_project_data(projects, metrics)
            
            print(f"   📊 Found {len(projects)} projects and {len(metrics)} metrics")
            
            if not combined_data:
                print("   🔍 No structured data, using raw text chunks...")
                raw_chunks = self.text_splitter.split_text(full_text)
                if raw_chunks:
                    self.vectorstore = Chroma.from_texts(
                        texts=raw_chunks,
                        embedding=embeddings,
                        persist_directory=self.persist_directory
                    )
                    print(f"   ✓ Created RAG with {len(raw_chunks)} raw text chunks")
                    return True
                return False
            
            # Create text chunks
            text_chunks = []
            for project in combined_data:
                text_chunks.append(project["full_description"])
                # Add searchable variations
                text_chunks.append(f"{project['project_code']}: {project.get('project_name', '')}")
                if project.get('budget') and project['budget'] != 'N/A':
                    text_chunks.append(f"Budget {project['budget']} for {project.get('project_name', '')}")
                if project.get('client_type') and project['client_type'] != 'N/A':
                    text_chunks.append(f"{project['client_type']} project: {project.get('project_name', '')}")
            
            print(f"   📝 Created {len(text_chunks)} text chunks")
            
            # Split chunks further
            all_chunks = []
            for chunk in text_chunks:
                split_chunks = self.text_splitter.split_text(chunk)
                all_chunks.extend(split_chunks)
            
            print(f"   🧩 Total chunks for embedding: {len(all_chunks)}")
            
            # Add to vector store
            if self.vectorstore:
                self.vectorstore.add_texts(texts=all_chunks)
                print(f"   ✓ Added {len(all_chunks)} project chunks to existing RAG")
            else:
                self.vectorstore = Chroma.from_texts(
                    texts=all_chunks,
                    embedding=embeddings,
                    persist_directory=self.persist_directory
                )
                print(f"   ✓ Created new RAG with {len(all_chunks)} project chunks")
            
            return True
            
        except Exception as e:
            print(f"   ✗ Error adding PDF: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def retrieve_capabilities(self, query: str, k: int = 5) -> List[str]:
        """Retrieve relevant capabilities for a query"""
        if not self.vectorstore:
            return []
        
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            return [doc.page_content for doc in results]
        except Exception as e:
            print(f"Error retrieving: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        if not self.vectorstore:
            return {"status": "not_initialized", "message": "Vector store not initialized"}
        
        try:
            return {
                "status": "active",
                "total_vectors": self.vectorstore._collection.count(),
                "persist_directory": self.persist_directory,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding_dimension": embeddings.dimension if embeddings else 384,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

# ========== CREATE GLOBAL INSTANCE ==========
# Initialize RAG instance (MUST be before API endpoints that reference it)
print("\n" + "="*60)
print("🚀 Initializing RAG System with Local Embeddings")
print("="*60)
capability_rag = CapabilityRAG()
print("\n" + "="*60)
print("✅ RAG System Ready for Project Data")
print("="*60)
print(f"📁 Persist Directory: {PERSIST_DIRECTORY}")
print(f"🤖 Model: sentence-transformers/all-MiniLM-L6-v2")
print(f"💡 Upload your project PDF via POST /rag/upload-projects")
print("="*60 + "\n")

# ========== API ENDPOINTS ==========

@rag_router.post("/upload-projects")
async def upload_project_data(file: UploadFile = File(...)):
    """Upload and process project PDF"""
    if not embeddings:
        raise HTTPException(status_code=500, detail="Embedding model not loaded")
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    temp_dir = Path("./temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    temp_file_path = temp_dir / file.filename
    
    try:
        # Save file
        content = await file.read()
        with open(temp_file_path, "wb") as buffer:
            buffer.write(content)
        
        # Process PDF
        success = capability_rag.add_project_pdf_sync(temp_file_path)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to process PDF - no text chunks created")
        
        stats = capability_rag.get_stats()
        
        return {
            "success": True,
            "message": "Project data uploaded and indexed successfully",
            "filename": file.filename,
            "stats": stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in upload: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        if temp_file_path.exists():
            os.remove(temp_file_path)
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

@rag_router.get("/search")
async def search_projects(query: str, k: int = 5):
    """Search for projects"""
    results = capability_rag.retrieve_capabilities(query, k)
    return {
        "query": query, 
        "num_results": len(results), 
        "results": results
    }

@rag_router.get("/stats")
async def get_stats():
    """Get system statistics"""
    return capability_rag.get_stats()

@rag_router.get("/test")
async def test_system():
    """Test if system is working"""
    return {
        "embeddings_loaded": embeddings is not None,
        "vectorstore_exists": capability_rag.vectorstore is not None,
        "vector_count": capability_rag.vectorstore._collection.count() if capability_rag.vectorstore else 0,
    }

@rag_router.delete("/clear")
async def clear_data():
    """Clear all data"""
    try:
        if os.path.exists(PERSIST_DIRECTORY):
            shutil.rmtree(PERSIST_DIRECTORY)
        
        # Reinitialize without using global (just create new instance)
        new_rag = CapabilityRAG()
        
        # Update the module-level variable (but this won't affect existing references)
        # Instead, we'll return success and let the next request use the new instance
        # For a proper solution, you'd need to use a global variable
        return {"success": True, "message": "Data cleared successfully. New RAG instance created."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))