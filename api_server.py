"""
FastAPI Backend for Advanced LLM-Powered Query-Retrieval System
Enterprise-grade API with optimized performance and comprehensive documentation
"""

import asyncio
import time
import uuid
import os
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import logging
from contextlib import asynccontextmanager

# FastAPI and related
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

# Pydantic models
from pydantic import BaseModel, Field, field_validator
from enum import Enum

# Custom imports
from advanced_rag_system import (
    AdvancedRAGSystem, 
    QueryRequest, 
    QueryResponse,
    DocumentType,
    QueryType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for tracking
active_sessions = {}
request_count = 0
total_response_time = 0.0
app_start_time = time.time()

# Initialize RAG system
rag_system = AdvancedRAGSystem(
    model_name="gemini-1.5-flash",
    vector_store_type="faiss",
    embedding_model="all-MiniLM-L6-v2"
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting Advanced Query-Retrieval System")
    try:
        # Initialize core components
        rag_system.initialize_llm()
        rag_system.embedding_manager.initialize()
        logger.info("✅ System initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize system: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down system")
    active_sessions.clear()

# Initialize FastAPI app
app = FastAPI(
    title="LLM-Powered Intelligent Query-Retrieval System",
    description="""
    Advanced enterprise-grade system for processing documents and answering queries
    across insurance, legal, HR, and compliance domains.
    
    **Key Features:**
    - Multi-format document processing (PDF, DOCX, Email)
    - Advanced RAG with semantic search
    - Intelligent query classification
    - Real-time clause matching
    - Explainable AI decisions
    - Token-optimized processing
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
VALID_TOKENS = {
    "hackrx-2024-bajaj-finserv": "hackrx_token",
    "03e834fbe3091bbd057af6a74fc056c509bf3a5a5a730b9e628d357d675b22a5": "team_token"
}

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify authentication token"""
    if credentials.credentials not in VALID_TOKENS:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Enhanced Models
class DocumentFormat(str, Enum):
    """Supported document formats"""
    PDF = "pdf"
    DOCX = "docx"
    EMAIL = "email"
    URL = "url"
    AUTO = "auto"

class ProcessingOptions(BaseModel):
    """Advanced processing options"""
    chunk_size: int = Field(default=1000, ge=500, le=2000)
    chunk_overlap: int = Field(default=200, ge=50, le=500)
    top_k_retrieval: int = Field(default=7, ge=3, le=15)
    include_metadata: bool = Field(default=True)
    optimize_for_speed: bool = Field(default=False)
    enable_caching: bool = Field(default=True)

class EnhancedQueryRequest(BaseModel):
    """Enhanced query request model"""
    documents: Union[str, List[str]] = Field(..., description="Document URL(s) or paths")
    questions: List[str] = Field(..., min_items=1, max_items=50, description="Questions to answer")
    document_format: DocumentFormat = Field(default=DocumentFormat.AUTO)
    processing_options: ProcessingOptions = Field(default_factory=ProcessingOptions)
    session_id: Optional[str] = Field(default=None, description="Session ID for tracking")
    
    @field_validator('questions')
    @classmethod
    def validate_questions(cls, v):
        if not v:
            raise ValueError('At least one question is required')
        for question in v:
            if not question.strip():
                raise ValueError('Questions cannot be empty')
        return v
    
    @field_validator('documents')
    @classmethod
    def validate_documents(cls, v):
        # Convert single document to list for internal processing
        if isinstance(v, str):
            return [v]
        return v

class DetailedAnswer(BaseModel):
    """Detailed answer with comprehensive metadata"""
    question: str
    answer: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    query_type: str
    source_citations: List[str]
    processing_time: float
    context_chunks_used: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EnhancedQueryResponse(BaseModel):
    """Enhanced response model with detailed analytics"""
    session_id: str
    answers: List[DetailedAnswer]
    total_processing_time: float
    total_token_usage: Dict[str, int]
    document_statistics: Dict[str, Any]
    performance_metrics: Dict[str, float]
    status: str = "success"
    timestamp: str

class SystemHealth(BaseModel):
    """System health status"""
    status: str
    uptime: float
    memory_usage: Dict[str, float]
    active_sessions: int
    total_requests: int
    average_response_time: float

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    details: str
    timestamp: str
    session_id: Optional[str] = None

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Advanced LLM-Powered Intelligent Query-Retrieval System",
        "version": "2.0.0",
        "status": "active",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=SystemHealth)
async def health_check():
    """Comprehensive health check endpoint"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    uptime = time.time() - app_start_time
    avg_response_time = total_response_time / max(request_count, 1)
    
    return SystemHealth(
        status="healthy",
        uptime=uptime,
        memory_usage={
            "rss": memory_info.rss / 1024 / 1024,  # MB
            "vms": memory_info.vms / 1024 / 1024,  # MB
            "cpu_percent": process.cpu_percent()
        },
        active_sessions=len(active_sessions),
        total_requests=request_count,
        average_response_time=avg_response_time
    )

@app.get("/api/v1/health", response_model=SystemHealth)
async def health_check_v1():
    """Health check endpoint for API v1"""
    return await health_check()

@app.post("/api/v1/hackrx/run")
async def process_hackrx_request(
    request: EnhancedQueryRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """
    Main endpoint for HackRX submission
    Processes documents and answers questions - returns simple format for hackathon
    """
    global request_count, total_response_time
    
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        logger.info(f"Processing HackRX request: {session_id}")
        request_count += 1
        
        # Store session
        active_sessions[session_id] = {
            "start_time": start_time,
            "status": "processing",
            "question_count": len(request.questions)
        }
        
        # Convert to internal format
        query_request = QueryRequest(
            documents=request.documents,
            questions=request.questions,
            document_type=request.document_format.value,
            options=request.processing_options.dict()
        )
        
        # Process request
        response = await rag_system.process_query_request(query_request)
        
        processing_time = time.time() - start_time
        total_response_time += processing_time
        
        # Update session
        active_sessions[session_id]["status"] = "completed"
        active_sessions[session_id]["processing_time"] = processing_time
        
        # Return simple format for hackathon compatibility
        simple_response = {
            "answers": response.answers
        }
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_session, session_id)
        
        logger.info(f"Successfully processed request {session_id} in {processing_time:.2f}s")
        return simple_response
        
    except Exception as e:
        logger.error(f"Error processing request {session_id}: {str(e)}")
        
        # Update session with error
        if session_id in active_sessions:
            active_sessions[session_id]["status"] = "error"
            active_sessions[session_id]["error"] = str(e)
        
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="Processing failed",
                details=str(e),
                timestamp=datetime.now().isoformat(),
                session_id=session_id
            ).dict()
        )

@app.post("/api/v1/query", response_model=EnhancedQueryResponse)
async def process_query(
    request: EnhancedQueryRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """
    Standard query processing endpoint with detailed response
    Returns full EnhancedQueryResponse for frontend compatibility
    """
    global request_count, total_response_time
    
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        logger.info(f"Processing detailed query request: {session_id}")
        request_count += 1
        
        # Store session
        active_sessions[session_id] = {
            "start_time": start_time,
            "status": "processing",
            "question_count": len(request.questions)
        }
        
        # Convert to internal format
        query_request = QueryRequest(
            documents=request.documents,
            questions=request.questions,
            document_type=request.document_format.value,
            options=request.processing_options.dict()
        )
        
        # Process request
        response = await rag_system.process_query_request(query_request)
        
        # Create detailed answers
        detailed_answers = []
        for i, question in enumerate(request.questions):
            detailed_answer = DetailedAnswer(
                question=question,
                answer=response.answers[i],
                confidence_score=response.confidence_scores[i],
                query_type="auto-detected",
                source_citations=response.source_citations[i],
                processing_time=response.processing_time / len(request.questions),
                context_chunks_used=7,
                metadata={
                    "question_index": i,
                    "word_count": len(response.answers[i].split()),
                    "character_count": len(response.answers[i])
                }
            )
            detailed_answers.append(detailed_answer)
        
        processing_time = time.time() - start_time
        total_response_time += processing_time
        
        # Update session
        active_sessions[session_id]["status"] = "completed"
        active_sessions[session_id]["processing_time"] = processing_time
        
        # Create enhanced response
        enhanced_response = EnhancedQueryResponse(
            session_id=session_id,
            answers=detailed_answers,
            total_processing_time=processing_time,
            total_token_usage=response.token_usage,
            document_statistics={
                "document_count": len(request.documents),
                "total_questions": len(request.questions),
                "average_confidence": sum(response.confidence_scores) / len(response.confidence_scores)
            },
            performance_metrics={
                "questions_per_second": len(request.questions) / processing_time,
                "tokens_per_second": sum(response.token_usage.values()) / processing_time,
                "average_question_time": processing_time / len(request.questions)
            },
            timestamp=datetime.now().isoformat()
        )
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_session, session_id)
        
        logger.info(f"Successfully processed detailed request {session_id} in {processing_time:.2f}s")
        return enhanced_response
        
    except Exception as e:
        logger.error(f"Error processing detailed request {session_id}: {str(e)}")
        
        # Update session with error
        if session_id in active_sessions:
            active_sessions[session_id]["status"] = "error"
            active_sessions[session_id]["error"] = str(e)
        
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="Processing failed",
                details=str(e),
                timestamp=datetime.now().isoformat(),
                session_id=session_id
            ).dict()
        )

@app.post("/api/v1/upload", response_model=Dict[str, str])
async def upload_document(
    file: UploadFile = File(...),
    token: str = Depends(verify_token)
):
    """
    Upload document for processing
    """
    try:
        # Save uploaded file
        file_path = f"./uploads/{file.filename}"
        os.makedirs("./uploads", exist_ok=True)
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return {
            "message": "File uploaded successfully",
            "file_path": file_path,
            "filename": file.filename,
            "size": str(len(content))  # Convert to string to match response model
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/v1/sessions/{session_id}")
async def get_session_status(
    session_id: str,
    token: str = Depends(verify_token)
):
    """Get status of a processing session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return active_sessions[session_id]

@app.get("/api/v1/sessions")
async def list_sessions(token: str = Depends(verify_token)):
    """List all active sessions"""
    return {
        "active_sessions": len(active_sessions),
        "sessions": active_sessions
    }

@app.delete("/api/v1/sessions/{session_id}")
async def delete_session(
    session_id: str,
    token: str = Depends(verify_token)
):
    """Delete a session"""
    if session_id in active_sessions:
        del active_sessions[session_id]
        return {"message": f"Session {session_id} deleted"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/api/v1/metrics")
async def get_metrics(token: str = Depends(verify_token)):
    """Get system performance metrics"""
    uptime = time.time() - app_start_time
    avg_response_time = total_response_time / max(request_count, 1)
    
    return {
        "uptime_seconds": uptime,
        "total_requests": request_count,
        "average_response_time": avg_response_time,
        "active_sessions": len(active_sessions),
        "requests_per_minute": request_count / (uptime / 60) if uptime > 0 else 0
    }

# Helper functions
async def cleanup_session(session_id: str):
    """Background task to cleanup old sessions"""
    await asyncio.sleep(3600)  # Wait 1 hour
    if session_id in active_sessions:
        del active_sessions[session_id]

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "details": str(exc),
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

if __name__ == "__main__":
    import os
    
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    
    print("🚀 Starting Advanced Query-Retrieval API Server")
    print(f"📍 Server will be available at: http://{host}:{port}")
    print(f"📖 API Documentation: http://{host}:{port}/docs")
    print(f"🔧 Alternative Docs: http://{host}:{port}/redoc")
    
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
