"""
Pydantic schemas for the Stuttgart Building Registration Agent API
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class BuildingSearchRequest(BaseModel):
    """Request model for building search"""
    query: str = Field(..., description="Search query for buildings")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results to return")

class BuildingResult(BaseModel):
    """Individual building search result"""
    content: str = Field(..., description="Building information content")
    score: float = Field(..., description="Relevance score")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    source: Optional[str] = Field(default=None, description="Source document")

class BuildingSearchResponse(BaseModel):
    """Response model for building search"""
    results: List[BuildingResult] = Field(..., description="List of search results")
    query: str = Field(..., description="Original search query")
    total_results: int = Field(..., description="Total number of results")
    timestamp: str = Field(..., description="Response timestamp")

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., description="User message")
    conversation_id: Optional[str] = Field(default=None, description="Optional conversation ID")

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    message: str = Field(..., description="Assistant response")
    timestamp: str = Field(..., description="Response timestamp")
    context_used: Optional[int] = Field(default=None, description="Number of context documents used")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID")

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="System status (healthy/degraded/unhealthy)")
    timestamp: str = Field(..., description="Health check timestamp")
    environment: str = Field(..., description="Environment (Local/Railway)")
    api_ready: bool = Field(..., description="Whether API client is ready")
    rag_ready: bool = Field(..., description="Whether RAG system is ready")
    components: Dict[str, str] = Field(..., description="Status of individual components")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
    timestamp: str = Field(..., description="Error timestamp")

# Additional models for future features
class DocumentUploadRequest(BaseModel):
    """Request model for document upload"""
    filename: str = Field(..., description="Document filename")
    content_type: str = Field(..., description="Document content type")

class DocumentUploadResponse(BaseModel):
    """Response model for document upload"""
    document_id: str = Field(..., description="Uploaded document ID")
    filename: str = Field(..., description="Document filename")
    status: str = Field(..., description="Upload status")
    timestamp: str = Field(..., description="Upload timestamp")