#!/usr/bin/env python3
"""
Multi-Agent Flask Application for Stuttgart Building Regulations
ENHANCED WITH VISION AI for Plot-Specific Queries
"""
from dotenv import load_dotenv
load_dotenv()

import os
import json
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Import our multi-agent system
from optimized_crew_ai_system import StuttgartBuildingRegulationCrew, RegulationQuery

# Existing imports
from schemas import ChatRequest, ChatResponse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
crew_system: Optional[StuttgartBuildingRegulationCrew] = None

class MultiAgentRequest(BaseModel):
    """Request model for multi-agent analysis"""
    query: str
    project_type: str = "mixed-use"
    location: str = "Stuttgart"
    district: str = "general"
    urgency: str = "normal"
    use_multi_agent: bool = True
    plot_number: Optional[str] = None  # NEW: For plot-specific queries

class PlotQueryRequest(BaseModel):
    """Request model for plot-specific queries (NEW!)"""
    plot_number: str  # Required, e.g., "9232/79"
    query: str  # What to analyze about the plot
    project_type: str = "residential"
    include_visual_analysis: bool = True  # Whether to analyze plans visually
    include_text_regulations: bool = True  # Whether to include text regulations

class MultiAgentResponse(BaseModel):
    """Response model for multi-agent analysis"""
    analysis: str
    timestamp: str
    query_details: Dict[str, Any]
    processing_time: Optional[float] = None
    agents_used: List[str] = []
    plot_info: Optional[Dict[str, Any]] = None  # NEW: Plot-specific info if applicable

class PlotAnalysisResponse(BaseModel):
    """Response model for plot-specific analysis (NEW!)"""
    plot_number: str
    found_in_plans: bool
    plan_file: Optional[str] = None
    visual_analysis: Optional[str] = None
    regulatory_analysis: Optional[str] = None
    combined_analysis: str
    timestamp: str
    processing_time: Optional[float] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global crew_system
    
    try:
        print("🚀 Starting Stuttgart Building Regulation Multi-Agent System...")
        print("   WITH VISION AI CAPABILITIES")
        
        # Validate environment
        print("Step 1: Validating environment variables...")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        print("✅ Environment variables validated")
        
        # Initialize multi-agent system
        print("Step 2: Initializing multi-agent crew with vision...")
        crew_system = StuttgartBuildingRegulationCrew(
            openai_api_key,
            use_gpt4=False,  # ✅ Use gpt-4o-mini (fast + vision)
            enable_vision=True  # ✅ Enable vision AI
        )
        # ✅ ADD THIS - Enable verbose output
        import logging
        logging.getLogger('crewai').setLevel(logging.DEBUG)
        logging.getLogger('optimized_crew_ai_system').setLevel(logging.DEBUG)

        print("✅ Multi-agent crew initialized successfully")
        
        # Test vision agent
        if crew_system.vision_agent:
            print("✅ Vision agent ready for plot analysis")
        else:
            print("⚠️ Vision agent not available")
        
        print("✅ APP STARTED SUCCESSFULLY!")
        print("="*60)
        
        yield
        
        print("\n🛑 Shutting down...")
        crew_system = None
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        raise

# Create FastAPI app
app = FastAPI(
    title="Stuttgart Building Regulations API - With Vision AI",
    description="Multi-agent system with visual plan analysis for Stuttgart building regulations",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ADD THESE LINES HERE ⬇️
# Serve static files (for the interactive UX)
from pathlib import Path
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ============================================================================
# NEW ENDPOINTS - PLOT-SPECIFIC QUERIES
# ============================================================================

@app.post("/api/plot/analyze", response_model=PlotAnalysisResponse)
async def analyze_plot(request: PlotQueryRequest, background_tasks: BackgroundTasks):
    """
    Analyze a specific plot using vision AI and text regulations
    
    NEW ENDPOINT! Combines:
    - Visual analysis of landuse plans
    - Text-based regulatory search
    - Comprehensive plot-specific recommendations
    """
    if not crew_system:
        raise HTTPException(status_code=503, detail="AI system not initialized")
    
    start_time = datetime.now()
    
    try:
        logger.info(f"📍 Plot analysis request for: {request.plot_number}")
        
        # 1. Visual analysis (if enabled and available)
        visual_result = None
        plan_file = None
        
        if request.include_visual_analysis and crew_system.vision_agent:
            logger.info("🎨 Running visual analysis...")
            
            # Search for plot in plans
            plot_search = crew_system.vision_agent.find_plot(request.plot_number)
            
            if plot_search['found']:
                plan_file = plot_search['plan_file']
                visual_result = plot_search['analysis']
                
                # Get detailed analysis
                detailed_analysis = crew_system.vision_agent.analyze_plot_requirements(
                    request.plot_number
                )
                visual_result = detailed_analysis
                
                logger.info(f"✅ Plot found in: {plan_file}")
            else:
                logger.warning(f"⚠️ Plot {request.plot_number} not found in plans")
                visual_result = f"Plot {request.plot_number} not found in available landuse plans."
        
        # 2. Text regulations analysis (if enabled)
        regulatory_result = None
        
        if request.include_text_regulations:
            logger.info("📚 Running regulatory analysis...")
            
            # Create regulation query
            reg_query = RegulationQuery(
                query=f"{request.query} for plot {request.plot_number}",
                project_type=request.project_type,
                location="Stuttgart",
                district="",
                plot_number=request.plot_number
            )
            
            # Run full crew analysis
            result_dict = crew_system.execute_analysis(reg_query)
            regulatory_result = result_dict.get('analysis', str(result_dict))  # ← Extract the text
        
        # 3. Combine results
        combined = f"""COMPREHENSIVE PLOT ANALYSIS: {request.plot_number}
{'='*80}

Query: {request.query}
Project Type: {request.project_type}

"""
        
        if visual_result:
            combined += f"""VISUAL ANALYSIS FROM LANDUSE PLANS:
{'='*80}
{visual_result}

"""
        
        if regulatory_result:
            combined += f"""REGULATORY ANALYSIS:
        {'='*80}
        {regulatory_result}  # ← Now it's a string

"""
        
        if not visual_result and not regulatory_result:
            combined += "No analysis results available. Check if plot exists and regulations are loaded."
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return PlotAnalysisResponse(
            plot_number=request.plot_number,
            found_in_plans=plan_file is not None,
            plan_file=plan_file,
            visual_analysis=visual_result,
            regulatory_analysis=regulatory_result,
            combined_analysis=combined,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"❌ Plot analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Plot analysis failed: {str(e)}")


@app.get("/api/plot/search/{plot_number}")
async def search_plot(plot_number: str):
    """
    Quick search for a plot in landuse plans (vision only, no full analysis)
    
    NEW ENDPOINT! Quick visual search without running full crew.
    """
    if not crew_system or not crew_system.vision_agent:
        raise HTTPException(
            status_code=503, 
            detail="Vision AI not available. Check if plans directory exists."
        )
    
    try:
        logger.info(f"🔍 Quick plot search: {plot_number}")
        
        result = crew_system.vision_agent.find_plot(plot_number)
        
        return JSONResponse(content={
            "plot_number": plot_number,
            "found": result['found'],
            "plan_file": result.get('plan_file'),
            "analysis": result.get('analysis') if result['found'] else result.get('message'),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Plot search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/plans/available")
async def list_available_plans():
    """
    List all available landuse plan images
    
    NEW ENDPOINT! Shows what plans are available for analysis.
    """
    if not crew_system or not crew_system.vision_agent:
        return JSONResponse(content={
            "available": False,
            "message": "Vision AI not initialized",
            "plans": []
        })
    
    try:
        plans = crew_system.vision_agent._get_available_plans()
        
        plan_info = [
            {
                "filename": p.name,
                "path": str(p.relative_to(crew_system.vision_agent.plans_dir)),
                "size_mb": p.stat().st_size / (1024 * 1024)
            }
            for p in plans
        ]
        
        return JSONResponse(content={
            "available": True,
            "count": len(plans),
            "plans_directory": str(crew_system.vision_agent.plans_dir),
            "plans": plan_info
        })
        
    except Exception as e:
        logger.error(f"❌ Error listing plans: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ENHANCED EXISTING ENDPOINTS
# ============================================================================

@app.post("/api/multi-agent/analyze")  # ✅ Removed response_model
async def multi_agent_analysis(request: MultiAgentRequest, background_tasks: BackgroundTasks):
    """
    Multi-agent analysis - ENHANCED with plot support
    
    Now supports plot_number field for plot-specific queries.
    """
    if not crew_system:
        raise HTTPException(status_code=503, detail="AI system not initialized")
    
    start_time = datetime.now()
    
    try:
        logger.info(f"🎯 Multi-agent analysis: {request.query}")
        if request.plot_number:
            logger.info(f"📍 Plot-specific query for: {request.plot_number}")
        
        # Create query
        query = RegulationQuery(
            query=request.query,
            project_type=request.project_type,
            location=request.location,
            district=request.district,
            urgency=request.urgency,
            plot_number=request.plot_number
        )
        
        # Execute analysis - returns dict with 'success', 'analysis', etc.
        result = crew_system.execute_analysis(query)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Check if plot was found (if plot query)
        plot_info = None
        if request.plot_number and crew_system.vision_agent:
            plot_search = crew_system.vision_agent.find_plot(request.plot_number)
            if plot_search['found']:
                plot_info = {
                    "found": True,
                    "plan_file": plot_search['plan_file']
                }
        
        # Build response from result dict
        return {
            "success": result.get('success', True),
            "analysis": result.get('analysis', str(result)),  # ✅ Extract analysis string
            "timestamp": datetime.now().isoformat(),
            "query_details": {
                "query": request.query,
                "project_type": request.project_type,
                "location": request.location,
                "district": request.district,
                "urgency": request.urgency,
                "plot_number": request.plot_number
            },
            "processing_time": result.get('processing_time', processing_time),
            "agents_used": [
                "Document & Visual Plan Specialist",
                "Regulatory Legal Analyst"
            ],
            "plot_info": plot_info,
            "vision_used": result.get('vision_used', False),
            "is_plot_query": result.get('is_plot_query', False)
        }
        
    except Exception as e:
        logger.error(f"❌ Multi-agent analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def legacy_chat_endpoint(request: ChatRequest):
    """Legacy single-agent endpoint for backward compatibility"""
    try:
        if crew_system:
            # Detect plot numbers in message
            import re
            plot_match = re.search(r'\d{4}/\d+', request.message)
            plot_number = plot_match.group(0) if plot_match else None
            
            multi_agent_request = MultiAgentRequest(
                query=request.message,
                project_type="mixed-use",
                location="Stuttgart",
                district="general",
                plot_number=plot_number
            )
            
            result = await multi_agent_analysis(multi_agent_request, BackgroundTasks())
            
            return ChatResponse(
                message=result.analysis,
                timestamp=result.timestamp,
                context_used=5,
                conversation_id=getattr(request, 'conversation_id', None)
            )
        else:
            raise HTTPException(status_code=503, detail="AI system not available")
            
    except Exception as e:
        logger.error(f"❌ Legacy chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint - serve interactive UX"""
    static_file = Path(__file__).parent / "static" / "index.html"
    if static_file.exists():
        return HTMLResponse(content=static_file.read_text())
    else:
        # Fallback to basic HTML if static file not found
        return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Stuttgart Building Regulations API - With Vision AI</title>
    <style>
        body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
        .feature { background: #f0f0f0; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .new { background: #e8f5e9; border-left: 4px solid #4caf50; }
        code { background: #333; color: #fff; padding: 2px 6px; border-radius: 3px; }
    </style>
</head>
<body>
    <h1>🏗️ Stuttgart Building Regulations API</h1>
    <h2>NOW WITH VISION AI! 🎨</h2>
    
    <div class="feature new">
        <h3>🆕 NEW: Plot-Specific Analysis</h3>
        <p>Analyze specific plots using visual landuse plans:</p>
        <code>POST /api/plot/analyze</code>
        <p>Quick plot search: <code>GET /api/plot/search/{plot_number}</code></p>
    </div>
    
    <div class="feature new">
        <h3>🆕 NEW: Available Plans</h3>
        <p>See what landuse plans are loaded:</p>
        <code>GET /api/plans/available</code>
    </div>
    
    <div class="feature">
        <h3>📚 Multi-Agent Analysis</h3>
        <p>Comprehensive analysis with multiple specialized agents:</p>
        <code>POST /api/multi-agent/analyze</code>
        <p>Now supports <code>plot_number</code> field!</p>
    </div>
    
    <div class="feature">
        <h3>📖 Documentation</h3>
        <p>API docs: <a href="/docs">/docs</a></p>
        <p>Interactive: <a href="/redoc">/redoc</a></p>
    </div>
    
    <h3>Example Plot Query:</h3>
    <pre style="background: #f5f5f5; padding: 15px; border-radius: 5px;">
{
  "plot_number": "9232/79",
  "query": "What can I build here? What are the setback requirements?",
  "project_type": "residential",
  "include_visual_analysis": true,
  "include_text_regulations": true
}
    </pre>
</body>
</html>
""")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "crew_initialized": crew_system is not None,
        "vision_available": crew_system.vision_agent is not None if crew_system else False,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    
    print(f"\n🚀 Starting server on port {port}...")
    print(f"📍 Docs: http://localhost:{port}/docs")
    print(f"🎨 Vision AI: Enabled")
    print(f"📐 Plans directory: data/raw/Landuse Plans/\n")
    
    uvicorn.run(
        "multi_agent_app:app",  # ✅ Changed from 'app' to string format
        host="0.0.0.0",
        port=port,
        reload=True  # ✅ Added auto-reload
    )