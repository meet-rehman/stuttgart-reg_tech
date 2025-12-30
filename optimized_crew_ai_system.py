#!/usr/bin/env python3
"""
Multi-Agent Stuttgart Building Regulations System using CrewAI
OPTIMIZED VERSION with Vision AI Integration
Addresses: Time Duration, Token Usage, Query Failures
"""
import os
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import logging


# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print(" ⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
    print("    Or set environment variables manually")

# CrewAI imports
from crewai import Agent, Task, Crew, Process

# Try to import LangChain Tool
try:
    from langchain.agents import Tool as LangChainTool
except ImportError:
    # Simple fallback if import fails
    class LangChainTool:
        def __init__(self, name, description, func):
            self.name = name
            self.description = description
            self.func = func
                
# Add ChatOpenAI import
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    try:
        from langchain.chat_models import ChatOpenAI
    except ImportError:
        print("Warning: ChatOpenAI not found")
        
# Custom imports
from railway_optimized_rag import RailwayOptimizedRAG as PrecomputedRAGSystem
from optimized_vision_agent import OptimizedVisionAgent, VisionConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # ✅ Override any existing config
)
logger = logging.getLogger(__name__)

@dataclass
class RegulationQuery:
    """Structure for regulation queries"""
    query: str
    project_type: str = "mixed-use"
    location: str = "Stuttgart"
    district: str = "general"
    urgency: str = "normal"
    plot_number: Optional[str] = None
    enable_vision: bool = True     # NEW: Allow disabling vision per query
    vision_timeout: int = 30     # NEW: Configurable vision timeout

# ============================================================================
# MULTI-AGENT CREW SYSTEM - OPTIMIZED
# ============================================================================
class StuttgartBuildingRegulationCrew:
    """
    CrewAI-based multi-agent system for Stuttgart building regulations

    OPTIMIZATIONS:
    - Vision Agent with smart filtering and caching
    - Parallel plan analysis 
    - Graceful fallback when vision fails
    - Performance metrics tracking
    """

    def __init__(
        self, 
        openai_api_key: str,
        use_gpt4: bool = False,
        vision_config: Optional[VisionConfig] = None,
        enable_vision: bool = True
    ):
        """
        Initialize the crew with RAG system and optional Vision AI
        """
        self.openai_api_key = openai_api_key
        
        # ========================================================================
        # LLM CONFIGURATION - OPTIMIZED FOR SPEED + VISION
        # ========================================================================
        
        # Default: gpt-4o-mini (fast, cheap, has vision)
        vision_model = "gpt-4o" if use_gpt4 else "gpt-4o-mini"
        
        self.vision_llm = ChatOpenAI(
            model=vision_model,
            temperature=0.1,
            openai_api_key=openai_api_key
        )
        
        # Faster text-only LLM for Legal Analyst (doesn't need vision)
        text_model = "gpt-4o-mini" if use_gpt4 else "gpt-3.5-turbo"
        self.text_llm = ChatOpenAI(
            model=text_model,
            temperature=0.1,
            openai_api_key=openai_api_key
        )
        
        self.llm = self.vision_llm
        
        # Initialize RAG system
        logger.info(" 🔧  Initializing RAG system...")
        self.rag_system = PrecomputedRAGSystem()
        
        # Initialize Vision Agent (with optimization)
        self.vision_agent = None
        self.vision_enabled = enable_vision
        
        if enable_vision:
            try:
                logger.info(" 🎨  Initializing Optimized Vision Agent...")
                self.vision_agent = OptimizedVisionAgent(
                    openai_api_key=openai_api_key,
                    config=vision_config or VisionConfig()
                )
                logger.info(" ✅  Vision Agent ready")
            except Exception as e:
                logger.warning(f" ⚠️  Vision Agent initialization failed: {e}")
                logger.warning("   Continuing with text-only mode")
                self.vision_enabled = False
        
        # Create agents with tools
        logger.info(" 👥  Creating specialized agents...")
        self.agents = self._create_agents()
        logger.info(" ✅  All agents initialized")
        
        # Metrics
        self.metrics = {
            'total_queries': 0,
            'vision_used': 0,
            'vision_timeouts': 0,
            'vision_failures': 0,
            'total_processing_time': 0.0
        }
    
    # ========================================================================
    # TOOL DEFINITIONS - Enhanced with Vision Tools
    # ========================================================================
    
    def _create_tools(self) -> Dict[str, LangChainTool]:
        """Create tools for agents (text + vision)"""
        
        # ---- TEXT-BASED TOOLS ----
        
        def search_regulations(query: str) -> str:
            """Search building regulations"""
            try:
                # Force search to hit documents that contain specific terms
                forced_query = f"{query} HBA GRZ GFZ Bebauungsplan" 
                results = self.rag_system.search(forced_query, top_k=5) 
                
                if not results:
                    return "No relevant regulations found."
                
                response = "**Found Regulations:**\n\n"
                for i, res in enumerate(results, 1):
                    content = res.content if hasattr(res, 'content') else str(res)
                    source = res.source if hasattr(res, 'source') else 'Unknown'
                    score = res.score if hasattr(res, 'score') else 0
                    
                    response += f"{i}. (Score: {score:.2f}) {content[:300]}...\n"
                    response += f"   Source: {source}\n\n"
                
                return response
            except Exception as e:
                return f"Error searching regulations: {str(e)}"
        
        def get_context(topic: str) -> str:
            """Get detailed regulatory context"""
            try:
                results = self.rag_system.search(topic, top_k=3)
                
                if not results:
                    return "No context found."
                
                output = []
                for i, result in enumerate(results, 1):
                    content = result.content if hasattr(result, 'content') else str(result)
                    source = result.source if hasattr(result, 'source') else 'Unknown'
                    score = result.score if hasattr(result, 'score') else 0
                    
                    output.append(f"**Context {i}** (Score: {score:.2f})\n"
                                  f"Source: {source}\n"
                                  f"{content[:600]}...\n")
                
                return "\n".join(output)
            except Exception as e:
                return f"Error getting context: {str(e)}"
            
        
        def analyze_hierarchy(topic: str) -> str:
            """Analyze regulatory hierarchy"""
            hierarchy = """
**German Building Regulation Hierarchy:**
1. **Federal Level (Bundesrecht):**
   - Baugesetzbuch (BauGB) - Federal Building Code
   - Highest authority, applies nationwide
   
2. **State Level (Landesrecht):**
   - Landesbauordnung Baden-Württemberg (LBO BW)
   - State building regulations
   
3. **Local Level (Kommunalrecht):**
   - Stuttgart City Building Plan (Bebauungsplan)
   - Local zoning and regulations
   
**Precedence:** Federal > State > Local
"""
            return hierarchy
        
        def estimate_costs(requirements: str) -> str:
            """Estimate compliance costs"""
            return f"""
**Estimated Compliance Costs:**
Based on typical Stuttgart projects:
- Building permit: €500-2,000
- Structural engineer: €2,000-10,000
- Energy certificate: €300-800
- Fire safety review: €1,000-5,000
- Timeline: 3-6 months for approval
Note: Actual costs vary by project complexity.
Contact local architect for detailed quote.
"""
        
        # ---- VISION TOOLS ----
        
        def search_plot_in_plans(plot_number: str) -> str:
            """Search for a plot in visual landuse plans"""
            if not self.vision_enabled or not self.vision_agent:
                return " ⚠️  Vision analysis not available. Using text regulations only."
            
            try:
                # IMPORTANT: This method must exist on OptimizedVisionAgent
                result = self.vision_agent.find_plot_parallel(plot_number)
                
                self.metrics['vision_used'] += 1
                
                if result['found']:
                    return f"""
✅  **Plot {plot_number} found in {result['plan_file']}**
{result['analysis']}
📊  Search completed in {result.get('search_time', 0):.1f} seconds
    Analyzed {result.get('plans_analyzed', 0)} plans
"""
                else:
                    return f"""
⚠️  **Plot {plot_number} not found**
Searched {result.get('searched_plans', 0)} relevant plans.
Time: {result.get('search_time', 0):.1f} seconds
Recommendation: Consult Stuttgart city planning office for plot-specific information.
"""
                
            except TimeoutError:
                self.metrics['vision_timeouts'] += 1
                return " ⏱️  Vision search timed out. Continuing with text regulations only."
            
            except Exception as e:
                self.metrics['vision_failures'] += 1
                logger.error(f"Vision search error: {e}")
                return f" ⚠️  Vision analysis failed: {str(e)}. Using text regulations only."
        
        def analyze_plot_details(plot_number: str) -> str:
            """
            Get comprehensive plot analysis from visual plans (uses vision agent)
            """
            if not self.vision_enabled or not self.vision_agent:
                return " ⚠️  Vision analysis not available."
            
            try:
                # IMPORTANT: This method must exist on OptimizedVisionAgent
                result = self.vision_agent.analyze_plot_requirements(plot_number)
                return result
            
            except TimeoutError:
                return " ⏱️  Detailed plot analysis timed out."
            
            except Exception as e:
                logger.error(f"Plot analysis error: {e}")
                return f" ⚠️  Plot analysis failed: {str(e)}"
        
        def analyze_general_plan(area_query: str) -> str:
            """
            Analyze plans for general area questions (not plot-specific)
            """
            if not self.vision_enabled or not self.vision_agent:
                return " ⚠️  Vision analysis not available."
            
            try:
                # Get relevant plans
                plans = self.vision_agent._filter_relevant_plans(
                    area_query, 
                    self.vision_agent._get_available_plans()
                )
                
                if not plans:
                    return "No relevant plans found for this area."
                
                # Analyze first relevant plan with low detail (fast, cheap)
                result = self.vision_agent.analyze_plan(
                    plans[0], 
                    area_query,
                    detail="low"  # Use low detail for general overview
                )
                
                return f"""
📍  **Area Analysis**
Based on {plans[0].name}:
{result}
"""
            except Exception as e:
                return f" ⚠️  General plan analysis failed: {str(e)}"
        
        # Return all tools
        return {
            'text_search': LangChainTool(
                name="search_regulations",
                description="Search building regulations from text documents",
                func=search_regulations
            ),
            'context': LangChainTool(
                name="get_context",
                description="Get detailed regulatory context with citations",
                func=get_context
            ),
            'hierarchy': LangChainTool(
                name="analyze_hierarchy",
                description="Analyze German building regulation hierarchy",
                func=analyze_hierarchy
            ),
            'costs': LangChainTool(
                name="estimate_costs",
                description="Estimate compliance costs and timeline",
                func=estimate_costs
            ),
            'plot_search': LangChainTool(
                name="search_plot_in_plans",
                description="Search for specific plot number in visual landuse plans (optimized, with timeout)",
                func=search_plot_in_plans
            ),
            'plot_analysis': LangChainTool(
                name="analyze_plot_details",
                description="Get comprehensive analysis of a specific plot from visual plans",
                func=analyze_plot_details
            ),
            'general_plan': LangChainTool(
                name="analyze_general_plan",
                description="Analyze landuse plans for general area/zoning questions",
                func=analyze_general_plan
            )
        }
    
    # ========================================================================
    # AGENT CREATION
    # ========================================================================
    
    def _create_agents(self) -> Dict[str, Agent]:
        """Create specialized agents with tools"""
        
        tools = self._create_tools()
            
        # AGENT 1: Document & Visual Plan Specialist
        document_specialist = Agent(
            role="Document & Visual Plan Specialist",
            goal="Find EXACT numerical values from building regulations using multiple targeted searches, and analyze site plans visually with legend knowledge",
            backstory="""Expert at searching German building regulation documents and reading architectural plans. Can identify plots, extract zoning information, and measure distances from visual plans. 
        
        **VISION CAPABILITIES:**
        - Can identify plots in visual Bebauungspläne
        - Has access to extracted legend knowledge (WA/MI/GE zones, colors, symbols)
        - Interprets zoning colors: red=residential, yellow=commercial, green=parks
        - Recognizes standard abbreviations: GRZ, GFZ, HBA, EFH

        **CRITICAL SEARCH TECHNIQUE:**
        You know that German building plans store technical values as simple labels:
        - HBA 283.25 (building height)
        - GRZ 0.4 (plot coverage ratio) 
        - EFH 261.50 (floor elevation)
        **YOUR PROVEN STRATEGY (Always follow this):**
        1. Make MULTIPLE separate searches (5-7 searches), NOT one complex query
        2. Use SHORT queries (2-4 words): "Stgt 272", "HBA", "GRZ", "GFZ"
        3. Avoid long queries like "building regulations for residential in Stuttgart with height requirements"
        4. Extract EXACT numbers from results - never use generic ranges or typical values
        5. If you don't find exact values after multiple searches, state that clearly
        6. **CRITICAL: Distinguish elevation from building height:**
       - "m ü. NN" or "HBA XXX" = Plot ELEVATION above sea level (topographic reference)
       - "Höhe max" or "maximale Gebäudehöhe" = Actual BUILDING HEIGHT limit
       - Example: "HBA 283.25 m ü. NN" means plot elevation is 283.25m above sea level
       - This is NOT the maximum building height!
       - Always search separately for actual height restrictions

        **WHY THIS WORKS:**
        -  ✅  Search "HBA"  →  Finds "HBA 283.25" (good match)
        -  ❌  Search "maximum building height for residential area"  →  Poor match
        You understand that technical documents need technical search terms.
        You never give generic German building code knowledge - only specific values you actually find.
        You know when to use vision tools vs text search based on the query type.""",
            llm=self.vision_llm,
            verbose=True,
            tools=[
                tools['text_search'], 
                tools['context'],
                tools['plot_search'],
                tools['plot_analysis']
            ],
            allow_delegation=False,
            max_iter=20   # Increased iteration limit
        )

        #Agent 2: architecture_consultant_agent
        architecture_consultant_agent = Agent(
        role="Architecture & Real Estate Development Consultant",
        goal="""Provide professional architectural and real estate development 
        consultation based on Stuttgart building regulations. Analyze development 
        potential, identify opportunities and constraints, and deliver actionable 
        strategic recommendations for architects, developers, and investors.""",
        
        backstory="""You are a Senior Consultant with 15+ years of experience in 
        German urban development and building regulations, specializing in Stuttgart 
        municipal law. You have successfully guided 100+ development projects through 
        the permitting process, from initial feasibility to final approval.
        
        Your expertise includes:
        - Interpreting complex Bebauungspläne and zoning regulations
        - Calculating maximum buildable parameters (GRZ, GFZ, heights)
        - Identifying regulatory risks and opportunities
        - Providing strategic development recommendations
        - Preparing professional consultation reports for clients
        
        You translate complex regulatory language into practical, actionable guidance 
        that architects and developers can immediately use. Your reports are known 
        for being thorough, accurate, and client-focused.""",
        
        verbose=True,
        allow_delegation=False,
        llm=self.text_llm,
        max_iter=3,
        memory=True
        )
            
        # AGENT 3: Legal Analyst
        legal_analyst = Agent(
            role="Regulatory Legal Analyst",
            goal="Interpret regulatory hierarchy and legal requirements",
            backstory="""Legal expert in German building law hierarchy. Understands precedence 
            between BauGB (federal), LBO (state), and local regulations. Identifies potential conflicts 
            and provides clear legal interpretations.
            Then refer to the Textteil from stgt and compare the legal requirements of the particular plot/query:
            A. Planungsrechtliche Festsetzungen nach BauGB und BauNVO
            B. Örtliche Bauvorschriften nach LBO
            C. Hinweise
            - Zeichenerklärung
            - Ausfertigung
            Then explain shortly about findings from the stgt and the text details for the particular query""",
            llm=self.text_llm,
            verbose=True,
            tools=[tools['hierarchy']],
            allow_delegation=False,
            max_iter=5
        )
        
        return {
            "document_specialist": document_specialist,
            "architecture_consultant": architecture_consultant_agent,
            "legal_analyst": legal_analyst,
        }
    

    def classify_query_type(self, query: RegulationQuery) -> str:
        """
        Classify query into types:
        - 'site_specific': Requires consultation report (plot/building analysis)
        - 'definition': Simple definition/explanation
        - 'process': Administrative process question
        - 'general': General information
        """
        query_lower = query.query.lower()
        
        # Site-specific indicators
        site_keywords = [
            'build', 'plot', 'grundstück', 'flurstück', 
            'can i', 'möchte bauen', 'planning to build',
            'development', 'projekt', 'single family',
            'einfamilienhaus', 'wohngebäude', 'storeys',
            'geschosse', 'height', 'höhe', 'area', 'fläche',
            'permit for', 'genehmigung für'
        ]
        
        # Definition indicators
        definition_keywords = [
            'what is', 'was ist', 'define', 'definition',
            'bedeutung', 'explain', 'erklären', 'difference between',
            'unterschied zwischen', 'meaning of'
        ]
        
        # Process indicators
        process_keywords = [
            'how to', 'wie kann ich', 'steps', 'schritte',
            'process', 'verfahren', 'register', 'anmelden',
            'apply', 'beantragen', 'submit', 'einreichen',
            'procedure', 'ablauf'
        ]
        
        # General information indicators
        general_keywords = [
            'tell me about', 'erzähl mir über',
            'overview', 'überblick', 'history', 'geschichte',
            'climate', 'klima', 'statistics', 'statistik'
        ]
        
        # Check query type (order matters - more specific first)
        if query.plot_number or any(kw in query_lower for kw in site_keywords):
            return 'site_specific'
        elif any(kw in query_lower for kw in definition_keywords):
            return 'definition'
        elif any(kw in query_lower for kw in process_keywords):
            return 'process'
        elif any(kw in query_lower for kw in general_keywords):
            return 'general'
        else:
            # Default: if unclear, treat as site-specific to be safe
            return 'site_specific'

    # ========================================================================
    # CORE EXECUTION METHOD (Must be indented inside the class)
    # ========================================================================

    def execute_analysis(self, query: RegulationQuery) -> Dict[str, Any]:
        """
        Execute multi-agent analysis with query-type-aware routing
        """
        start_time = time.time()
        
        # ADD THESE PRINT STATEMENTS FOR VISIBILITY
        print("\n" + "="*70)
        print("🚀 CREWAI MULTI-AGENT SYSTEM STARTING")
        print("="*70)
        print(f"📝 Query: {query.query}")
        print(f"🏢 Type: {query.project_type}")
        print(f"📍 Location: {query.location}, {query.district}")
        if query.plot_number:
            print(f"📋 Plot: {query.plot_number}")
        print("="*70)
        
        logger.info(f"🎯 Starting analysis for: {query.query}")
        
        self.metrics['total_queries'] += 1
        
        # =====================================================================
        # NEW: CLASSIFY QUERY TYPE
        # =====================================================================
        
        query_type = self.classify_query_type(query)
        print(f"🔍 Query Type: {query_type.upper()}")
        
        # =====================================================================
        # ROUTE BASED ON QUERY TYPE
        # =====================================================================
        
        if query_type == 'definition':
            return self._handle_definition_query(query, start_time)
        
        elif query_type == 'process':
            return self._handle_process_query(query, start_time)
        
        elif query_type == 'general':
            return self._handle_general_query(query, start_time)
        
        # =====================================================================
        # SITE-SPECIFIC QUERIES (Original Logic)
        # =====================================================================
        
        print("🏗️ SITE-SPECIFIC MODE: Full consultation report")
        
        # Detect plot-specific query
        is_plot_query = query.plot_number is not None or any(
            keyword in query.query.lower() 
            for keyword in ['plot', 'flurstück', 'grundstück', 'parzelle', 'flurstücknummer', 'einfamilienhaus', 'efh']
        )
        
        if is_plot_query:
            print("📍 Detected plot-specific query") 
            logger.info("📍 Detected plot-specific query")
        
        # Check if vision should be used for this query
        use_vision = (
            self.vision_enabled and 
            query.enable_vision and 
            (is_plot_query or 'plan' in query.query.lower())
        )
        
        # ADD THESE VISIBILITY MESSAGES
        if use_vision:
            print("👁️ Vision analysis ENABLED")
        else:
            print("📚 Text-only analysis mode")
        
        # =====================================================================
        # TASK 1: Research (Text + Vision)
        # =====================================================================
        
        # ADD THIS BEFORE TASK CREATION
        print("\n" + "-"*70)
        print("📚 AGENT 1: DOCUMENT SPECIALIST")
        print("-"*70)
        print("🔍 Starting regulatory document search...")
        
        research_description = f"""Research this building regulation query:
    "{query.query}"

    Project Details:
    - Type: {query.project_type}
    - Location: {query.location}, {query.district}
    {'- Plot Number: ' + query.plot_number if query.plot_number else ''}

    **CRITICAL: MULTI-STEP SEARCH STRATEGY**
    You MUST make multiple separate searches (5-8 searches) using SHORT queries.
    Complex queries fail - simple queries succeed!

    """
            
        if use_vision and is_plot_query:
            research_description += f"""
    **VISION-FIRST STRATEGY FOR PLOT {query.plot_number or 'QUERY'}:**

    STEP 1 (MANDATORY - DO THIS FIRST):
    Tool: search_plot_in_plans
    Input: "{query.plot_number or 'plot number from query'}"
    Purpose: Find plot location and identify which Bebauungsplan it's in
    Expected output: Plan name (e.g., "Stgt 286-2"), zoning (e.g., "WA")

    STEP 2 (IF STEP 1 FOUND THE PLOT):
    Tool: analyze_plot_details  
    Input: "{query.plot_number or 'plot number from query'}"
    Purpose: Get comprehensive plot analysis
    Expected output: Detailed analysis with any visible regulatory values

    STEP 3 (AFTER VISION STEPS):
    Tool: search_regulations
    Input: [Plan name from Step 1] (e.g., "Stgt 286-2" or "Stgt 272")
    Purpose: Find text-based regulations for this specific plan
    
    STEP 4:
    Tool: search_regulations
    Input: "GRZ"
    Purpose: Find plot coverage ratio values
    
    STEP 5:
    Tool: search_regulations
    Input: "GFZ"
    Purpose: Find floor area ratio values
    
    STEP 6:
    Tool: search_regulations
    Input: "HBA"
    Purpose: Find building height values

    STEP 7 (OPTIONAL):
    Tool: get_context
    Input: [Plan name from Step 1]
    Purpose: Get additional regulatory context

    # ADD THIS CRITICAL INSTRUCTION BLOCK
    **CRITICAL FALLBACK (ONLY IF ALL SEARCHES ABOVE RETURN NOTHING):**
    If, after all searches, you still lack specific GRZ, GFZ, or HBA values:
    Assume Stuttgart's most common residential values for the final report:
    - GRZ: 0.4
    - GFZ: 1.2
    - HBA: 283.25 m ü. NN (or approx 3-4 stories)
    State clearly in your output: "Specific regulatory values not found; proceeding with standard Stuttgart assumptions (GRZ 0.4 / GFZ 1.2)."

    **CRITICAL RULES:**
    1. DO Steps 1-2 (vision) BEFORE any text searches
    2. If vision times out or fails, continue with Steps 3-6
    3. Vision gives you: plot location, zoning, plan name
    4. Text searches give you: GRZ, GFZ, HBA values
    5. Combine both sources in your final answer

    **IF VISION FAILS:**
    Continue with text-only strategy (Steps 3-6), but state: 
    "Vision analysis unavailable. Using text regulations only."
    """
        else:
            research_description += """
    **TEXT-ONLY SEARCH STRATEGY:**

    STEP 1:
    Tool: search_regulations
    Input: "GRZ"
    Purpose: Find plot coverage ratio
    
    STEP 2:
    Tool: search_regulations
    Input: "GFZ"
    Purpose: Find floor area ratio
    
    STEP 3:
    Tool: search_regulations
    Input: "HBA"
    Purpose: Find building height values

    STEP 4:
    Tool: get_context
    Input: [Plan name from Step 1]
    Purpose: Get additional regulatory details

    **CRITICAL FALLBACK (ONLY IF ALL SEARCHES ABOVE RETURN NOTHING):**
    If, after all searches, you still lack specific GRZ, GFZ, or HBA values:
    Assume Stuttgart's most common residential values for the final report:
    - GRZ: 0.4
    - GFZ: 1.2
    - HBA: 283.25 m ü. NN (or approx 3-4 stories)
    State clearly in your output: "Specific regulatory values not found; proceeding with standard Stuttgart assumptions (GRZ 0.4 / GFZ 1.2)."

    **WHY SHORT QUERIES WORK BETTER:**
    - ✅ "HBA" matches "HBA 283.25" (good)
    - ❌ "building height in Stgt 272 for residential"  →  Poor match
    """
            
        research_description += """

    **EXTRACT EXACT VALUES:**
    After each search, look for specific numbers:
    - HBA values: "HBA 283.25" or "HBA 28325" (may lack decimal)
    - GRZ values: "GRZ 0.4" or "Grundflächenzahl: 0,4"  
    - GFZ values: "GFZ 1.2" or "Geschossflächenzahl: 1,2"
    - EFH values: "EFH 261.50" (floor elevation)
    - Zoning: "WA" (Wohngebiet), "MI" (Mischgebiet), "GE" (Gewerbegebiet)

    **FORMATTING RULES:**
    ✅ GOOD: "GRZ: 0.4 (Source: stgt-286-2-bebauungsplan, Page 1)"
    ✅ GOOD: "Plot 18A is in WA zone (Source: Vision analysis of Stgt 286-2)"
    ✅ GOOD: "HBA: 283.25 m ü. NN (Source: site draft)"
    ❌ BAD: "GRZ typically 0.3-0.4" (don't use generic values)
    ❌ BAD: "around 0.4" (be exact)
    ❌ BAD: Generic German building code knowledge without sources

    **IF VALUES NOT FOUND:**
    State clearly: "Exact [GRZ/GFZ/HBA] value not found in available documents"
    Do NOT make up typical values or estimates.

    **FOR LOCATION QUESTIONS:**
    - Identify specific streets from search results
    - State zoning types: WA (residential), MI (mixed), GE (commercial)
    - Example: "Allowed in WA zones along Nordbahnhofstraße and Türlenstraße"

    **RESPONSE LENGTH:**
    Keep under 800 words. Focus on:
    1. Exact values with sources
    2. Clear citations
    3. No repetition
    4. No generic information

    Provide detailed findings with EXACT values and clear source citations.
    """

        # Create Task 1        
        research_task = Task(
            description=research_description,
            agent=self.agents["document_specialist"],
            expected_output="Detailed findings with EXACT numerical values and sources. For plot queries, include both vision analysis results (plot location, zoning) and text-based values (GRZ, GFZ, HBA). Maximum 800 words."
        )

        # =====================================================================
        # TASK 2: Architecture Consultation (FINAL TASK)
        # =====================================================================
        
        # ADD THIS BEFORE TASK 2
        print("\n" + "-"*70)
        print("🏗️ AGENT 2: ARCHITECTURE CONSULTANT")
        print("-"*70)
        print("📊 Will analyze findings and prepare professional report...")
        
        consultation_task = Task(
            description=f"""Create a professional consultation report for {query.project_type} project.

        **CRITICAL: Use the EXACT values provided by the Research Specialist.**

        The Research Specialist has already found:
        - Plot location and zoning (from vision analysis if available)
        - GRZ, GFZ, HBA values (from text search)
        - Bebauungsplan reference
        - Regulatory constraints

        **Your job:**
        1. Take those EXACT values from the research findings
        2. Calculate buildable areas (GRZ × plot area, GFZ × plot area)
        3. Provide realistic development scenario
        4. List specific opportunities and constraints based on the findings
        5. Give actionable next steps

        **Output a professional report with:**

        # CONSULTATION REPORT: {query.project_type} Development

        ## EXECUTIVE SUMMARY
        [2-3 sentences using ACTUAL findings from research. If research found GRZ 0.4 and WA zoning, mention those specific values here.]

        ## SITE ANALYSIS
        - Plot: {query.plot_number or "General Stuttgart area"}
        - Zoning: [From research - e.g., "WA (Wohngebiet)" - use EXACT zoning from research]
        - Applicable Plan: [From research - e.g., "Stgt 286-2" - use EXACT plan name from research]

        ## REGULATORY PARAMETERS (from Research Specialist)

        **Use the EXACT values the Research Specialist found. Do NOT say "Not specified" if they found values!**

        - **GRZ (Grundflächenzahl):** [EXACT value from research, e.g., "0.4"] 
        → Meaning: This allows [GRZ × 100]% ground coverage. For a 1000m² plot, max building footprint = [GRZ × 1000]m²

        - **GFZ (Geschossflächenzahl):** [EXACT value from research, e.g., "1.2"]
        → Meaning: Total buildable floor area = [GFZ × plot area]. For 1000m² plot = [GFZ × 1000]m² total

        - **Plot Elevation:** [If HBA found, e.g., "HBA 283.25 m ü. NN (elevation above sea level)"]
        → Meaning: This is the topographic reference, NOT a height restriction

        - **Building Height Limit:** [If "Höhe max" found, e.g., "12 meters" or "3 stories"]
        → Meaning: Maximum allowed building height above ground level
        → [If not found: "Height limit not specified in available documents - confirm with building department"]

        - **Setbacks:** [From research if available, otherwise state "To be confirmed with building department"]

        ## DEVELOPMENT POTENTIAL

        **If research found GRZ and GFZ values, do calculations:**

        Assuming typical Stuttgart residential plot of 800-1000m²:
        - Maximum ground coverage: [GRZ × 900]m² (e.g., 0.4 × 900 = 360m²)
        - Maximum total floor area: [GFZ × 900]m² (e.g., 1.2 × 900 = 1,080m²)

        **Realistic Development Scenario for {query.project_type}:**
        - Ground floor: [GRZ × 900]m² (e.g., 360m²)
        - Upper floors: [remaining area across stories] (e.g., 720m² across 2 upper floors = 360m² each)
        - Total built area: Approximately [GFZ × 900]m²
        - Configuration: [Describe realistic building - e.g., "3-story Einfamilienhaus with 360m² per floor, including ground floor with living areas, 2 upper floors with bedrooms, plus garage"]

        **If research did NOT find specific values:**
        Specific dimensional requirements not found in available documents. General Stuttgart {query.project_type} projects typically allow GRZ 0.4-0.6 and GFZ 1.2-1.8, but exact values must be confirmed with Stuttgart Building Department for this specific location.

        ## KEY OPPORTUNITIES
        
        **Base these on ACTUAL research findings:**

        ✅ [If WA zoning found: "WA (Wohngebiet) zoning allows residential development without special permits"]
        ✅ [If GRZ found: "GRZ [value] allows [generous/moderate/limited] building footprint of up to [calculated]m²"]
        ✅ [If location/streets found: "Located near [street names from research] providing good accessibility"]
        ✅ [If specific plan found: "Subject to [plan name] which provides clear development framework"]

        **If no specific findings:**
        ✅ Stuttgart location provides access to infrastructure
        ✅ Clear regulatory framework under BauGB and LBO Baden-Württemberg
        ✅ Residential market demand in Stuttgart

        ## CRITICAL CONSTRAINTS

        **Base these on ACTUAL research findings:**

        ⚠️ [If GRZ found: "GRZ [value] limits ground coverage to [percentage]%"]
        ⚠️ [If height limits found: "Building height restricted to [value] meters"]
        ⚠️ [If missing values: "Specific setback requirements not specified in available documents - must confirm with building department"]
        ⚠️ [General: "Compliance with LBO Baden-Württemberg required for all construction"]

        ## RECOMMENDED NEXT STEPS

        ### Immediate Actions (Next 30 Days):

        1. **Contact Stuttgart Building Department (Baurechtsamt):**
        - Request official Bebauungsplan for {query.plot_number or "this location"}
        - Confirm exact GRZ, GFZ, and height restrictions
        - Obtain setback requirements
        - Inquire about current permit processing times

        2. **Commission Technical Surveys:**
        - Official cadastral survey for exact plot dimensions
        - Geotechnical investigation
        - Environmental assessment (if applicable)

        3. **Engage Professionals:**
        - Local Stuttgart architect familiar with {query.district or "area"} regulations
        - Structural engineer for preliminary design

        ### Planning Phase (Months 2-4):

        1. Develop concept design respecting confirmed regulatory parameters
        2. Pre-application consultation with building department
        3. Prepare preliminary cost estimates

        ### Permitting Phase (Months 5-9):

        1. Finalize architectural plans
        2. Submit Bauantrag (building permit application)
        3. Address any review comments
        4. Obtain building permit

        ## ESTIMATED TIMELINE & COSTS

        **Timeline:**
        - Regulatory clarification & surveys: 1-2 months
        - Design development: 2-3 months  
        - Permitting: 4-6 months
        - Construction: 12-18 months
        - **Total: 19-29 months**

        **Estimated Costs (for {query.project_type}):**
        - Building permit fees: €500-2,000
        - Architect fees: €15,000-50,000 (depending on project scope)
        - Engineering: €10,000-30,000
        - Construction: €2,500-4,000/m² (varies by specification)

        *Note: Costs are estimates. Obtain detailed quotes from local professionals.*

        ## REGULATORY REFERENCES

        - **Bebauungsplan:** [From research if found, e.g., "Stgt 286-2"]
        - **Federal Law:** Baugesetzbuch (BauGB)
        - **State Law:** Landesbauordnung Baden-Württemberg (LBO BW)
        - **Vision Analysis:** [If vision was used: "Visual plan analysis of [plan name]"]
        - **Text Sources:** [List specific documents from research]

        ---

        **Prepared By:** Architecture & Real Estate Development Consultant   
        **Date:** {datetime.now().strftime('%Y-%m-%d')}  
        **Project Type:** {query.project_type}
        **Location:** {query.location}, {query.district}

        **Basis:** Research findings from Stuttgart building regulations database and {
            "visual plan analysis" if use_vision and is_plot_query else "text-based regulatory documents"
        }

        **Important Disclaimer:** This consultation is based on available regulatory data as of {datetime.now().strftime('%Y-%m-%d')}. 
        All dimensional requirements, costs, and timelines must be verified with:
        - Stuttgart Building Department (Baurechtsamt Stuttgart)
        - Licensed local architect
        - Structural engineer

        Plot-specific requirements may vary. Official cadastral survey and building department consultation are mandatory before proceeding with design.

        ---

        **CRITICAL INSTRUCTIONS FOR YOU:**
        - Use EXACT values from research (don't say "Not specified" if research found values!)
        - If research found GRZ 0.4, USE 0.4 in ALL calculations
        - If research found zoning WA, EXPLAIN what WA allows for {query.project_type}
        - Make calculations realistic with example plot sizes (800-1000m² typical)
        - Provide SPECIFIC advice based on actual research findings, not generic templates
        - If certain values missing, clearly state what's missing and how to obtain it
        """,
            agent=self.agents["architecture_consultant"],
            expected_output="Professional consultation report using exact research values, with realistic calculations and specific recommendations. Report should be 1200-1800 words and client-ready."
        )

        # =====================================================================
        # CREATE AND EXECUTE CREW
        # =====================================================================
        
        # ADD THESE VISIBILITY MESSAGES
        print("\n" + "="*70)
        print("🤝 CREATING CREW WITH 2 AGENTS")
        print("="*70)
        print("✅ Document Specialist: Ready")
        print("✅ Architecture Consultant: Ready")
        print("-"*70)

        crew = Crew(
            agents=[
                self.agents["document_specialist"],
                self.agents["architecture_consultant"]
            ],  # Only 2 agents needed
            tasks=[research_task, consultation_task],  # Only 2 tasks
            process=Process.sequential,
            verbose=True,
            memory=True  # Enable context sharing between agents
        )

        try:
            # ADD THIS BEFORE KICKOFF
            print("\n🚀 EXECUTING CREW.KICKOFF()...")
            print("⏳ This may take 30-90 seconds...")
            print("-"*70)
            
            logger.info("🚀 Starting crew execution...")
            
            # CAPTURE AND DISPLAY OUTPUT
            import sys
            from io import StringIO
            
            # Store original stdout
            original_stdout = sys.stdout
            
            # Create a custom stdout that both captures and displays
            class TeeOutput:
                def __init__(self):
                    self.terminal = original_stdout
                    self.log = StringIO()
                
                def write(self, message):
                    self.terminal.write(message)  # Display to console
                    self.log.write(message)      # Capture for later
                
                def flush(self):
                    self.terminal.flush()
            
            # Replace stdout to capture CrewAI output
            sys.stdout = TeeOutput()
            
            # Execute crew
            result = crew.kickoff()

            
            # Get captured output and restore stdout
            captured_output = sys.stdout.log.getvalue()
            sys.stdout = original_stdout
            
            elapsed = time.time() - start_time
            self.metrics['total_processing_time'] += elapsed
            
            # ADD COMPLETION MESSAGES
            print("-"*70)
            print(f"✅ CREW EXECUTION COMPLETE!")
            print(f"⏱️ Total time: {elapsed:.2f} seconds")
            
            logger.info(f"✅ Analysis complete in {elapsed:.2f}s")
            
            # =====================================================================
            # Extract FINAL task output (Architecture Consultation)
            # =====================================================================
            
            final_analysis = None
            
            if hasattr(result, 'tasks_output') and len(result.tasks_output) > 0:
                # Get the LAST task output (Architecture Consultation)
                final_analysis = result.tasks_output[-1].raw
                print(f"📊 Final report length: {len(final_analysis)} characters")
                logger.info(f"📊 Final report length: {len(final_analysis)} characters")
            elif hasattr(result, 'raw'):
                final_analysis = result.raw
            else:
                final_analysis = str(result)
            
            # Verify we got meaningful output
            if not final_analysis or len(final_analysis) < 100:
                logger.warning("⚠️ Final analysis too short, using full result")
                final_analysis = str(result)
            
            # ADD FINAL SUMMARY
            print("\n" + "="*70)
            print("📊 ANALYSIS SUMMARY")
            print("="*70)
            print(f"✅ Success: True")
            print(f"📝 Report length: {len(final_analysis)} characters")
            print(f"👁️ Vision used: {use_vision}")
            print(f"📍 Plot query: {is_plot_query}")
            print(f"⏱️ Processing time: {elapsed:.2f} seconds")
            print("="*70 + "\n")
            
            # Build response with metrics
            return {
                'success': True,
                'analysis': final_analysis,   # Returns final consultation report
                'processing_time': elapsed,
                'vision_used': use_vision,
                'is_plot_query': is_plot_query,
                'timestamp': datetime.now().isoformat(),
                'metrics': self._get_query_metrics()
            }
            
        except Exception as e:
            elapsed = time.time() - start_time
            
            # ADD ERROR VISIBILITY
            print("\n" + "="*70)
            print("❌ ERROR OCCURRED")
            print("="*70)
            print(f"Error: {str(e)}")
            print(f"Time elapsed: {elapsed:.2f} seconds")
            print("="*70 + "\n")
            
            logger.error(f"❌ Crew execution error: {e}", exc_info=True)
            
            return {
                'success': False,
                'error': str(e),
                'analysis': f"Error: Unable to complete analysis - {str(e)}",
                'processing_time': elapsed,
                'timestamp': datetime.now().isoformat()
            }
            
    # ========================================================================
    # QUERY CLASSIFICATION & ROUTING
    # ========================================================================
    
    def classify_query_type(self, query: RegulationQuery) -> str:
        """
        Classify query into types:
        - 'site_specific': Requires consultation report (plot/building analysis)
        - 'definition': Simple definition/explanation
        - 'process': Administrative process question
        - 'general': General information
        """
        query_lower = query.query.lower()
        
        # Site-specific indicators
        site_keywords = [
            'build', 'plot', 'grundstück', 'flurstück', 
            'can i', 'möchte bauen', 'planning to build',
            'development', 'projekt', 'single family',
            'einfamilienhaus', 'wohngebäude', 'storeys',
            'geschosse', 'height', 'höhe', 'area', 'fläche',
            'permit for', 'genehmigung für', '3 storey',
            '12m high', 'total area', 'residential building'
        ]
        
        # Definition indicators
        definition_keywords = [
            'what is', 'was ist', 'define', 'definition',
            'bedeutung', 'explain', 'erklären', 'difference between',
            'unterschied zwischen', 'meaning of', 'what does', 'was bedeutet'
        ]
        
        # Process indicators
        process_keywords = [
            'how to', 'wie kann ich', 'steps', 'schritte',
            'process', 'verfahren', 'register', 'anmelden',
            'apply', 'beantragen', 'submit', 'einreichen',
            'procedure', 'ablauf', 'how do i'
        ]
        
        # General information indicators
        general_keywords = [
            'tell me about', 'erzähl mir über',
            'overview', 'überblick', 'history', 'geschichte',
            'climate', 'klima', 'statistics', 'statistik'
        ]
        
        # Check query type (order matters - more specific first)
        if query.plot_number or any(kw in query_lower for kw in site_keywords):
            return 'site_specific'
        elif any(kw in query_lower for kw in definition_keywords):
            return 'definition'
        elif any(kw in query_lower for kw in process_keywords):
            return 'process'
        elif any(kw in query_lower for kw in general_keywords):
            return 'general'
        else:
            # Default: if unclear, treat as site-specific to be safe
            return 'site_specific'
    
    def _handle_definition_query(self, query: RegulationQuery, start_time: float) -> Dict[str, Any]:
        """Handle simple definition/explanation queries"""
        
        print("📚 DEFINITION MODE: Using Document Specialist only")
        
        # Simple task for Document Specialist
        definition_task = Task(
            description=f"""Answer this definition/explanation question:
            "{query.query}"
            
            Search the documents and provide a clear, concise explanation (200-400 words).
            Include relevant citations and examples.
            
            Format your answer as:
            **[Term/Concept]**
            
            [Clear explanation in 2-3 paragraphs]
            
            **Example:**
            [Practical example if relevant]
            
            **Legal Reference:**
            [Cite specific laws/regulations: BauGB, BauNVO, LBO BW]
            
            Do NOT create a full consultation report - just answer the question directly.
            """,
            agent=self.agents["document_specialist"],
            expected_output="Clear explanation with citations (200-400 words)"
        )
        
        # Single agent crew
        crew = Crew(
            agents=[self.agents["document_specialist"]],
            tasks=[definition_task],
            process=Process.sequential,
            verbose=True
        )
        
        try:
            print("⏳ Processing definition query...")
            result = crew.kickoff()
            elapsed = time.time() - start_time
            
            # Extract answer
            answer = result.tasks_output[-1].raw if hasattr(result, 'tasks_output') else str(result)
            
            print(f"✅ Definition query complete in {elapsed:.2f}s")
            
            return {
                'success': True,
                'analysis': answer,
                'processing_time': elapsed,
                'query_type': 'definition',
                'timestamp': datetime.now().isoformat(),
                'metrics': self._get_query_metrics()
            }
        
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ Definition query error: {e}")
            return {
                'success': False,
                'error': str(e),
                'analysis': f"Error processing definition query: {str(e)}",
                'processing_time': elapsed,
                'timestamp': datetime.now().isoformat()
            }

    def _handle_process_query(self, query: RegulationQuery, start_time: float) -> Dict[str, Any]:
        """Handle administrative process questions"""
        
        print("📋 PROCESS MODE: Using Document Specialist for procedure guidance")
        
        # Research process
        process_task = Task(
            description=f"""Find information about this administrative process:
            "{query.query}"
            
            Search for and provide:
            
            **Required Steps:**
            [List the sequential steps clearly]
            
            **Required Documents:**
            [List all necessary documents]
            
            **Responsible Authorities:**
            [Which offices to contact]
            
            **Timeline:**
            [Expected processing time]
            
            **Relevant Regulations:**
            [Cite applicable laws]
            
            **Contact Information:**
            [If available in documents]
            
            Provide clear, actionable information (400-600 words).
            Do NOT create a full consultation report - focus on the process.
            """,
            agent=self.agents["document_specialist"],
            expected_output="Process information with steps and requirements (400-600 words)"
        )
        
        crew = Crew(
            agents=[self.agents["document_specialist"]],
            tasks=[process_task],
            process=Process.sequential,
            verbose=True
        )
        
        try:
            print("⏳ Processing procedure query...")
            result = crew.kickoff()
            elapsed = time.time() - start_time
            
            answer = result.tasks_output[-1].raw if hasattr(result, 'tasks_output') else str(result)
            
            print(f"✅ Process query complete in {elapsed:.2f}s")
            
            return {
                'success': True,
                'analysis': answer,
                'processing_time': elapsed,
                'query_type': 'process',
                'timestamp': datetime.now().isoformat(),
                'metrics': self._get_query_metrics()
            }
        
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ Process query error: {e}")
            return {
                'success': False,
                'error': str(e),
                'analysis': f"Error processing procedure query: {str(e)}",
                'processing_time': elapsed,
                'timestamp': datetime.now().isoformat()
            }

    def _handle_general_query(self, query: RegulationQuery, start_time: float) -> Dict[str, Any]:
        """Handle general information queries"""
        
        print("ℹ️ GENERAL MODE: Using Document Specialist for information")
        
        general_task = Task(
            description=f"""Provide information about:
            "{query.query}"
            
            Search the documents and provide a comprehensive overview (400-700 words).
            
            Include:
            - Key facts and context
            - Relevant statistics (if available)
            - Historical background (if relevant)
            - Current situation
            - References to source documents
            
            Structure your response clearly with sections if needed.
            """,
            agent=self.agents["document_specialist"],
            expected_output="Comprehensive information with context (400-700 words)"
        )
        
        crew = Crew(
            agents=[self.agents["document_specialist"]],
            tasks=[general_task],
            process=Process.sequential,
            verbose=True
        )
        
        try:
            print("⏳ Processing general information query...")
            result = crew.kickoff()
            elapsed = time.time() - start_time
            
            answer = result.tasks_output[-1].raw if hasattr(result, 'tasks_output') else str(result)
            
            print(f"✅ General query complete in {elapsed:.2f}s")
            
            return {
                'success': True,
                'analysis': answer,
                'processing_time': elapsed,
                'query_type': 'general',
                'timestamp': datetime.now().isoformat(),
                'metrics': self._get_query_metrics()
            }
        
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ General query error: {e}")
            return {
                'success': False,
                'error': str(e),
                'analysis': f"Error processing general query: {str(e)}",
                'processing_time': elapsed,
                'timestamp': datetime.now().isoformat()
            }        


    # ========================================================================
    # METRICS & MONITORING
    # ========================================================================
    def _get_query_metrics(self) -> Dict[str, Any]:
        """Get metrics for last query"""
        metrics = {
            'crew_metrics': self.metrics.copy()
        }
        
        if self.vision_agent:
            metrics['vision_metrics'] = self.vision_agent.get_metrics()
        
        return metrics

    def print_performance_report(self):
        """Print comprehensive performance report"""
        print("\n" + "="*70)
        print(" 📊  MULTI-AGENT CREW PERFORMANCE REPORT")
        print("="*70)
        
        print("\n 🤖  Crew Metrics:")
        print(f"   Total queries:         {self.metrics['total_queries']}")
        print(f"   Vision queries:        {self.metrics['vision_used']}")
        print(f"   Vision timeouts:       {self.metrics['vision_timeouts']}")
        print(f"   Vision failures:       {self.metrics['vision_failures']}")
        print(f"   Total time:            {self.metrics['total_processing_time']:.2f}s")
        
        if self.metrics['total_queries'] > 0:
            avg_time = self.metrics['total_processing_time'] / self.metrics['total_queries']
            print(f"   Avg time per query:    {avg_time:.2f}s")
        
        if self.vision_agent:
            print("\n 🎨  Vision Agent Metrics:")
            # CORRECTED ACCESS: Use get_metrics() which exists
            print(self.vision_agent.get_metrics())
        
        print("="*70 + "\n")

# ============================================================================
# TESTING
# ============================================================================
if __name__ == "__main__":
    import sys
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(" ❌  OPENAI_API_KEY environment variable required")
        sys.exit(1)
    
    print("\n" + "="*70)
    print(" 🧪  TESTING OPTIMIZED MULTI-AGENT CREW")
    print("="*70)
    
    # Create optimized crew
    crew = StuttgartBuildingRegulationCrew(
        openai_api_key=api_key,
        use_gpt4=False,  # Use GPT-3.5 for cost efficiency
        vision_config=VisionConfig(
            max_plans_to_analyze=3,  # Analyze max 3 plans
            parallel_workers=2,      # 2 plans in parallel
            timeout_per_plan=60,     # 20s timeout per plan
            total_timeout=120        # 40s total timeout
        ),
        enable_vision=True
    )
    
    # TEST 1: Plot-specific query
    print("\n" + "="*70)
    print("TEST 1: Plot-Specific Query with Vision")
    print("="*70)
    
    query1 = RegulationQuery(
        query="What can I build on plot 12B in Stuttgart-Nord?",
        location="Stuttgart",
        district="Stadtbezirk Stuttgart-Nord",
        plot_number="12B",
        enable_vision=True,
        vision_timeout=120
    )
    
    result1 = crew.execute_analysis(query1)
    
    print(f"\n ✅  Success: {result1['success']}")
    print(f" ⏱️  Processing time: {result1['processing_time']:.2f}s")
    print(f" 👁️  Vision used: {result1.get('vision_used', False)}")
    print("\n 📄  Analysis:")
    print("-" * 70)
    print(result1.get('analysis', result1.get('error')))
    
    # TEST 2: General regulation query (no vision)
    print("\n" + "="*70)
    print("TEST 2: General Query (Text-Only)")
    print("="*70)
    
    query2 = RegulationQuery(
        query="What can be the official steps that one can take as an Architect, builder and devleloper, muncipal authorities needs to be considered according to each individual authority.",
        project_type="residential",
        location="Stuttgart",
        district="Stadtbezirk Stuttgart-Nord",
        enable_vision=False  # Explicitly disable vision
    )
    
    result2 = crew.execute_analysis(query2)
    
    print(f"\n ✅  Success: {result2['success']}")
    print(f" ⏱️  Processing time: {result2['processing_time']:.2f}s")
    print(f" 👁️  Vision used: {result2.get('vision_used', False)}")
    print("\n 📄  Analysis:")
    print("-" * 70)
    print(result2.get('analysis', result2.get('error'))[:500] + "...")
    
    # Print performance report
    crew.print_performance_report()
    
    print("\n ✅  Testing complete!")