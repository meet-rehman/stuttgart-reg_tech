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
        Classify query into types with improved keyword matching
        """
        query_lower = query.query.lower()
        
        # PRIORITY 1: Check for explicit plot/building indicators
        site_keywords = [
            'i want to build', 'i am planning', 'i\'m planning',
            'build a house', 'build a building', 'construct',
            'storeys', 'stories', 'geschosse', 'meters high',
            'm high', 'total area', 'floor area', 'plot',
            'grundstück', 'flurstück', 'parzelle',
            'single family house', 'einfamilienhaus',
            'residential building', 'wohngebäude',
            'development project', 'entwicklungsprojekt'
        ]
        
        # PRIORITY 2: Check for definition questions
        definition_keywords = [
            'what is', 'was ist', 'what does', 'was bedeutet',
            'define', 'definition', 'bedeutung',
            'explain', 'erklären', 'erklärung',
            'difference between', 'unterschied zwischen',
            'meaning of', 'what are', 'was sind'
        ]
        
        # PRIORITY 3: Check for process questions
        process_keywords = [
            'how to', 'wie kann ich', 'how do i', 'wie mache ich',
            'what steps', 'welche schritte', 'procedure', 'verfahren',
            'process for', 'ablauf', 'apply for', 'beantragen',
            'register', 'anmelden', 'submit', 'einreichen',
            'application', 'antrag'
        ]
        
        # PRIORITY 4: Check for general information
        general_keywords = [
            'tell me about', 'erzähl mir über', 'erzähle mir',
            'structure of', 'struktur von', 'organization of',
            'overview of', 'überblick über',
            'history of', 'geschichte von',
            'information about', 'informationen über',
            'what authorities', 'welche behörden'
        ]
        
        # Classification logic - ORDER MATTERS (most specific first)
        
        # Check if it has plot number (always site-specific)
        if query.plot_number:
            return 'site_specific'
        
        # Check for definition patterns (high priority)
        if any(kw in query_lower for kw in definition_keywords):
            # But exclude if it's actually asking to build something
            if not any(kw in query_lower for kw in ['i want', 'i am planning', 'i\'m planning', 'permit for']):
                return 'definition'
        
        # Check for process patterns
        if any(kw in query_lower for kw in process_keywords):
            return 'process'
        
        # Check for general information patterns
        if any(kw in query_lower for kw in general_keywords):
            return 'general'
        
        # Check for site-specific patterns (last, as it's most common)
        if any(kw in query_lower for kw in site_keywords):
            return 'site_specific'
        
        # Default: if unclear and short query, likely definition/general
        if len(query_lower.split()) <= 8:
            return 'general'
        
        # Otherwise treat as site-specific (safe default for building queries)
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
        # CLASSIFY QUERY TYPE (CRITICAL: DO THIS FIRST!)
        # =====================================================================
        
        query_type = self.classify_query_type(query)
        
        print(f"\n{'='*70}")
        print(f"🔍 QUERY CLASSIFICATION")
        print(f"{'='*70}")
        print(f"Query: '{query.query}'")
        print(f"Classified as: {query_type.upper()}")
        print(f"{'='*70}\n")
        
        # =====================================================================
        # ROUTE BASED ON QUERY TYPE
        # =====================================================================
        
        if query_type == 'definition':
            return self._handle_definition_query(query, start_time)
        
        elif query_type == 'process':
            return self._handle_process_query(query, start_time)
        
        elif query_type == 'general':
            return self._handle_general_query(query, start_time)
        
        else:  
            # ================================================================
            # SITE-SPECIFIC QUERIES (Original Logic)
            # ================================================================
            
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
            description=f"""You are reviewing research findings about a {query.project_type} project in {query.location}.

        **RESEARCH FINDINGS FROM DOCUMENT SPECIALIST:**
        [The Document Specialist has already searched and will provide their findings above]

        **YOUR TASK:**
        Write a professional consultation report using the EXACT values the researcher found.

        **CRITICAL RULES:**
        1. Use ONLY the specific values found in the research (GRZ, GFZ, HBA, zoning, plan names)
        2. Do NOT invent or assume values that weren't found
        3. If a value is missing, state: "Not specified - confirm with building department"
        4. Write naturally - don't follow a rigid template

        **REPORT STRUCTURE:**

        # CONSULTATION REPORT: {query.project_type} Development

        ## EXECUTIVE SUMMARY
        [2-3 sentences summarizing what you found - write naturally based on actual findings]

        ## REGULATORY PARAMETERS
        [List the exact values found: GRZ, GFZ, zoning, plan name, height limits, setbacks]
        [For each value, briefly explain what it means for this project]

        ## DEVELOPMENT POTENTIAL
        [Calculate buildable areas using the GRZ/GFZ values found]
        [Describe a realistic development scenario]

        ## OPPORTUNITIES & CONSTRAINTS
        [Based on actual findings, what's possible and what's restricted?]

        ## NEXT STEPS
        [Practical advice: what to do next, who to contact, what to verify]

        ---
        **Prepared by:** Architecture Consultant
        **Date:** {datetime.now().strftime('%Y-%m-%d')}
        **Basis:** Research findings from Stuttgart regulations database

        **Disclaimer:** Verify all values with Stuttgart Building Department before proceeding.

        ---

        **WRITING GUIDELINES:**
        - Be concise and professional
        - Use the researcher's exact values (don't round or estimate)
        - Write in clear, natural language (not template-style)
        - Focus on actionable information
        - Total length: 500-900 words
        """,
            agent=self.agents["architecture_consultant"],
            expected_output="Professional consultation report (800-1200 words) using exact research values, written naturally without rigid formatting"
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
        Classify query with improved keyword detection
        """
        query_lower = query.query.lower()
        
        # Priority 1: If has plot number, always site-specific
        if query.plot_number:
            return 'site_specific'
        
        # Priority 2: Definition patterns (MUST check first)
        if query_lower.startswith('what is ') or query_lower.startswith('was ist '):
            return 'definition'
        
        if query_lower.startswith('what are ') or query_lower.startswith('was sind '):
            # Check if asking about requirements/regulations (general info)
            if any(word in query_lower for word in ['requirement', 'regulation', 'rule', 'law']):
                return 'general'
            else:
                return 'definition'
        
        if query_lower.startswith('what does ') or query_lower.startswith('explain '):
            return 'definition'
        
        # Priority 3: Process patterns
        if query_lower.startswith('how to ') or query_lower.startswith('how do i '):
            return 'process'
        
        if any(phrase in query_lower for phrase in ['what steps', 'procedure for', 'how can i apply']):
            return 'process'
        
        # Priority 4: General information patterns
        if any(phrase in query_lower for phrase in [
            'requirements for', 'regulations for', 'rules for',
            'structure of', 'overview of', 'tell me about',
            'information about', 'what authorities'
        ]):
            return 'general'
        
        # Priority 5: Site-specific patterns (building/construction)
        if any(phrase in query_lower for phrase in [
            'i want to build', 'i am planning', "i'm planning",
            'planning to build', 'build a house', 'build a building',
            'construct a', 'development project',
            'storeys', 'stories', 'meters high', 'm high',
            'total area', 'floor area', 'single family house'
        ]):
            return 'site_specific'
        
        # Priority 6: If mentions plot/land (site-specific)
        if any(word in query_lower for word in [
            'plot', 'grundstück', 'flurstück', 'parzelle', 'land', 'site'
        ]):
            return 'site_specific'
        
        # Default: Short questions are likely general/definition
        if len(query_lower.split()) <= 10:
            return 'general'
        
        # Fallback
        return 'site_specific'
    
    def _handle_definition_query(self, query: RegulationQuery, start_time: float) -> Dict[str, Any]:
        """Handle simple definition/explanation queries"""
        
        print("="*70)
        print("📚 DEFINITION MODE ACTIVATED")
        print("="*70)
        print(f"Query: {query.query}")
        print("Using Document Specialist ONLY (no full report)")
        print("-"*70)
        
        definition_task = Task(
            description=f"""Define/explain this term or concept:
            "{query.query}"
            
            **INSTRUCTIONS:**
            1. Make ONE search for the term/concept
            2. Provide a clear definition (200-300 words)
            3. Include an example if helpful
            4. Cite the legal reference
            5. STOP after providing the definition
            
            **FORMAT:**
            
            **[Term/Concept Name]**
            
            [Clear definition in 2-3 sentences]
            
            **Example:**
            [Practical example if relevant]
            
            **Legal Reference:**
            [BauGB § / BauNVO § / LBO BW §]
            
            DO NOT search multiple times or create a long report.
            """,
            agent=self.agents["document_specialist"],
            expected_output="Clear definition with example (200-300 words)"
        )
        
        crew = Crew(
            agents=[self.agents["document_specialist"]],
            tasks=[definition_task],
            process=Process.sequential,
            verbose=True,
            max_iter=5  # ← CRITICAL: Very low limit for definitions
        )
        
        try:
            print("⏳ Processing definition query...")
            result = crew.kickoff()
            elapsed = time.time() - start_time
            
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
        
        print("="*70)
        print("📋 PROCESS MODE ACTIVATED")
        print("="*70)
        print(f"Query: {query.query}")
        print("Using Document Specialist for step-by-step guidance (no full report)")
        print("-"*70)
        
        process_task = Task(
            description=f"""Explain this administrative process:
            "{query.query}"
            
            **INSTRUCTIONS:**
            1. Make ONE search for process/procedure information
            2. List the steps clearly (400-600 words)
            3. Include required documents and authorities
            4. STOP after listing the steps
            
            **FORMAT:**
            
            **Required Steps:**
            1. [Step 1]
            2. [Step 2]
            3. [Step 3]
            
            **Required Documents:**
            - [Document 1]
            - [Document 2]
            
            **Responsible Authority:**
            [Which office/department]
            
            **Timeline:**
            [Expected processing time]
            
            DO NOT create a full consultation report.
            """,
            agent=self.agents["document_specialist"],
            expected_output="Step-by-step process guide (400-600 words)"
        )
        
        crew = Crew(
            agents=[self.agents["document_specialist"]],
            tasks=[process_task],
            process=Process.sequential,
            verbose=True,
            max_iter=8  # ← CRITICAL: Medium limit for process queries
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
            
        print("="*70)
        print("ℹ️  GENERAL INFORMATION MODE ACTIVATED")
        print("="*70)
        print(f"Query: {query.query}")
        print("Using Document Specialist for information overview (no full report)")
        print("-"*70)
        
        general_task = Task(
            description=f"""Answer this question about Stuttgart regulations:
            "{query.query}"
            
            **INSTRUCTIONS:**
            1. Make ONE search using the tool: search_regulations
            2. Extract key information from the results
            3. Format as a clear, concise answer (300-500 words)
            4. Include relevant citations
            5. STOP after providing the answer
            
            **FORMAT YOUR ANSWER AS:**
            
            [Direct answer to the question in 2-3 paragraphs]
            
            **Key Requirements:**
            - [Bullet point 1]
            - [Bullet point 2]
            - [Bullet point 3]
            
            **Legal References:**
            [Cite specific documents/regulations found]
            
            **DO NOT:**
            - Create a full consultation report
            - Search endlessly for more details
            - Generate a multi-page analysis
            
            Keep it concise and focused on answering the specific question asked.
            """,
            agent=self.agents["document_specialist"],
            expected_output="Concise answer (300-500 words) with citations"
        )
        
        # Create crew with LOWER max_iter for simple queries
        crew = Crew(
            agents=[self.agents["document_specialist"]],
            tasks=[general_task],
            process=Process.sequential,
            verbose=True,
            max_iter=10  # ← CRITICAL: Lower limit for simple queries
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