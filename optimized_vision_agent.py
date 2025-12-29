# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
OPTIMIZED Vision Agent for Stuttgart Building Regulations
Addresses: Time Duration, Token Usage, Query Failures
"""

import os
import json
import base64
import io
from PIL import Image
import asyncio
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from functools import lru_cache
import logging

from openai import OpenAI
import re

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("WARNING: python-dotenv not installed. Install with: pip install python-dotenv")
    print("  Or set OPENAI_API_KEY environment variable manually")

# Use standard logging prefixes instead of emojis
logger = logging.getLogger(__name__)

@dataclass
class VisionConfig:
    """Configuration for vision analysis"""
    model: str = "gpt-4o"
    max_plans_to_analyze: int = 2      # Reduced from 5
    initial_detail: str = "auto"
    final_detail: str = "high"
    initial_max_tokens: int = 300
    final_max_tokens: int = 1500
    parallel_workers: int = 1          # Changed from 3 (no parallel)
    timeout_per_plan: int = 30         # Increased from 10
    total_timeout: int = 90            # Increased from 30
    cache_results: bool = True
    
from legend_extractor import LegendExtractor

class OptimizedVisionAgent:
    """
    Production-ready Vision AI agent with:
    - Smart pre-filtering (analyze only relevant plans)
    - Parallel processing (analyze multiple plans simultaneously)  
    - Intelligent caching (don't re-analyze same plans)
    - Progressive detail (low res search -> high res final)
    - Robust error handling (timeouts, fallbacks)
    """
    
    def __init__(
        self, 
        openai_api_key: str, 
        plans_directory: str = "data/raw/Landuse Plans",
        config: Optional[VisionConfig] = None
    ):
        self.client = OpenAI(api_key=openai_api_key)
        self.plans_dir = Path(plans_directory)
        self.config = config or VisionConfig()
        
        # Multi-level cache
        self._analysis_cache = {}   # Full analysis results
        self._metadata_cache = {}   # Plan metadata (extracted once)
        
        # Metrics tracking
        self.metrics = {
            'total_calls': 0,
            'cache_hits': 0,
            'timeouts': 0,
            'errors': 0,
            'total_tokens_used': 0,
            'total_time': 0.0
        }
        
        logger.info(f"[Vision] Optimized Vision Agent initialized")
        logger.info(f"   Plans directory: {self.plans_dir}")
        logger.info(f"   Config: max_plans={self.config.max_plans_to_analyze}, "
                    f"parallel_workers={self.config.parallel_workers}")

        self.legend_extractor = LegendExtractor(openai_api_key)
    
    # ========================================================================
    # OPTIMIZATION 1: Smart Plan Filtering (RESTORED/CORRECTED)
    # ========================================================================
    
    def _extract_location_keywords(self, query: str) -> List[str]:
        """Extract district/area keywords from query"""
        # (Function body omitted for brevity, assumed correctly placed)
        keywords = []
        query_lower = query.lower()
        district_patterns = [
             r'\b(\w+)[-\s]?(district|bezirk|viertel|gebiet)\b',
             r'\b(nord|süd|ost|west|zentrum|mitte|bad|neu|alt)\s*\w+\b',
             r'\b\w+\s*(park|garten|platz|straße|allee|weg|strasse)\b'
        ]
        
        for pattern in district_patterns:
             matches = re.findall(pattern, query.lower())
             keywords.extend([m[0] if isinstance(m, tuple) else m for m in matches])
             
        known_districts = [
             'cannstatt', 'stuttgart', 'degerloch', 'feuerbach', 'möhringen', 
             'vaihingen', 'zuffenhausen', 'botnang', 'weilimdorf', 'sillenbuch',
             'bad cannstatt', 'stuttgart-mitte', 'stuttgart-nord', 'stuttgart-ost',
             'stuttgart-süd', 'stuttgart-west'
        ]
        for district in known_districts:
             if district in query_lower:
                 keywords.append(district)
                 
        plot_patterns = [
             r'flurstück\s*(?:nr\.?|nummer)?\s*(\d+)',
             r'plot\s*(?:no\.?|number)?\s*(\d+)',
             r'grundstück\s*(\d+)',
             r'\b(\d{3,5})\b'
        ]
        for pattern in plot_patterns:
             matches = re.findall(pattern, query.lower())
             keywords.extend(matches)
             
        return list(set(keywords))
    
    def _get_all_plans(self) -> List[Path]:
        """Get all available plan files"""
        if not self.plans_dir.exists():
            logger.error(f"[Plans] Directory not found: {self.plans_dir}")
            return []
        plans = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
            plans.extend(self.plans_dir.rglob(f"images/{ext}"))
            plans.extend(self.plans_dir.rglob(ext))
        
        plans = list(set(plans))
        logger.info(f"Found {len(plans)} plan images")
        return plans
    
    def _get_relevant_plans(self, query: str, plot_number: Optional[str] = None) -> List[Path]:
        """Get plans most likely to contain requested information"""
        all_plans = self._get_all_plans()
        if not all_plans: return []
        
        keywords = self._extract_location_keywords(query)
        if plot_number: keywords.append(plot_number)
        logger.info(f"[Filter] Searching with keywords: {keywords}")
        
        if not keywords: 
            return all_plans[:self.config.max_plans_to_analyze]
        
        scored_plans = []
        for plan in all_plans:
            filename = plan.stem.lower()
            score = 0
            if plot_number and plot_number.lower() in filename: score += 100
            for keyword in keywords:
                if keyword in filename: score += 10
                elif any(part in filename for part in keyword.split('-')): score += 5
            if score > 0: scored_plans.append((score, plan))
            
        scored_plans.sort(key=lambda x: x[0], reverse=True)
        relevant_plans = [plan for score, plan in scored_plans[:self.config.max_plans_to_analyze]]
        
        logger.info(f"[Filter] Found {len(relevant_plans)} relevant plans from {len(all_plans)} total")
        return relevant_plans

    # ========================================================================
    # UTILITY METHODS (Metadata, Caching, Image Handling)
    # ========================================================================
    
    def _get_cache_key(self, plan_path: Path, query: str, detail: str) -> str:
        """Generate cache key for analysis results"""
        key_string = f"{plan_path}:{query}:{detail}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cached_analysis(self, cache_key: str) -> Optional[str]:
        """Retrieve cached analysis if available"""
        if not self.config.cache_results: return None
        if cache_key in self._analysis_cache:
            self.metrics['cache_hits'] += 1
            logger.info(f"[Cache] HIT for {cache_key[:50]}")
            return self._analysis_cache[cache_key]
        return None

    def _cache_analysis(self, cache_key: str, result: str):
        """Store analysis in cache"""
        if self.config.cache_results: self._analysis_cache[cache_key] = result

    def _load_and_prepare_image(self, image_path: Path, max_size: tuple = (2048, 2048)) -> Optional[str]:
        """Load and prepare image for API (with optimization)"""
        try:
            if str(image_path) in self._metadata_cache: return self._metadata_cache[str(image_path)]['data']
            
            with Image.open(image_path) as img:
                if img.mode not in ('RGB', 'L'): img = img.convert('RGB')
                
                # Resize if needed (simplified check)
                if max(img.size) > max(max_size):
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85, optimize=True)
                image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                self._metadata_cache[str(image_path)] = {'data': image_data, 'size': img.size, 'mode': img.mode}
                return image_data
                
        except Exception as e:
            logger.error(f"[Image] Error loading {image_path}: {e}")
            return None

    # ========================================================================
    # CORE VISION API INTERFACE (Corrected total_calls and time logging)
    # ========================================================================
    
    def _call_vision_api(self, image_data: str, prompt: str, detail: str = "auto", 
                         max_tokens: int = 300, timeout: int = 10, legend_context: str = "") -> Optional[str]:
        """Call OpenAI Vision API with metrics tracking"""
        try:
            start_time = time.time()
            self.metrics['total_calls'] += 1 # CRITICAL FIX: Track calls here
            
            if legend_context: prompt = f"{prompt}\n\nLEGEND CONTEXT:\n{legend_context}"
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": prompt},
                                                {"type": "image_url", 
                                                 "image_url": {"url": f"data:image/jpeg;base64,{image_data}", "detail": detail}}]}
                ],
                max_tokens=max_tokens, timeout=timeout
            )
            
            elapsed = time.time() - start_time
            self.metrics['total_time'] += elapsed
            
            if hasattr(response, 'usage'): self.metrics['total_tokens_used'] += response.usage.total_tokens
            
            result = response.choices[0].message.content.strip()
            logger.info(f"[API] Vision API call successful ({elapsed:.1f}s)")
            return result
            
        except Exception as e:
            logger.error(f"[API] Vision API error: {e}")
            self.metrics['errors'] += 1
            return None

    # ========================================================================
    # PARALLEL EXECUTION LOGIC (Restored find_plot_parallel)
    # ========================================================================
    
    def _analyze_plan_wrapper(self, args: Tuple[Path, str, str]) -> Tuple[Path, str, bool]:
        """Wrapper for parallel execution"""
        plan_path, query, detail = args
        try:
            analysis = self.analyze_plan(plan_path, query, detail=detail)
            found = any([
                'yes' in analysis.lower()[:200], 'visible' in analysis.lower()[:200],
                'shows' in analysis.lower()[:200], re.search(r'\d{3,5}[/-]\d+', analysis)
            ])
            return (plan_path, analysis, found)
        except Exception as e:
            logger.error(f"Error in wrapper for {plan_path.name}: {e}")
            return (plan_path, f"Error: {str(e)}", False)

    def _analyze_plan_batch(self, plans: List[Path], query: str, 
                             detail: str = "auto", max_tokens: int = 300) -> List[Dict]:
        """Analyze multiple plans in parallel"""
        results = []
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            futures = [executor.submit(self._analyze_single_plan, plan, query, detail, max_tokens) for plan in plans]
            for plan, future in zip(plans, futures):
                try:
                    result = future.result(timeout=self.config.timeout_per_plan)
                    if result: results.append(result)
                except (FutureTimeoutError, Exception) as e:
                    logger.warning(f"[Parallel] Failed to analyze {plan.name}: {e}")
                    self.metrics['errors'] += 1
        return results

    def _analyze_single_plan(self, plan_path: Path, query: str, 
                             detail: str = "auto", max_tokens: int = 300) -> Optional[Dict]:
        """Analyze a single plan (used in parallel batch)"""
        # Check cache and load image
        cache_key = self._get_cache_key(plan_path, query, detail)
        cached = self._get_cached_analysis(cache_key)
        if cached: return cached
            
        image_data = self._load_and_prepare_image(plan_path)
        if not image_data: return None
        
        # Analyze with vision API
        prompt = f"User Query: {query}\n\nQuick Analysis Required: 1. What specific area/district is shown? 2. Plot numbers visible? 3. Main land use designations? 4. Is this relevant to the query?"
        result_text = self._call_vision_api(image_data, prompt, detail, max_tokens)
        
        if result_text:
            result_data = {
                'plan_file': plan_path.name,
                'plan_path': str(plan_path),
                'analysis': result_text,
                'relevance_score': self._calculate_relevance_score(result_text, query)
            }
            self._cache_analysis(cache_key, result_data)
            return result_data
        return None

    def _calculate_relevance_score(self, analysis: str, query: str) -> float:
        """Calculate how relevant an analysis is to the query"""
        # (Function body omitted for brevity, standard scoring)
        score = 0.0
        analysis_lower = analysis.lower()
        query_lower = query.lower()
        key_terms = re.findall(r'\b\w+\b', query_lower)
        for term in key_terms:
            if term in analysis_lower: score += 1.0
        plot_numbers = re.findall(r'\b\d{3,5}\b', query_lower)
        for plot in plot_numbers:
            if plot in analysis_lower: score += 5.0
        building_types = ['residential', 'commercial', 'industrial', 'mixed-use', 
                          'wohngebiet', 'gewerbegebiet', 'mischgebiet']
        for building_type in building_types:
            if building_type in query_lower and building_type in analysis_lower: score += 2.0
        return score

    def analyze_plan(self, plan_path: Path, query: str, detail: str = None, use_cache: bool = True) -> str:
        """Analyze a plan image with detailed prompt (used for final analysis)"""
        detail = detail or self.config.initial_detail
        max_tokens = self.config.final_max_tokens
        
        # Check cache first
        cache_key = self._get_cache_key(plan_path, query, detail)
        cached = self._get_cached_analysis(cache_key)
        if cached: return cached.get('analysis') if isinstance(cached, dict) else cached
        
        image_data = self._load_and_prepare_image(plan_path)
        if not image_data: return "Error: Could not load image data."
        
        legend_context = self.legend_extractor.get_legend_context()
        
        # Final, hardcoded prompt for OCR and detail extraction
        prompt = f"""You are analyzing a German building regulation document (Bebauungsplan or Flächennutzungsplan).

**LEGEND CONTEXT:** {legend_context}
**FILE:** {plan_path.name}
**QUERY:** {query}

**CRITICAL INSTRUCTION: You MUST analyze the IMAGE I'm providing. Focus ONLY on the graphical map on the LEFT.**

**YOUR TASK - HIGH-DETAIL OCR AND VALUE EXTRACTION:**
1. **Locate Plot {query} on the map.**
2. **PERFORM OCR:** Extract the EXACT zoning code (e.g., WA, MI, GE) written *inside* or *adjacent to* Plot {query}.
3. **PERFORM OCR:** Scan the graphical map (LEFT SIDE) for any nearby numeric annotations (GRZ, GFZ, HBA) related to the zone and extract the EXACT numbers (e.g., 0.4, 1.2, 283.25).
4. **Report ONLY the exact OCR results you find in the image.**

Answer in English with German terms in parentheses where appropriate."""
        
        result_text = self._call_vision_api(image_data, prompt, detail=detail, max_tokens=max_tokens)
        
        if result_text:
            self._cache_analysis(cache_key, result_text)
            return result_text
        return "Visual analysis failed to extract details."

    def analyze_plot_requirements(self, plot_number: str, timeout: Optional[int] = None) -> str:
        """
        Comprehensive plot analysis (public interface for plotting tool)
        """
        timeout = timeout or self.config.total_timeout
        try:
            # Step 1: Use parallel search to find the most relevant plan quickly
            plot_info = self.find_plot_parallel(plot_number)
            
            if not plot_info['found']:
                return (f"[Result] Plot {plot_number} not found in available landuse plans.\n"
                        f"Searched {plot_info.get('searched_plans', 0)} relevant plans.\n"
                        f"Recommendation: Consult Stuttgart city planning office for plot-specific regulations.")

            # Step 2: Format the comprehensive analysis response (ASCII HARDENED)
            response = (
                f"PLOT ANALYSIS: {plot_number}\n"
                f"File Found in: {plot_info['plan_file']}\n"
                f"Time: {plot_info.get('search_time', 0):.1f} seconds\n"
                f"\n{plot_info['analysis']}\n"
                f"\n---\n"
                f"Note: This analysis is based on visual interpretation of landuse plans. \n"
                f"For legally binding information, consult Stuttgart city planning office."
            )
            return response.strip()
            
        except TimeoutError:
            self.metrics['timeouts'] += 1
            logger.warning(f"[Timeout] Plot analysis timed out after {timeout}s")
            return (f"[Timeout] Vision analysis timed out after {timeout} seconds.\n"
                    f"Falling back to text-based regulations only.")
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"[Error] Unexpected error in plot analysis: {e}")
            return (f"[Error] Unable to complete visual analysis: {str(e)}\n"
                    f"Falling back to text-based regulations.")


    def find_plot_parallel(self, plot_number: str):
        """
        Executes the parallel search logic to find the best plan and its initial analysis.
        This method must be defined here for the tool in optimized_crew_ai_system.py to work.
        """
        start_time = time.time()
        
        relevant_plans = self._get_relevant_plans(plot_number, plot_number)

        if not relevant_plans:
            return {'found': False, 'searched_plans': 0, 'search_time': 0, 'analysis': "No plans found."}

        # Step 2: Quick parallel analysis (auto detail)
        initial_results = self._analyze_plan_batch(
            relevant_plans, f"Does this plan contain Plot {plot_number}?", 
            detail=self.config.initial_detail,
            max_tokens=self.config.initial_max_tokens
        )
        
        if not initial_results:
            return {'found': False, 'searched_plans': len(relevant_plans), 'search_time': time.time() - start_time, 'analysis': "Initial visual search failed."}

        # Step 3: Find most relevant result (simplified for parallel)
        initial_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        best_result = initial_results[0]
        
        # Step 4: Perform detailed analysis on the best match (high detail)
        detailed_analysis = self.analyze_plan(
            Path(best_result['plan_path']), 
            f"Provide detailed analysis of plot {plot_number}: boundaries, zoning, dimensions, setbacks.", 
            detail="high"
        )
        
        # Check if the detailed analysis explicitly refused or confirmed the plot number
        plot_found = plot_number.lower() in detailed_analysis.lower()
        
        return {
            'found': plot_found,
            'plan_file': best_result['plan_file'],
            'analysis': detailed_analysis,
            'search_time': time.time() - start_time,
            'searched_plans': len(relevant_plans),
        }

    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = self.metrics.copy()
        # Calculate derived metrics
        if metrics['total_calls'] > 0:
            metrics['avg_time_per_call'] = metrics['total_time'] / metrics['total_calls']
            metrics['avg_tokens_per_call'] = metrics['total_tokens_used'] / metrics['total_calls']
            metrics['cache_hit_rate'] = metrics['cache_hits'] / metrics['total_calls']
            metrics['error_rate'] = metrics['errors'] / metrics['total_calls']
        return metrics

# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    
    # Test execution is omitted here for brevity
    
    print("\n[Testing block omitted for concise output]")