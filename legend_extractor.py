#!/usr/bin/env python3
"""
Legend Extractor for Stuttgart Building Regulations
Extracts and stores legend information for use in plan analysis
"""

import os
import json
from pathlib import Path
from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv
import base64
import io
from PIL import Image

load_dotenv()

class LegendExtractor:
    """Extract and store legend information from plan documents"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.legends_cache = {}
        self.cache_file = Path("data/legends_cache.json")
        
        # Load existing cache
        if self.cache_file.exists():
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                self.legends_cache = json.load(f)
    
    def extract_legend(self, legend_image_path: Path) -> Dict:
        """
        Extract legend information from an image
        
        Returns a structured dictionary of:
        - Zoning symbols (WA, MI, GE, etc.)
        - Color meanings
        - Symbol definitions
        - Text abbreviations
        """
        
        # Check cache first
        cache_key = legend_image_path.name
        if cache_key in self.legends_cache:
            print(f"✅ Using cached legend for {cache_key}")
            return self.legends_cache[cache_key]
        
        print(f"🔍 Extracting legend from {legend_image_path.name}...")
        
        # Encode image
        img = Image.open(legend_image_path)
        
        # Convert to RGB if needed
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
        
        # Encode to base64
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.read()).decode('utf-8')
        
        # Call GPT-4 Vision with specialized prompt
        prompt = """You are analyzing a LEGEND PAGE (Zeichenerklärung/Legende) for a German land use plan (Flächennutzungsplan or Bebauungsplan).

**YOUR TASK: Extract ALL symbol definitions, color meanings, and abbreviations from this legend.**

Please provide a structured JSON response with the following format:
```json
{
  "document_type": "Flächennutzungsplan" or "Bebauungsplan",
  "document_name": "Full document name from the legend",
  
  "zoning_types": {
    "WA": "Wohngebiet (residential area)",
    "MI": "Mischgebiet (mixed use area)",
    "GE": "Gewerbegebiet (commercial area)",
    "... add all visible zoning abbreviations"
  },
  
  "color_meanings": {
    "red": "Wohnfläche (residential)",
    "yellow": "Gewerbefläche (commercial)",
    "green": "Grünfläche (green space)",
    "... add all color-coded areas"
  },
  
  "symbols": {
    "solid_line": "Plot boundary (Grundstücksgrenze)",
    "dashed_line": "Planning boundary (Plangrenze)",
    "... add all symbols visible"
  },
  
  "abbreviations": {
    "GRZ": "Grundflächenzahl (site coverage ratio)",
    "GFZ": "Geschossflächenzahl (floor area ratio)",
    "HBA": "Höhe baulicher Anlagen (building height)",
    "... add all abbreviations"
  },
  
  "special_designations": {
    "Any special markers, protected areas, restrictions, etc."
  }
}
```

**CRITICAL INSTRUCTIONS:**
1. Extract EVERY symbol, color, and abbreviation you can see
2. Read ALL text in the legend, including small print
3. Match colors to their meanings
4. Capture both German terms and English translations where clear
5. Be comprehensive - this legend will be used to interpret other plans

**Respond ONLY with valid JSON, no additional text.**"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a German urban planning expert analyzing legend pages. You extract structured information about symbols, colors, and zoning designations."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"  # High detail for legend extraction
                            }
                        }
                    ]
                }
            ],
            max_tokens=2000,
            temperature=0
        )
        
        result_text = response.choices[0].message.content
        
        # Parse JSON response
        try:
            # Strip markdown code blocks if present
            if result_text.startswith("```json"):
                result_text = result_text.replace("```json\n", "").replace("\n```", "").strip()
            elif result_text.startswith("```"):
                result_text = result_text.replace("```\n", "").replace("\n```", "").strip()
            
            legend_data = json.loads(result_text)
            
            # Cache the result
            self.legends_cache[cache_key] = legend_data
            self._save_cache()
            
            print(f"✅ Legend extracted successfully!")
            print(f"   Zoning types: {len(legend_data.get('zoning_types', {}))}")
            print(f"   Colors: {len(legend_data.get('color_meanings', {}))}")
            print(f"   Symbols: {len(legend_data.get('symbols', {}))}")
            
            return legend_data
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON response: {e}")
            print(f"Response: {result_text[:500]}")
            return {"error": "Failed to parse legend", "raw_response": result_text}
    
    def _save_cache(self):
        """Save legends cache to file"""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.legends_cache, f, indent=2, ensure_ascii=False)
    
    def get_legend_context(self, plan_type: str = "Flächennutzungsplan") -> str:
        """
        Get formatted legend context for use in plan analysis prompts
        
        Args:
            plan_type: Type of plan (Flächennutzungsplan or Bebauungsplan)
        
        Returns:
            Formatted string with legend information
        """
        relevant_legends = [
            legend for legend in self.legends_cache.values()
            if legend.get('document_type', '').lower() in plan_type.lower()
        ]
        
        if not relevant_legends:
            return "No legend information available."
        
        # Use the most comprehensive legend
        legend = max(relevant_legends, key=lambda x: len(str(x)))
        
        context = f"""**LEGEND REFERENCE for {plan_type}:**

**Zoning Types:**
"""
        for abbr, meaning in legend.get('zoning_types', {}).items():
            context += f"- {abbr}: {meaning}\n"
        
        context += "\n**Color Meanings:**\n"
        for color, meaning in legend.get('color_meanings', {}).items():
            context += f"- {color}: {meaning}\n"
        
        context += "\n**Common Abbreviations:**\n"
        for abbr, meaning in legend.get('abbreviations', {}).items():
            context += f"- {abbr}: {meaning}\n"
        
        return context


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    import sys
    
    api_key = os.getenv("OPENAI_API_KEY")
    extractor = LegendExtractor(api_key)
    
    # ✅ IMPROVED: Better auto-search for legend files
    if len(sys.argv) > 1:
        # Path provided as argument
        legend_path = Path(sys.argv[1])
    else:
        # Auto-search for legend files
        plans_dir = Path("data/raw/Landuse Plans")
        
        print("🔍 Searching for legend files...")
        
        # Search for fnp-61-anlage1-planzeichnung files (these are legends)
        legend_files = list(plans_dir.glob("**/fnp-61-anlage1-planzeichnung*.png"))
        
        if legend_files:
            # Prefer page_3 (the main legend) or page_1
            preferred = [f for f in legend_files if 'page_3' in f.name or 'page_1' in f.name]
            legend_path = preferred[0] if preferred else legend_files[0]
            print(f"✅ Found legend: {legend_path.name}")
            print(f"   Location: {legend_path.parent}")
        else:
            # Fallback: search for any file with legend keywords
            legend_keywords = ['anlage', 'legend', 'legende', 'zeichenerklärung']
            all_images = list(plans_dir.glob("**/*.png"))
            
            legend_files = [
                img for img in all_images
                if any(kw in img.name.lower() for kw in legend_keywords)
            ]
            
            if legend_files:
                legend_path = legend_files[0]
                print(f"✅ Using: {legend_path.name}")
            else:
                print("❌ No legend files found!")
                print("\nSearched for files containing: 'anlage', 'legend', 'legende'")
                print("\nUsage: python legend_extractor.py <path_to_legend_image>")
                sys.exit(1)
    
    if not legend_path.exists():
        print(f"❌ File not found: {legend_path}")
        sys.exit(1)
    
    print(f"\n📄 Extracting legend from: {legend_path.name}")
    print("="*60)
    
    legend_data = extractor.extract_legend(legend_path)
    
    print("\n" + "="*60)
    print("📋 EXTRACTED LEGEND DATA")
    print("="*60)
    print(json.dumps(legend_data, indent=2, ensure_ascii=False))
    
    # Save to a readable file as well
    output_file = Path("data/legend_extracted.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(legend_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Saved to: {output_file}")
    
    print("\n" + "="*60)
    print("📝 LEGEND CONTEXT FOR PROMPTS")
    print("="*60)
    print(extractor.get_legend_context("Flächennutzungsplan"))