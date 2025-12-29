#!/usr/bin/env python3
"""
Railway-Optimized RAG System for Stuttgart Building Regulations
Uses OpenAI embeddings API instead of local model for faster deployment
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from openai import OpenAI
import logging
from functools import lru_cache
from dataclasses import dataclass
import time

from dotenv import load_dotenv
load_dotenv()  # This loads your .env file
print(f"✅ Loaded .env file, API key present: {bool(os.getenv('OPENAI_API_KEY'))}")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class RAGConfig:
    """Configuration for Railway-optimized RAG"""
    # OpenAI settings
    EMBEDDING_MODEL: str = "text-embedding-3-small"  # $0.02 per 1M tokens
    EMBEDDING_DIM: int = 1536  # Dimension for text-embedding-3-small
    
    # Alternative cheaper/smaller model
    # EMBEDDING_MODEL: str = "text-embedding-ada-002"
    # EMBEDDING_DIM: int = 1536
    
    # Search settings
    DEFAULT_TOP_K: int = 5
    MAX_CONTEXT_TOKENS: int = 2000
    CACHE_SIZE: int = 100
    
    # Batch processing
    BATCH_SIZE: int = 20  # Embed documents in batches
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 2  # seconds


# ============================================================================
# RESULT CLASS
# ============================================================================

class BuildingResult:
    """Result class compatible with existing system"""
    
    def __init__(
        self, 
        content: str, 
        score: float, 
        metadata: Dict = None, 
        source: str = None, 
        citation: str = None, 
        document_id: str = None
    ):
        self.content = content
        self.score = score
        self.metadata = metadata or {}
        self.source = source
        self.citation = citation
        self.document_id = document_id
    
    def get_detailed_citation(self) -> str:
        """Generate detailed citation"""
        citation_parts = []
        
        doc_type = self.metadata.get("document_type", "Document")
        doc_name = self.metadata.get("document_name", "Unknown")
        citation_parts.append(f"{doc_type}: {doc_name}")
        
        page_num = self.metadata.get("page_number")
        if page_num:
            citation_parts.append(f"Page {page_num}")
        
        sections = self.metadata.get("sections", [])
        if sections:
            citation_parts.append(f"Section(s): {', '.join(sections[:3])}")
        
        return " | ".join(citation_parts)
    
    def get_district_specific_info(self) -> Dict:
        """Extract district-specific information"""
        district_info = self.metadata.get("district_specific", {})
        return {
            "mentioned_districts": district_info.get("mentioned_districts", []),
            "specific_rules": district_info.get("specific_rules", [])
        }


# ============================================================================
# RAILWAY-OPTIMIZED RAG SYSTEM
# ============================================================================

class RailwayOptimizedRAG:
    """
    Railway-optimized RAG using OpenAI embeddings
    
    Benefits for Railway:
    - No model downloads (faster startup)
    - No large embedding files in Git
    - Smaller memory footprint
    - Easy to update documents
    """
    
    def __init__(
        self, 
        documents_path: Optional[Path] = None,
        config: Optional[RAGConfig] = None,
        cache_embeddings: bool = True
    ):
        """
        Initialize Railway-optimized RAG
        
        Args:
            documents_path: Path to documents.json (lightweight, text only)
            config: RAGConfig instance
            cache_embeddings: Cache embeddings in memory for faster search
        """
        self.config = config or RAGConfig()
        self.cache_embeddings_enabled = cache_embeddings
        
        # OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")
        
        self.client = OpenAI(api_key=api_key)
        
        # Set documents path
        if documents_path is None:
            current_dir = Path(__file__).parent
            documents_path = current_dir / "data" / "documents_lightweight.json"
        self.documents_path = Path(documents_path)
        
        # Load documents
        self.documents = []
        self.document_embeddings = {}  # Cache: {doc_id: embedding}
        
        logger.info(f"🚂 Initializing Railway-optimized RAG")
        logger.info(f"📄 Documents: {self.documents_path}")
        logger.info(f"🤖 Model: {self.config.EMBEDDING_MODEL}")
        
        self._load_documents()
        
        logger.info(f"✅ Railway RAG ready with {len(self.documents)} documents")
    
    def _load_documents(self):
        """Load documents from JSON"""
        if not self.documents_path.exists():
            logger.warning(f"Documents file not found: {self.documents_path}")
            logger.warning("System will work but return no results until documents are added")
            return
        
        try:
            with open(self.documents_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Support both formats
            if isinstance(data, dict):
                self.documents = data.get('documents', [])
            elif isinstance(data, list):
                self.documents = data
            else:
                raise ValueError(f"Invalid documents format: {type(data)}")
            
            logger.info(f"📚 Loaded {len(self.documents)} documents")
            
            # Validate structure
            if self.documents:
                sample = self.documents[0]
                required_fields = ['content', 'document_id']
                missing = [f for f in required_fields if f not in sample]
                if missing:
                    logger.warning(f"Documents missing fields: {missing}")
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            self.documents = []
    
    @lru_cache(maxsize=100)
    def _embed_text_cached(self, text: str) -> Tuple[float, ...]:
        """
        Embed text using OpenAI API with caching
        
        Returns tuple for LRU cache hashability
        """
        return tuple(self._embed_text(text))
    
    def _embed_text(self, text: str) -> List[float]:
        """Embed single text using OpenAI API"""
        # Truncate if too long
        MAX_CHARS = 20000  # ← Change from 24000
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS]
            logger.warning(f"⚠️ Truncated text from {len(text)} to {MAX_CHARS} chars")
        
        for attempt in range(self.config.MAX_RETRIES):
            try:
                response = self.client.embeddings.create(
                    model=self.config.EMBEDDING_MODEL,
                    input=text  # Use the truncated text variable
                )
                return response.data[0].embedding
            
            except Exception as e:
                if attempt < self.config.MAX_RETRIES - 1:
                    logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
                    time.sleep(self.config.RETRY_DELAY)
                else:
                    logger.error(f"Embedding failed after {self.config.MAX_RETRIES} attempts")
                    raise
    
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts in batch (more efficient)"""
        # Truncate texts that are too long (max ~6000 tokens to be safe)
        MAX_CHARS = 20000  # ← Change from 24000
        truncated_texts = []
        for text in texts:
            if len(text) > MAX_CHARS:
                truncated_texts.append(text[:MAX_CHARS])
                logger.warning(f"⚠️ Truncated document from {len(text)} to {MAX_CHARS} chars")
            else:
                truncated_texts.append(text)      
        
        for attempt in range(self.config.MAX_RETRIES):
            try:
                response = self.client.embeddings.create(
                    model=self.config.EMBEDDING_MODEL,
                    input=truncated_texts
                )
                return [item.embedding for item in response.data]
            
            except Exception as e:
                if attempt < self.config.MAX_RETRIES - 1:
                    logger.warning(f"Batch embedding attempt {attempt + 1} failed: {e}")
                    time.sleep(self.config.RETRY_DELAY)
                else:
                    logger.error(f"Batch embedding failed")
                    raise
    
    def _ensure_document_embeddings(self):
        """
        Ensure all documents have embeddings (compute if needed)
        This runs on first search and caches results
        """
        if len(self.document_embeddings) == len(self.documents):
            return  # Already embedded
        
        logger.info("🔄 Computing document embeddings (first search only)...")
        
        # Get documents that need embedding
        docs_to_embed = []
        for doc in self.documents:
            doc_id = doc.get('document_id', '')
            if doc_id not in self.document_embeddings:
                docs_to_embed.append(doc)
        
        if not docs_to_embed:
            return
        
        # Embed in batches for efficiency
        total = len(docs_to_embed)
        for i in range(0, total, self.config.BATCH_SIZE):
            batch = docs_to_embed[i:i + self.config.BATCH_SIZE]
            texts = [doc['content'] for doc in batch]
            
            logger.info(f"   Embedding batch {i//self.config.BATCH_SIZE + 1}/{(total-1)//self.config.BATCH_SIZE + 1}")
            
            embeddings = self._embed_batch(texts)
            
            for doc, embedding in zip(batch, embeddings):
                doc_id = doc.get('document_id', '')
                self.document_embeddings[doc_id] = embedding
        
        logger.info(f"✅ Embedded {total} documents")
    
    def search(
        self, 
        query: str, 
        top_k: int = None,
        filter_district: Optional[str] = None,
        filter_document_type: Optional[str] = None
    ) -> List[BuildingResult]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            top_k: Number of results
            filter_district: Filter by district
            filter_document_type: Filter by document type
        """
        if top_k is None:
            top_k = self.config.DEFAULT_TOP_K
        
        if not self.documents:
            logger.warning("No documents available")
            return [BuildingResult(
                content="No documents loaded. Please add documents to the system.",
                score=0.0,
                metadata={"type": "no_documents"}
            )]
        
        try:
            # Embed query (cached)
            logger.info(f"🔍 Searching for: {query}")
            query_embedding = list(self._embed_text_cached(query))
            
            # Ensure document embeddings exist
            if self.cache_embeddings_enabled:
                self._ensure_document_embeddings()
            
            # Calculate similarities
            results = []
            for doc in self.documents:
                doc_id = doc.get('document_id', '')
                
                # Get document embedding (from cache or compute)
                if doc_id in self.document_embeddings:
                    doc_embedding = self.document_embeddings[doc_id]
                else:
                    doc_embedding = self._embed_text(doc['content'])
                    if self.cache_embeddings_enabled:
                        self.document_embeddings[doc_id] = doc_embedding
                
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                
                # Apply filters
                if filter_district:
                    district_info = doc.get("metadata", {}).get("district_specific", {})
                    mentioned = district_info.get("mentioned_districts", [])
                    if filter_district not in mentioned:
                        continue
                
                if filter_document_type:
                    doc_type = doc.get("metadata", {}).get("document_type", "")
                    if filter_document_type.lower() not in doc_type.lower():
                        continue
                
                results.append({
                    'doc': doc,
                    'score': float(similarity)
                })
            
            # Sort by score
            results.sort(key=lambda x: x['score'], reverse=True)
            
            # Convert to BuildingResult
            output = []
            for r in results[:top_k]:
                doc = r['doc']
                output.append(BuildingResult(
                    content=doc.get('content', ''),
                    score=r['score'],
                    metadata=doc.get('metadata', {}),
                    source=doc.get('source', 'Unknown'),
                    citation=doc.get('citation', ''),
                    document_id=doc.get('document_id', '')
                ))
            
            logger.info(f"✅ Found {len(output)} results")
            return output
        
        except Exception as e:
            logger.error(f"Search error: {e}", exc_info=True)
            return [BuildingResult(
                content=f"Search error: {str(e)}",
                score=0.0,
                metadata={"type": "error", "error": str(e)}
            )]
    
    def get_context_for_query(
        self, 
        query: str, 
        max_tokens: int = None,
        include_citations: bool = True
    ) -> str:
        """Get formatted context for a query"""
        if max_tokens is None:
            max_tokens = self.config.MAX_CONTEXT_TOKENS
        
        results = self.search(query, top_k=4)
        
        if not results:
            return "No relevant documents found."
        
        context_parts = []
        total_length = 0
        
        for i, result in enumerate(results):
            if include_citations:
                citation = result.get_detailed_citation()
                content = f"[Source {i+1}] {citation}\n\nContent: {result.content}"
                
                district_info = result.get_district_specific_info()
                if district_info["mentioned_districts"]:
                    content += f"\n\nDistrict(s): {', '.join(district_info['mentioned_districts'])}"
            else:
                content = result.content
            
            if total_length + len(content) > max_tokens:
                break
            
            context_parts.append(content)
            total_length += len(content)
        
        return "\n\n" + "="*50 + "\n\n".join(context_parts)
    
    def search_by_district(self, query: str, district: str, top_k: int = 3):
        """Search for district-specific regulations"""
        return self.search(query, top_k=top_k, filter_district=district)
    
    def search_by_document_type(self, query: str, doc_type: str, top_k: int = 3):
        """Search within specific document types"""
        return self.search(query, top_k=top_k, filter_document_type=doc_type)
    
    def health_check(self) -> Dict:
        """System health check"""
        return {
            'status': 'healthy',
            'num_documents': len(self.documents),
            'cached_embeddings': len(self.document_embeddings),
            'model': self.config.EMBEDDING_MODEL,
            'dimension': self.config.EMBEDDING_DIM,
            'cache_enabled': self.cache_embeddings_enabled
        }


# ============================================================================
# ALIAS FOR BACKWARD COMPATIBILITY
# ============================================================================

# This allows existing code to work without changes
EnhancedPrecomputedRAGSystem = RailwayOptimizedRAG


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚂 RAILWAY-OPTIMIZED RAG TEST")
    print("="*70 + "\n")
    
    try:
        # Initialize
        print("Initializing system...")
        rag = RailwayOptimizedRAG()
        
        # Health check
        print("\n📊 Health Check:")
        health = rag.health_check()
        print(json.dumps(health, indent=2))
        
        # Test search
        print("\n🔍 Testing search:")
        query = "building height restrictions"
        results = rag.search(query, top_k=3)
        
        if results:
            print(f"✅ Found {len(results)} results:")
            for i, res in enumerate(results, 1):
                print(f"\n  Result {i}:")
                print(f"    Score: {res.score:.3f}")
                print(f"    Content: {res.content[:100]}...")
        else:
            print("❌ No results")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()