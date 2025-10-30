"""
Evidence Retrieval System for VeritasAI
Hybrid retrieval combining local vector store + web search with reranking
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import hashlib


class Evidence(BaseModel):
    """Single piece of evidence"""
    id: str = Field(description="Unique evidence ID")
    content: str = Field(description="Evidence text content")
    source_url: str = Field(description="Source URL")
    source_name: str = Field(description="Name of the source")
    retrieval_score: float = Field(description="Initial retrieval score", ge=0, le=1)
    rerank_score: Optional[float] = Field(default=None, description="Reranked score", ge=0, le=1)
    published_date: Optional[str] = Field(default=None, description="Publication date")
    snippet: str = Field(default="", description="Brief snippet/summary")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    retrieval_method: str = Field(description="Method used: 'vector_store' or 'web_search'")


class EvidenceResult(BaseModel):
    """Result from evidence retrieval for a single claim"""
    claim: str = Field(description="The claim being verified")
    evidences: List[Evidence] = Field(description="Retrieved and ranked evidences")
    total_retrieved: int = Field(description="Total number of evidences retrieved before filtering")
    retrieval_timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class RetrievalConfig(BaseModel):
    """Configuration for retrieval system"""
    vector_store_top_k: int = Field(default=5, description="Top K from vector store")
    web_search_top_k: int = Field(default=5, description="Top K from web search")
    final_top_k: int = Field(default=3, description="Final top K after reranking")
    enable_vector_store: bool = Field(default=True, description="Use local vector store")
    enable_web_search: bool = Field(default=True, description="Use web search")
    enable_reranking: bool = Field(default=True, description="Enable reranking")
    min_relevance_score: float = Field(default=0.3, description="Minimum relevance threshold")


def generate_evidence_id(content: str, source_url: str) -> str:
    """Generate unique evidence ID from content and source"""
    hash_input = f"{content[:100]}{source_url}".encode('utf-8')
    return hashlib.md5(hash_input).hexdigest()[:12]


def deduplicate_evidences(evidences: List[Evidence]) -> List[Evidence]:
    """Remove duplicate evidences based on content similarity"""
    seen_ids = set()
    unique_evidences = []
    
    for evidence in evidences:
        # Simple deduplication by ID
        if evidence.id not in seen_ids:
            seen_ids.add(evidence.id)
            unique_evidences.append(evidence)
    
    return unique_evidences
