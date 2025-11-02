"""
Reranker for Evidence Relevance Scoring
Uses LLM-based reranking to score evidence relevance to claims
"""

from typing import List, Optional
import time
import hashlib
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from backend.retrieval_models import Evidence

load_dotenv()

# Rate limiting configuration
RERANK_DELAY = 2.5  # Delay between API calls in seconds
MAX_RETRIES = 3  # Maximum retries for failed calls
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"


class RelevanceScore(BaseModel):
    """Relevance score for a single evidence"""
    evidence_id: str = Field(description="Evidence ID")
    relevance_score: float = Field(description="Relevance score 0-1", ge=0, le=1)
    reasoning: str = Field(description="Brief reasoning for the score")
    supports_claim: bool = Field(description="Whether evidence supports the claim")
    contradicts_claim: bool = Field(description="Whether evidence contradicts the claim")


class RerankResult(BaseModel):
    """Result of reranking multiple evidences"""
    scores: List[RelevanceScore] = Field(description="Relevance scores for each evidence")


RERANK_PROMPT = """You are an expert fact-checker tasked with scoring the relevance of evidence to a claim.

Given a CLAIM and a piece of EVIDENCE, provide:
1. A relevance score from 0 to 1 (where 1 is highly relevant)
2. Brief reasoning for the score
3. Whether the evidence supports the claim
4. Whether the evidence contradicts the claim

Consider:
- How directly the evidence addresses the claim
- The credibility of the source
- Whether it provides verification or refutation
- Specificity and detail of the evidence

Be objective and analytical in your assessment.
"""


class EvidenceReranker:
    """Reranks evidence using LLM-based relevance scoring"""
    
    def __init__(self):
        """Initialize reranker with LLM"""
        # Use Gemini 2.0 Flash Lite for higher rate limits and lower cost
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",  # Lite version for higher quota
            temperature=0.1,
            max_retries=MAX_RETRIES,
            timeout=30
        )
        self.cache = {}  # Simple cache for rerank results
        self.last_api_call = 0  # Track last API call time for rate limiting
    
    def _rate_limit(self):
        """Enforce rate limiting between API calls"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        
        if time_since_last_call < RERANK_DELAY:
            sleep_time = RERANK_DELAY - time_since_last_call
            time.sleep(sleep_time)
        
        self.last_api_call = time.time()
    
    def _get_cache_key(self, claim: str, evidence_content: str) -> str:
        """Generate cache key for claim-evidence pair"""
        combined = f"{claim}|{evidence_content}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def rerank(self, claim: str, evidences: List[Evidence], top_k: int = 3) -> List[Evidence]:
        """
        Rerank evidences based on relevance to claim
        
        Args:
            claim: The claim being verified
            evidences: List of evidence to rerank
            top_k: Number of top evidences to return
            
        Returns:
            List of top-k reranked evidences
        """
        if not evidences:
            return []
        
        # Skip reranking if disabled or quota issues
        if not ENABLE_RERANKING:
            print("⚠️ Reranking disabled (set ENABLE_RERANKING=true to enable)")
            evidences.sort(key=lambda x: x.retrieval_score, reverse=True)
            return evidences[:top_k]
        
        try:
            # Score each evidence
            scored_evidences = []
            
            for i, evidence in enumerate(evidences):
                # Apply rate limiting between calls
                if i > 0:  # Skip delay for first call
                    self._rate_limit()
                
                relevance = self._score_evidence(claim, evidence)
                
                # Update evidence with rerank score
                evidence.rerank_score = relevance.relevance_score
                evidence.metadata.update({
                    "rerank_reasoning": relevance.reasoning,
                    "supports_claim": relevance.supports_claim,
                    "contradicts_claim": relevance.contradicts_claim
                })
                
                scored_evidences.append(evidence)
                
                # Print progress
                if (i + 1) % 3 == 0 or (i + 1) == len(evidences):
                    print(f"  Reranked {i + 1}/{len(evidences)} evidences...")
            
            # Sort by rerank score (descending)
            scored_evidences.sort(key=lambda x: x.rerank_score or 0, reverse=True)
            
            # Return top-k
            return scored_evidences[:top_k]
            
        except Exception as e:
            print(f"❌ Reranking failed: {e}")
            print("Falling back to retrieval scores...")
            # Fallback: sort by original retrieval score
            evidences.sort(key=lambda x: x.retrieval_score, reverse=True)
            return evidences[:top_k]
    
    def _score_evidence(self, claim: str, evidence: Evidence) -> RelevanceScore:
        """Score a single evidence against the claim"""
        
        # Check cache first
        cache_key = self._get_cache_key(claim, evidence.content)
        if cache_key in self.cache:
            cached_score = self.cache[cache_key]
            cached_score.evidence_id = evidence.id
            return cached_score
        
        prompt = f"""{RERANK_PROMPT}

CLAIM: {claim}

EVIDENCE:
Source: {evidence.source_name} ({evidence.source_url})
Content: {evidence.content[:500]}...

Provide your relevance assessment:
"""
        
        try:
            # Use structured output with retry
            response = self.llm.with_structured_output(RelevanceScore).invoke(prompt)
            response.evidence_id = evidence.id
            
            # Cache the result
            self.cache[cache_key] = response
            
            return response
            
        except Exception as e:
            print(f"⚠️ Failed to score evidence {evidence.id}: {e}")
            # Fallback: use retrieval score
            return RelevanceScore(
                evidence_id=evidence.id,
                relevance_score=evidence.retrieval_score,
                reasoning="Fallback to retrieval score due to scoring error",
                supports_claim=False,
                contradicts_claim=False
            )
    
    def batch_rerank(
        self,
        claim: str,
        evidences: List[Evidence],
        top_k: int = 3,
        batch_size: int = 10
    ) -> List[Evidence]:
        """
        Rerank evidences in batches (more efficient for large lists)
        
        Args:
            claim: The claim being verified
            evidences: List of evidence to rerank
            top_k: Number of top evidences to return
            batch_size: Maximum evidences to score per batch
            
        Returns:
            List of top-k reranked evidences
        """
        if len(evidences) <= batch_size:
            return self.rerank(claim, evidences, top_k)
        
        # Process in batches
        all_scored = []
        for i in range(0, len(evidences), batch_size):
            batch = evidences[i:i+batch_size]
            scored_batch = self.rerank(claim, batch, len(batch))
            all_scored.extend(scored_batch)
        
        # Sort all and return top-k
        all_scored.sort(key=lambda x: x.rerank_score or 0, reverse=True)
        return all_scored[:top_k]


# Singleton instance
_reranker = None

def get_reranker() -> EvidenceReranker:
    """Get or create reranker singleton"""
    global _reranker
    if _reranker is None:
        _reranker = EvidenceReranker()
    return _reranker
