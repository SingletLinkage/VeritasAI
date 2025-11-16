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
    evidence_id: str = Field(description="Evidence ID (number)")
    relevance_score: float = Field(description="Relevance score 0-1", ge=0, le=1)
    reasoning: str = Field(description="Brief reasoning for the score")
    supports_claim: bool = Field(description="Whether evidence supports the claim")
    contradicts_claim: bool = Field(description="Whether evidence contradicts the claim")


class BatchRerankResult(BaseModel):
    """Result of reranking multiple evidences in a single batch"""
    scores: List[RelevanceScore] = Field(description="Relevance scores for each evidence, in the same order as input")


RERANK_PROMPT = """You are an expert fact-checker tasked with scoring the relevance of multiple pieces of evidence to a claim.

Given a CLAIM and a LIST of EVIDENCE items, provide a relevance score for EACH evidence item.

For each evidence, provide:
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

Return the scores in the SAME ORDER as the evidence items provided.
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
        Rerank evidences based on relevance to claim using BATCH processing
        
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
            # Score ALL evidences in a SINGLE batch request
            print(f"  🔄 Batch reranking {len(evidences)} evidences in 1 API call...")
            batch_result = self._batch_score_evidences(claim, evidences)
            
            # Update evidences with rerank scores
            for evidence, relevance in zip(evidences, batch_result.scores):
                evidence.rerank_score = relevance.relevance_score
                evidence.metadata.update({
                    "rerank_reasoning": relevance.reasoning,
                    "supports_claim": relevance.supports_claim,
                    "contradicts_claim": relevance.contradicts_claim
                })
            
            # Sort by rerank score (descending)
            evidences.sort(key=lambda x: x.rerank_score or 0, reverse=True)
            
            print(f"  ✅ Batch reranking complete!")
            
            # Return top-k
            return evidences[:top_k]
            
        except Exception as e:
            print(f"❌ Batch reranking failed: {e}")
            print("Falling back to retrieval scores...")
            # Fallback: sort by original retrieval score
            evidences.sort(key=lambda x: x.retrieval_score, reverse=True)
            return evidences[:top_k]
    
    def _batch_score_evidences(self, claim: str, evidences: List[Evidence]) -> BatchRerankResult:
        """Score ALL evidences in a single batch API call"""
        
        # Build the batch prompt with numbered evidence
        evidence_text = ""
        for i, evidence in enumerate(evidences):
            evidence_text += f"""
--- EVIDENCE {i} ---
ID: {i}
Source: {evidence.source_name} ({evidence.source_url})
Content: {evidence.content[:400]}...

"""
        
        prompt = f"""{RERANK_PROMPT}

CLAIM: {claim}

EVIDENCE LIST ({len(evidences)} items):
{evidence_text}

Provide relevance assessments for ALL {len(evidences)} evidence items in order (0 to {len(evidences)-1}):
"""
        
        try:
            # Single API call for all evidences
            response = self.llm.with_structured_output(BatchRerankResult).invoke(prompt)
            
            # Validate we got the right number of scores
            if len(response.scores) != len(evidences):
                print(f"⚠️ Expected {len(evidences)} scores, got {len(response.scores)}. Padding...")
                # Pad with fallback scores if needed
                while len(response.scores) < len(evidences):
                    idx = len(response.scores)
                    response.scores.append(RelevanceScore(
                        evidence_id=str(idx),
                        relevance_score=evidences[idx].retrieval_score,
                        reasoning="Fallback score - not returned by LLM",
                        supports_claim=False,
                        contradicts_claim=False
                    ))
            
            return response
            
        except Exception as e:
            print(f"⚠️ Batch scoring failed: {e}")
            # Fallback: create scores from retrieval scores
            return BatchRerankResult(
                scores=[
                    RelevanceScore(
                        evidence_id=str(i),
                        relevance_score=evidence.retrieval_score,
                        reasoning="Fallback to retrieval score due to API error",
                        supports_claim=False,
                        contradicts_claim=False
                    )
                    for i, evidence in enumerate(evidences)
                ]
            )
    
    def batch_rerank(
        self,
        claim: str,
        evidences: List[Evidence],
        top_k: int = 3,
        batch_size: int = 20  # Increased from 10 since we're doing batch calls
    ) -> List[Evidence]:
        """
        Rerank evidences in batches (for very large lists)
        
        Args:
            claim: The claim being verified
            evidences: List of evidence to rerank
            top_k: Number of top evidences to return
            batch_size: Maximum evidences to score per batch (default 20)
            
        Returns:
            List of top-k reranked evidences
        """
        if len(evidences) <= batch_size:
            # Single batch - use main rerank method
            return self.rerank(claim, evidences, top_k)
        
        # Process in multiple batches for very large lists
        print(f"  📦 Processing {len(evidences)} evidences in batches of {batch_size}...")
        all_scored = []
        for i in range(0, len(evidences), batch_size):
            batch = evidences[i:i+batch_size]
            print(f"  Batch {i//batch_size + 1}/{(len(evidences)-1)//batch_size + 1}...")
            scored_batch = self.rerank(claim, batch, len(batch))
            all_scored.extend(scored_batch)
            
            # Small delay between batches to avoid rate limits
            if i + batch_size < len(evidences):
                time.sleep(1)
        
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
