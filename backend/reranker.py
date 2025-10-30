"""
Reranker for Evidence Relevance Scoring
Uses LLM-based reranking to score evidence relevance to claims
"""

from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from backend.retrieval_models import Evidence

load_dotenv()


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
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.1
        )
    
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
        
        try:
            # Score each evidence
            scored_evidences = []
            
            for evidence in evidences:
                relevance = self._score_evidence(claim, evidence)
                
                # Update evidence with rerank score
                evidence.rerank_score = relevance.relevance_score
                evidence.metadata.update({
                    "rerank_reasoning": relevance.reasoning,
                    "supports_claim": relevance.supports_claim,
                    "contradicts_claim": relevance.contradicts_claim
                })
                
                scored_evidences.append(evidence)
            
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
        
        prompt = f"""{RERANK_PROMPT}

CLAIM: {claim}

EVIDENCE:
Source: {evidence.source_name} ({evidence.source_url})
Content: {evidence.content}

Provide your relevance assessment:
"""
        
        try:
            # Use structured output
            response = self.llm.with_structured_output(RelevanceScore).invoke(prompt)
            response.evidence_id = evidence.id
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
