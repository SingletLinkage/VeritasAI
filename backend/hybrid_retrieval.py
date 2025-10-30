"""
Hybrid Evidence Retrieval System
Combines local vector store + web search with LLM-based reranking
"""

from typing import List, Optional
from backend.retrieval_models import Evidence, EvidenceResult, RetrievalConfig, deduplicate_evidences
from backend.vector_store import get_vector_store_manager
from backend.web_search import get_web_search_agent
from backend.reranker import get_reranker


class HybridRetriever:
    """
    Hybrid evidence retrieval system
    
    Process:
    1. Retrieve from local vector store (top_k)
    2. Retrieve from web search (top_k)
    3. Deduplicate combined results
    4. Rerank using LLM
    5. Return top_k final evidences
    """
    
    def __init__(self, config: Optional[RetrievalConfig] = None):
        """
        Initialize hybrid retriever
        
        Args:
            config: Retrieval configuration
        """
        self.config = config or RetrievalConfig()
        
        # Initialize components
        self.vector_store = get_vector_store_manager() if self.config.enable_vector_store else None
        self.web_search = get_web_search_agent() if self.config.enable_web_search else None
        self.reranker = get_reranker() if self.config.enable_reranking else None
        
        print("✅ Hybrid Retriever initialized")
        print(f"   Vector Store: {'✓' if self.vector_store else '✗'}")
        print(f"   Web Search: {'✓' if self.web_search else '✗'}")
        print(f"   Reranking: {'✓' if self.reranker else '✗'}")
    
    def retrieve(self, claim: str) -> EvidenceResult:
        """
        Retrieve and rank evidence for a single claim
        
        Args:
            claim: The claim to verify
            
        Returns:
            EvidenceResult with ranked evidences
        """
        print(f"\n🔍 Retrieving evidence for: '{claim[:100]}...'")
        
        all_evidences = []
        
        # Step 1: Vector store retrieval
        if self.vector_store and self.config.enable_vector_store:
            print(f"   📚 Searching local vector store (top {self.config.vector_store_top_k})...")
            vector_results = self.vector_store.search(
                claim,
                top_k=self.config.vector_store_top_k
            )
            all_evidences.extend(vector_results)
            print(f"   ✓ Retrieved {len(vector_results)} from vector store")
        
        # Step 2: Web search retrieval
        if self.web_search and self.config.enable_web_search:
            print(f"   🌐 Searching web (top {self.config.web_search_top_k})...")
            web_results = self.web_search.search(
                claim,
                top_k=self.config.web_search_top_k
            )
            all_evidences.extend(web_results)
            print(f"   ✓ Retrieved {len(web_results)} from web search")
        
        total_retrieved = len(all_evidences)
        
        # Step 3: Deduplicate
        print(f"   🔄 Deduplicating {total_retrieved} evidences...")
        all_evidences = deduplicate_evidences(all_evidences)
        print(f"   ✓ {len(all_evidences)} unique evidences after deduplication")
        
        # Step 4: Filter by minimum relevance
        all_evidences = [
            e for e in all_evidences
            if e.retrieval_score >= self.config.min_relevance_score
        ]
        print(f"   ✓ {len(all_evidences)} evidences above relevance threshold ({self.config.min_relevance_score})")
        
        # Step 5: Rerank
        if self.reranker and self.config.enable_reranking and all_evidences:
            print(f"   🎯 Reranking evidences (selecting top {self.config.final_top_k})...")
            final_evidences = self.reranker.rerank(
                claim,
                all_evidences,
                top_k=self.config.final_top_k
            )
            print(f"   ✓ Reranking complete - {len(final_evidences)} top evidences selected")
        else:
            # Fallback: sort by retrieval score
            all_evidences.sort(key=lambda x: x.retrieval_score, reverse=True)
            final_evidences = all_evidences[:self.config.final_top_k]
            print(f"   ✓ Sorted by retrieval score - {len(final_evidences)} top evidences")
        
        result = EvidenceResult(
            claim=claim,
            evidences=final_evidences,
            total_retrieved=total_retrieved
        )
        
        print(f"   ✅ Evidence retrieval complete: {len(final_evidences)} final evidences")
        
        return result
    
    def retrieve_batch(self, claims: List[str]) -> List[EvidenceResult]:
        """
        Retrieve evidence for multiple claims
        
        Args:
            claims: List of claims to verify
            
        Returns:
            List of EvidenceResult objects
        """
        print(f"\n📦 Batch retrieval for {len(claims)} claims")
        
        results = []
        for idx, claim in enumerate(claims, 1):
            print(f"\n[{idx}/{len(claims)}]")
            result = self.retrieve(claim)
            results.append(result)
        
        print(f"\n✅ Batch retrieval complete - {len(results)} claims processed")
        return results
    
    def update_config(self, config: RetrievalConfig):
        """Update retrieval configuration"""
        self.config = config
        print("✅ Retrieval configuration updated")


# Singleton instance
_hybrid_retriever = None

def get_hybrid_retriever(config: Optional[RetrievalConfig] = None) -> HybridRetriever:
    """Get or create hybrid retriever singleton"""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever(config)
    elif config is not None:
        _hybrid_retriever.update_config(config)
    return _hybrid_retriever


# Convenience function for quick retrieval
def retrieve_evidence(
    claim: str,
    vector_top_k: int = 5,
    web_top_k: int = 5,
    final_top_k: int = 3
) -> EvidenceResult:
    """
    Quick evidence retrieval with default configuration
    
    Args:
        claim: Claim to verify
        vector_top_k: Top K from vector store
        web_top_k: Top K from web search
        final_top_k: Final top K after reranking
        
    Returns:
        EvidenceResult with ranked evidences
    """
    config = RetrievalConfig(
        vector_store_top_k=vector_top_k,
        web_search_top_k=web_top_k,
        final_top_k=final_top_k
    )
    retriever = get_hybrid_retriever(config)
    return retriever.retrieve(claim)
