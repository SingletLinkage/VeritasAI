"""
Web Search Agent for Real-time Evidence Retrieval
Uses Google Search API (SerpAPI) for fact-checking
"""

from typing import List, Optional
import os
from dotenv import load_dotenv
import requests
from datetime import datetime

from backend.retrieval_models import Evidence, generate_evidence_id

load_dotenv()


class WebSearchAgent:
    """Web search agent using SerpAPI for Google Search"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize web search agent
        
        Args:
            api_key: SerpAPI key (optional, will use env var if not provided)
        """
        self.api_key = api_key or os.getenv("SERPAPI_API_KEY")
        self.base_url = "https://serpapi.com/search"
        
        # Fallback to mock search if no API key
        self.use_mock = not self.api_key
        
        if self.use_mock:
            print("⚠️ No SERPAPI_API_KEY found - using mock web search")
    
    def search(self, query: str, top_k: int = 5) -> List[Evidence]:
        """
        Search the web for evidence related to the claim
        
        Args:
            query: Search query (claim to verify)
            top_k: Number of results to return
            
        Returns:
            List of Evidence objects
        """
        if self.use_mock:
            return self._mock_search(query, top_k)
        
        try:
            # Enhance query for fact-checking
            fact_check_query = f"{query} fact check verification evidence"
            
            params = {
                "q": fact_check_query,
                "api_key": self.api_key,
                "num": top_k,
                "engine": "google"
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            return self._parse_search_results(data, top_k)
            
        except Exception as e:
            print(f"❌ Web search failed: {e}")
            print("Falling back to mock search...")
            return self._mock_search(query, top_k)
    
    def _parse_search_results(self, data: dict, top_k: int) -> List[Evidence]:
        """Parse SerpAPI response into Evidence objects"""
        evidences = []
        
        organic_results = data.get("organic_results", [])[:top_k]
        
        for idx, result in enumerate(organic_results):
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            link = result.get("link", "")
            
            # Combine title and snippet for content
            content = f"{title}. {snippet}"
            
            # Calculate relevance score (simple heuristic based on position)
            relevance_score = 1.0 - (idx * 0.1)  # First result gets 1.0, second 0.9, etc.
            
            evidence = Evidence(
                id=generate_evidence_id(content, link),
                content=content,
                source_url=link,
                source_name=self._extract_domain(link),
                retrieval_score=max(relevance_score, 0.5),  # Minimum 0.5
                published_date=result.get("date"),
                snippet=snippet,
                metadata={
                    "position": idx + 1,
                    "search_engine": "google"
                },
                retrieval_method="web_search"
            )
            evidences.append(evidence)
        
        return evidences
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain name from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return "web_source"
    
    def _mock_search(self, query: str, top_k: int) -> List[Evidence]:
        """Mock web search for testing without API key"""
        
        # Generate mock results based on query keywords
        mock_results = []
        
        # Common fact-checking sources
        fact_check_sources = [
            {
                "domain": "snopes.com",
                "name": "Snopes",
                "reliability": 0.95
            },
            {
                "domain": "factcheck.org",
                "name": "FactCheck.org",
                "reliability": 0.93
            },
            {
                "domain": "politifact.com",
                "name": "PolitiFact",
                "reliability": 0.92
            },
            {
                "domain": "reuters.com/fact-check",
                "name": "Reuters Fact Check",
                "reliability": 0.94
            },
            {
                "domain": "apnews.com/ap-fact-check",
                "name": "AP Fact Check",
                "reliability": 0.93
            }
        ]
        
        for idx, source in enumerate(fact_check_sources[:top_k]):
            content = f"Fact-check analysis of: '{query}'. {source['name']} investigates the claim and provides detailed verification with supporting evidence and expert sources."
            
            evidence = Evidence(
                id=generate_evidence_id(content, f"https://{source['domain']}"),
                content=content,
                source_url=f"https://{source['domain']}/fact-check/{query.replace(' ', '-')[:50]}",
                source_name=source['name'],
                retrieval_score=source['reliability'] * (1.0 - idx * 0.05),
                published_date=datetime.now().strftime("%Y-%m-%d"),
                snippet=content[:150] + "...",
                metadata={
                    "position": idx + 1,
                    "search_engine": "google",
                    "mock_data": True
                },
                retrieval_method="web_search"
            )
            mock_results.append(evidence)
        
        return mock_results


# Singleton instance
_web_search_agent = None

def get_web_search_agent() -> WebSearchAgent:
    """Get or create web search agent singleton"""
    global _web_search_agent
    if _web_search_agent is None:
        _web_search_agent = WebSearchAgent()
    return _web_search_agent
