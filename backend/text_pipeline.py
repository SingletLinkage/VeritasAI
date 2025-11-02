from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.models import ClaimList, FusedClaimList, TranslationResult
from backend.prompts import CLAIM_EXTRACTION_PROMPT, FUSION_PROMPT, TRANSLATION_PROMPT
from backend.hybrid_retrieval import get_hybrid_retriever
from backend.retrieval_models import RetrievalConfig
from backend.explainability import explain_simply
from dotenv import load_dotenv
from typing import TypedDict, List, Optional, Dict, Any
from langdetect import detect, LangDetectException
from pydantic import BaseModel, Field

load_dotenv()

# === Verdict Model
class TextVerdict(BaseModel):
    """Final verdict for text-based fact-checking"""
    verdict: str = Field(description="Overall verdict: LIKELY_TRUE, LIKELY_FALSE, MISLEADING, or INSUFFICIENT")
    confidence: float = Field(description="Confidence score 0-1", ge=0, le=1)
    reasoning: str = Field(description="Detailed reasoning for the verdict")
    red_flags: List[str] = Field(description="List of red flags or concerns identified")
    recommendation: str = Field(description="Actionable recommendation for the user")

# === 1. Define the State Schema
class GraphState(TypedDict):
    content: str
    translated_text: Optional[str]
    source_language: Optional[str]
    needs_translation: Optional[bool]
    claims: ClaimList
    fused_claims: FusedClaimList
    evidence_results: List[Dict[str, Any]]  # List of EvidenceResult dicts
    evidence_links: List[str]
    verdict: Optional[TextVerdict]  # Final verdict
    easy_explain: Optional[Dict[str, Any]]  # Simple explanation for older users

# === 2. Define the model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

translation_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
)


# === 3. Define Agents
def detect_language(state: GraphState) -> dict:
    """Fast language detection using classical NLP"""
    content = state["content"]
    
    try:
        detected_lang = detect(content)
        needs_translation = detected_lang != 'en'
    except LangDetectException:
        # If detection fails (e.g., very short text), assume English
        detected_lang = 'en'
        needs_translation = False
    
    print(f"🔍 Detected language: {detected_lang} | Translation needed: {needs_translation}")
    
    return {
        "source_language": detected_lang,
        "needs_translation": needs_translation
    }

def translation_agent(state: GraphState) -> dict:
    """Translate content to English and update state"""
    content = state["content"]
    print(f"🌐 Translating from {state['source_language']} to English...")
    
    prompt = TRANSLATION_PROMPT + f"\nInput:\n{content}"
    
    response = translation_model.with_structured_output(TranslationResult).invoke(prompt)
    
    # Return state updates
    return {
        "translated_text": response.translated_text,
        "source_language": response.source_language
    }

def skip_translation(state: GraphState) -> dict:
    """Skip translation for English content"""
    print(f"⏩ Skipping translation - content already in English")
    
    # Use original content as translated text
    return {
        "translated_text": state["content"]
    }

def extract_claim(state: GraphState) -> dict:
    """Extract claims from content and update state"""
    # Use translated text if available, otherwise use original content
    content = state.get("translated_text", state["content"])
    prompt = CLAIM_EXTRACTION_PROMPT + f"\n\nContent:\n{content}\n\nExtracted Claims:"
    
    response = llm.with_structured_output(ClaimList).invoke(prompt)
    
    # Return state updates
    return {"claims": response}

def fusion_agent(state: GraphState) -> dict:
    """Fuse claims and update state"""
    claims = state["claims"]
    
    # Format claims for the prompt
    claims_text = "\n".join([f"- {claim.statement}" for claim in claims.claims])
    prompt = FUSION_PROMPT + f"\n\nClaims:\n{claims_text}\n\nFused Claims:"
    
    response = llm.with_structured_output(FusedClaimList).invoke(prompt)
    
    # Return state updates
    return {"fused_claims": response}

def retrieval_tool(state: GraphState) -> dict:
    """Retrieve evidence for fused claims using hybrid retrieval"""
    import os
    
    fused_claims = state["fused_claims"]
    
    print(f"\n📚 Retrieving evidence for {len(fused_claims.fused_claims)} claims...")
    
    # Check environment variable for reranking
    enable_reranking = os.environ.get("ENABLE_RERANKING", "false").lower() == "true"
    
    # Initialize hybrid retriever
    config = RetrievalConfig(
        vector_store_top_k=5,
        web_search_top_k=5,
        final_top_k=3,
        enable_vector_store=True,
        enable_web_search=True,
        enable_reranking=enable_reranking
    )
    retriever = get_hybrid_retriever(config)
    
    # Retrieve evidence for each claim
    evidence_results = []
    all_evidence_links = []
    
    for claim in fused_claims.fused_claims:
        print(f"\n  🔍 Claim: {claim.fused_statement}")
        
        # Retrieve evidence
        result = retriever.retrieve(claim.fused_statement)
        
        # Convert to dict for state storage
        result_dict = {
            "claim": result.claim,
            "evidences": [
                {
                    "id": e.id,
                    "content": e.content,
                    "source_url": e.source_url,
                    "source_name": e.source_name,
                    "retrieval_score": e.retrieval_score,
                    "rerank_score": e.rerank_score,
                    "snippet": e.snippet,
                    "metadata": e.metadata
                }
                for e in result.evidences
            ],
            "total_retrieved": result.total_retrieved
        }
        evidence_results.append(result_dict)
        
        # Collect all evidence URLs
        for evidence in result.evidences:
            all_evidence_links.append(evidence.source_url)
        
        # Update claim with evidence links
        claim.evidence_links = [e.source_url for e in result.evidences]
        
        print(f"  ✅ Found {len(result.evidences)} evidences (from {result.total_retrieved} retrieved)")
    
    print(f"\n✅ Evidence retrieval complete - {len(all_evidence_links)} total evidences")
    
    # Return state updates
    return {
        "fused_claims": fused_claims,
        "evidence_results": evidence_results,
        "evidence_links": all_evidence_links
    }

def generate_verdict(state: GraphState) -> dict:
    """Generate final verdict based on claims and evidence"""
    claims = state.get("fused_claims", state.get("claims"))
    evidence_results = state.get("evidence_results", [])
    original_content = state.get("content", "")
    
    print(f"\n⚖️ Generating final verdict...")
    
    # Format claims and evidence for the prompt
    claims_text = "\n".join([
        f"- {claim.fused_statement if hasattr(claim, 'fused_statement') else claim.statement}"
        for claim in (claims.fused_claims if hasattr(claims, 'fused_claims') else claims.claims)
    ])
    
    evidence_summary = []
    for result in evidence_results:
        claim_text = result['claim']
        evidences_text = "\n  ".join([
            f"• {e['source_name']}: {e['content'][:200]}..."
            for e in result['evidences'][:2]  # Top 2 evidences per claim
        ])
        evidence_summary.append(f"Claim: {claim_text}\n  {evidences_text}")
    
    evidence_text = "\n\n".join(evidence_summary) if evidence_summary else "No evidence retrieved"
    
    prompt = f"""You are an expert fact-checker. Based on the original content, extracted claims, and retrieved evidence, provide a comprehensive verdict.

ORIGINAL CONTENT:
{original_content}

EXTRACTED CLAIMS:
{claims_text}

EVIDENCE:
{evidence_text}

Provide your assessment with:
1. **Overall Verdict**: Choose from:
   - LIKELY_TRUE: Strong evidence supports the claims
   - LIKELY_FALSE: Strong evidence contradicts the claims
   - MISLEADING: Claims are partially true but lack context or are exaggerated
   - INSUFFICIENT: Not enough evidence to make a determination

2. **Confidence**: Score from 0 to 1 indicating how confident you are in the verdict

3. **Reasoning**: Detailed explanation of your verdict based on the evidence

4. **Red Flags**: List any concerning elements (e.g., lack of sources, contradictory evidence, logical fallacies)

5. **Recommendation**: Actionable advice for the user (e.g., "Verify with additional sources", "Content appears credible")

Be thorough, objective, and consider the quality and credibility of the evidence.
"""
    
    try:
        response = llm.with_structured_output(TextVerdict).invoke(prompt)
        print(f"  ✅ Verdict: {response.verdict} (Confidence: {response.confidence:.1%})")
        return {"verdict": response}
    except Exception as e:
        print(f"  ⚠️ Failed to generate verdict: {e}")
        # Fallback verdict
        return {
            "verdict": TextVerdict(
                verdict="INSUFFICIENT",
                confidence=0.5,
                reasoning="Unable to generate automated verdict due to processing error. Please review the evidence manually.",
                red_flags=["Automated analysis incomplete"],
                recommendation="Manually review the evidence and claims provided above."
            )
        }

def generate_simple_explanation(state: GraphState) -> dict:
    """
    Generate simple, accessible explanation for older users with limited digital literacy
    ("Explain Like I'm 60")
    """
    verdict = state.get("verdict")
    if not verdict:
        print("  ⚠️ No verdict available, skipping simple explanation")
        return {"easy_explain": None}
    
    # Get the first claim for context
    claims = state.get("fused_claims", state.get("claims"))
    first_claim = ""
    if claims:
        claim_list = claims.fused_claims if hasattr(claims, 'fused_claims') else claims.claims
        if claim_list:
            first_claim = claim_list[0].fused_statement if hasattr(claim_list[0], 'fused_statement') else claim_list[0].statement
    
    # Get user's language
    language = state.get("source_language", "en")
    
    print(f"\n💡 Generating simple explanation (language: {language})...")
    
    try:
        # Generate simple explanation
        simple_exp = explain_simply(
            verdict=verdict.verdict,
            confidence=verdict.confidence,
            reasoning=verdict.reasoning,
            red_flags=verdict.red_flags,
            recommendation=verdict.recommendation,
            claim=first_claim or state.get("content", ""),
            language=language
        )
        
        print(f"  ✅ Simple explanation generated")
        print(f"  📝 Verdict: {simple_exp['simple_verdict']}")
        
        return {"easy_explain": simple_exp}
    
    except Exception as e:
        print(f"  ⚠️ Failed to generate simple explanation: {e}")
        # Fallback simple explanation
        return {
            "easy_explain": {
                "greeting": "Dear Uncle/Aunty" if language == "en" else "प्रिय अंकल/आंटी",
                "simple_verdict": "We checked this message for you.",
                "explanation": "Please be careful with messages you receive. Not everything on the internet is true.",
                "what_to_do": "Please check with your family before sharing any message.",
                "why_matters": "This helps keep you and your loved ones safe from false information.",
                "language": language
            }
        }

# === 4. Build the graph
graph = StateGraph(GraphState)

# Add nodes
graph.add_node("DetectLanguage", detect_language)
graph.add_node("Translation", translation_agent)
graph.add_node("SkipTranslation", skip_translation)
graph.add_node("ClaimExtraction", extract_claim)
graph.add_node("Fusion", fusion_agent)
graph.add_node("RetrieveEvidence", retrieval_tool)
graph.add_node("GenerateVerdict", generate_verdict)  # Add verdict generation
graph.add_node("GenerateSimpleExplanation", generate_simple_explanation)  # Add simple explanation generation

# Define conditional routing function
def route_translation(state: GraphState) -> str:
    """Route to translation or skip based on language detection"""
    if state.get("needs_translation", False):
        return "Translation"
    else:
        return "SkipTranslation"

# Define edges with conditional routing
graph.add_conditional_edges(
    "DetectLanguage",
    route_translation,
    {
        "Translation": "Translation",
        "SkipTranslation": "SkipTranslation"
    }
)
graph.add_edge("Translation", "ClaimExtraction")
graph.add_edge("SkipTranslation", "ClaimExtraction")
graph.add_edge("ClaimExtraction", "Fusion")
graph.add_edge("Fusion", "RetrieveEvidence")
graph.add_edge("RetrieveEvidence", "GenerateVerdict")  # Changed: Route to verdict
graph.add_edge("GenerateVerdict", "GenerateSimpleExplanation")  # Route to simple explanation
graph.add_edge("GenerateSimpleExplanation", END)  # End after simple explanation

# Set entry point
graph.set_entry_point("DetectLanguage")

# Compile the graph
text_pipeline = graph.compile()


# === 5. Wrapper function for easy use
def run_text_pipeline(text_input: str, enable_reranking: bool = True) -> dict:
    """
    Run the text pipeline with configurable options
    
    Args:
        text_input: Input text to analyze
        enable_reranking: Whether to enable LLM reranking (set to False to avoid API quota issues)
    
    Returns:
        Dictionary with pipeline results
    """
    # Store original reranking config
    import os
    original_reranking = os.environ.get("ENABLE_RERANKING", "true")
    
    try:
        # Set reranking config
        os.environ["ENABLE_RERANKING"] = "true" if enable_reranking else "false"
        
        # Run pipeline
        result = text_pipeline.invoke({"content": text_input})
        
        # Format the result for easier consumption
        return {
            "content": result.get("content", ""),
            "source_language": result.get("source_language", "unknown"),
            "translated_text": result.get("translated_text"),
            "extracted_claims": result.get("claims", {}).dict() if result.get("claims") else {},
            "fused_claims": result.get("fused_claims", {}).dict() if result.get("fused_claims") else {},
            "evidence_results": result.get("evidence_results", []),
            "evidence_links": result.get("evidence_links", []),
            "verdict": result.get("verdict", {}),  # May not exist yet
            "easy_explain": result.get("easy_explain", {})  # Simple explanation
        }
    finally:
        # Restore original config
        os.environ["ENABLE_RERANKING"] = original_reranking


# === 6. Run the pipeline
if __name__ == "__main__":
    # Test with English content
    print("=" * 60)
    print("TEST 1: English Content")
    print("=" * 60)
    input_text_en = {
        "content": "A viral post claims that drinking hot water every 15 minutes kills coronavirus."
    }
    
    result = text_pipeline.invoke(input_text_en)
    
    print("\n=== Final Result ===")
    print(f"Original Content: {result['content']}")
    print(f"Source Language: {result.get('source_language', 'N/A')}")
    print(f"Translated Text: {result.get('translated_text', 'N/A')}")
    print(f"\nExtracted Claims: {result['claims']}")
    print(f"\nFused Claims: {result['fused_claims']}")
    print(f"\nEvidence Links: {result['evidence_links']}")
    
    # Test with Hindi content
    print("\n" + "=" * 60)
    print("TEST 2: Hindi Content")
    print("=" * 60)
    input_text_hi = {
        "content": "एक वायरल पोस्ट में दावा किया गया है कि हर 15 मिनट में गर्म पानी पीने से कोरोनावायरस मर जाता है।"
    }
    
    result_hi = text_pipeline.invoke(input_text_hi)
    
    print("\n=== Final Result ===")
    print(f"Original Content: {result_hi['content']}")
    print(f"Source Language: {result_hi.get('source_language', 'N/A')}")
    print(f"Translated Text: {result_hi.get('translated_text', 'N/A')}")
    print(f"\nExtracted Claims: {result_hi['claims']}")
    print(f"\nFused Claims: {result_hi['fused_claims']}")
    print(f"\nEvidence Links: {result_hi['evidence_links']}")