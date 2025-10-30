"""
Multimodal Pipeline - Integrates Text and Image Analysis
Combines text_pipeline.py and image_pipeline.py for comprehensive fact-checking
"""
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, List, Optional
from dotenv import load_dotenv

# Import from text pipeline
from backend.models import ClaimList, FusedClaimList, VerdictResult
from backend.prompts import CLAIM_EXTRACTION_PROMPT, FUSION_PROMPT

# Import from image pipeline
from backend.models import ImageEvidence, ImageFusionResult
from backend.prompts import IMAGE_CAPTIONING_PROMPT, DEEPFAKE_DETECTION_PROMPT

# Import agents
from backend.text_pipeline import (
    detect_language,
    translation_agent,
    skip_translation,
    extract_claim,
    fusion_agent as text_fusion_agent,
    retrieval_tool
)
from backend.image_pipeline import (
    captioning_agent,
    deepfake_detection_agent,
    consolidate_evidence
)

load_dotenv()


# === 1. Define Unified Multimodal State
class MultimodalState(TypedDict):
    # Input
    content: str  # Text content
    image_path: Optional[str]  # Optional image
    
    # Text pipeline state
    translated_text: Optional[str]
    source_language: Optional[str]
    needs_translation: Optional[bool]
    claims: Optional[ClaimList]
    
    # Image pipeline state
    image_id: Optional[str]
    caption: Optional[str]
    detected_entities: Optional[List[str]]
    is_ai_generated: Optional[bool]
    ai_confidence: Optional[float]
    forensic_findings: Optional[str]
    ocr_text: Optional[str]
    image_evidence: Optional[ImageEvidence]
    
    # Fusion state
    fused_claims: Optional[FusedClaimList]
    evidence_links: Optional[List[str]]
    multimodal_verdict: Optional[VerdictResult]


# === 2. Define Models
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


# === 3. Define Multimodal Fusion Agent
def multimodal_fusion_agent(state: MultimodalState) -> dict:
    """
    Advanced fusion agent that combines:
    - Text claims
    - Image evidence
    - Cross-modal consistency checks
    """
    print("\n" + "=" * 70)
    print("🔗 MULTIMODAL FUSION AGENT")
    print("=" * 70)
    
    claims = state.get("claims")
    image_evidence = state.get("image_evidence")
    
    if not claims and not image_evidence:
        print("⚠️  No claims or image evidence to fuse")
        return {}
    
    # Extract claim statements
    claim_statements = []
    if claims:
        claim_statements = [claim.statement for claim in claims.claims]
    
    # Build comprehensive fusion prompt
    fusion_context = f"""
You are a multimodal fact-checking fusion agent.

TASK: Analyze the consistency and credibility across text and image modalities.

TEXT CLAIMS:
{chr(10).join(f"  {i+1}. {claim}" for i, claim in enumerate(claim_statements)) if claim_statements else "  No text claims provided"}

IMAGE ANALYSIS:
"""
    
    if image_evidence:
        fusion_context += f"""
  Caption: {image_evidence.caption}
  Entities: {', '.join(image_evidence.detected_entities or ['None'])}
  AI-Generated: {image_evidence.is_ai_generated} (confidence: {image_evidence.ai_confidence:.2%})
  Forensic Findings: {image_evidence.forensic_findings}
  OCR Text: {image_evidence.ocr_text or 'None'}
"""
    else:
        fusion_context += "  No image evidence provided\n"
    
    fusion_context += """

ANALYSIS FRAMEWORK:

1. **Cross-Modal Consistency**
   - Do the text claims align with the visual evidence?
   - Are there contradictions between what's claimed and what's shown?

2. **Credibility Assessment**
   - Is AI-generated content being passed off as real evidence?
   - Are there signs of manipulation or out-of-context usage?

3. **Verdict Categories**
   - LIKELY_TRUE: Strong evidence supports the claims
   - LIKELY_FALSE: Evidence contradicts the claims or shows manipulation
   - MISLEADING: Real content used in false context
   - INSUFFICIENT: Not enough evidence to determine

Output your analysis using the provided schema with these exact fields:
- verdict: one of ["LIKELY_TRUE", "LIKELY_FALSE", "MISLEADING", "INSUFFICIENT"]
- confidence: float between 0 and 1
- reasoning: detailed explanation
- red_flags: list of credibility issues (can be empty list)
- recommendation: what to investigate further
"""
    
    # Use structured output with Pydantic model
    response = llm.with_structured_output(VerdictResult).invoke(fusion_context)
    
    print(f"\n🎯 VERDICT: {response.verdict}")
    print(f"📊 Confidence: {response.confidence:.2%}")
    print(f"💭 Reasoning: {response.reasoning}")
    
    if response.red_flags:
        print(f"🚩 Red Flags:")
        for flag in response.red_flags:
            print(f"   - {flag}")
    
    return {"multimodal_verdict": response}


# === 4. Routing Functions
def has_image(state: MultimodalState) -> str:
    """Check if image is provided"""
    if state.get("image_path"):
        return "process_image"
    else:
        return "skip_image"


def skip_image_processing(state: MultimodalState) -> dict:
    """Skip image processing if no image provided"""
    print("⏩ No image provided, skipping image analysis")
    return {}


# === 5. Build Multimodal Graph
graph = StateGraph(MultimodalState)

# Text processing nodes
graph.add_node("DetectLanguage", detect_language)
graph.add_node("Translation", translation_agent)
graph.add_node("SkipTranslation", skip_translation)
graph.add_node("ClaimExtraction", extract_claim)

# Image processing nodes
graph.add_node("ImageCaptioning", captioning_agent)
graph.add_node("DeepfakeDetection", deepfake_detection_agent)
graph.add_node("ConsolidateEvidence", consolidate_evidence)
graph.add_node("SkipImage", skip_image_processing)

# Fusion node
graph.add_node("MultimodalFusion", multimodal_fusion_agent)

# Define conditional routing for translation
def route_translation(state: MultimodalState) -> str:
    if state.get("needs_translation", False):
        return "Translation"
    else:
        return "SkipTranslation"

# Define edges
# Text path
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

# After claim extraction, check for image
graph.add_conditional_edges(
    "ClaimExtraction",
    has_image,
    {
        "process_image": "ImageCaptioning",
        "skip_image": "SkipImage"
    }
)

# Image path
graph.add_edge("ImageCaptioning", "DeepfakeDetection")
graph.add_edge("DeepfakeDetection", "ConsolidateEvidence")
graph.add_edge("ConsolidateEvidence", "MultimodalFusion")

# Skip image path
graph.add_edge("SkipImage", "MultimodalFusion")

# Final
graph.add_edge("MultimodalFusion", END)

# Set entry point
graph.set_entry_point("DetectLanguage")

# Compile
multimodal_pipeline = graph.compile()


# === 6. Test the pipeline
if __name__ == "__main__":
    print("=" * 70)
    print("MULTIMODAL PIPELINE TEST")
    print("=" * 70)
    
    # Test 1: Text + Image
    print("\n🧪 TEST 1: Text + Image Analysis")
    print("-" * 70)
    
    result = multimodal_pipeline.invoke({
        "content": "According to NASA, chanting “Om” for 5 minutes every day increases oxygen levels in the body by 50%. That’s why Western scientists are now doing meditation in labs. Proud to be Indian! 🇮🇳",
        "image_path": "/home/arka/Desktop/Hackathons/ihub/pic3.png"
    })
    
    print("\n=== FINAL RESULTS ===")
    print(f"\nOriginal Content: {result['content']}")
    print(f"Language: {result.get('source_language', 'N/A')}")
    print(f"\nExtracted Claims: {len(result['claims'].claims) if result.get('claims') else 0}")
    if result.get('claims'):
        for i, claim in enumerate(result['claims'].claims, 1):
            print(f"  {i}. {claim.statement}")
    
    if result.get('image_evidence'):
        img_ev = result['image_evidence']
        print(f"\nImage Analysis:")
        print(f"  Caption: {img_ev.caption}")
        print(f"  AI-Generated: {img_ev.is_ai_generated} ({img_ev.ai_confidence:.2%})")
    
    if result.get('multimodal_verdict'):
        verdict = result['multimodal_verdict']
        print(f"\n🎯 MULTIMODAL VERDICT: {verdict.verdict}")
        print(f"   Confidence: {verdict.confidence:.2%}")
        print(f"   Reasoning: {verdict.reasoning}")
        if verdict.red_flags:
            print(f"   Red Flags: {', '.join(verdict.red_flags)}")
    
    print("\n" + "=" * 70)
