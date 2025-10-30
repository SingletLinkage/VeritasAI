"""
Image Analysis Pipeline for VeritasAI
Handles image captioning, deepfake detection, OCR, EXIF extraction, and fusion with text claims
"""
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.models import (
    ImageEvidence, 
    CaptionResult, 
    DeepfakeResult, 
    ImageFusionResult,
    OCRResult,
    ExifResult
)
from backend.prompts import (
    IMAGE_CAPTIONING_PROMPT,
    DEEPFAKE_DETECTION_PROMPT,
    IMAGE_FUSION_PROMPT,
    OCR_EXTRACTION_PROMPT
)
from backend.exif_tool import extract_exif
from dotenv import load_dotenv
from typing import TypedDict, Optional, List
import base64
from pathlib import Path

load_dotenv()


# === 1. Define the State Schema
class ImageGraphState(TypedDict):
    image_path: str
    image_id: Optional[str]
    claim_statement: Optional[str]  # For fusion with text claims
    caption: Optional[str]
    detected_entities: Optional[List[str]]
    is_ai_generated: Optional[bool]
    ai_confidence: Optional[float]
    forensic_findings: Optional[str]
    ocr_text: Optional[str]
    exif_data: Optional[dict]  # EXIF metadata
    image_evidence: Optional[ImageEvidence]
    fusion_result: Optional[ImageFusionResult]


# === 2. Define the models
vision_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

deepfake_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1
)

fusion_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)


# === 3. Helper function to encode image
def encode_image(image_path: str) -> str:
    """Encode image to base64 for API calls"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# === 4. Define Agents
def captioning_agent(state: ImageGraphState) -> dict:
    """Extract meaningful caption and entities from image"""
    image_path = state["image_path"]
    
    print(f"Captioning image: {Path(image_path).name}")
    
    # For Gemini, we need to use the proper image input format
    from langchain_core.messages import HumanMessage
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": IMAGE_CAPTIONING_PROMPT},
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{encode_image(image_path)}"
            }
        ]
    )
    
    response = vision_model.with_structured_output(CaptionResult).invoke([message])
    
    print(f"Caption generated: {response.caption}...")
    
    return {
        "caption": response.caption,
        "detected_entities": response.detected_entities,
        "ocr_text": response.visible_text_summary
    }


def deepfake_detection_agent(state: ImageGraphState) -> dict:
    """Detect if image is AI-generated, manipulated, or deepfaked"""
    image_path = state["image_path"]
    
    print(f"Analyzing image authenticity...")
    
    from langchain_core.messages import HumanMessage
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": DEEPFAKE_DETECTION_PROMPT},
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{encode_image(image_path)}"
            }
        ]
    )
    
    response = deepfake_model.with_structured_output(DeepfakeResult).invoke([message])
    
    status = "AI-GENERATED" if response.is_ai_generated else "✓ Authentic"
    print(f"{status} (confidence: {response.ai_confidence:.2f})")
    
    return {
        "is_ai_generated": response.is_ai_generated,
        "ai_confidence": response.ai_confidence,
        "forensic_findings": response.forensic_findings + "Manipulation indicators: " + ", ".join(response.manipulation_indicators or [])
    }


def ocr_agent(state: ImageGraphState) -> dict:
    """Extract text from image using OCR (optional, can be enhanced with dedicated OCR)"""
    # For now, we rely on the captioning agent's visible_text_summary
    # This can be enhanced with pytesseract or Google Vision API later
    
    print(f"📝 OCR text already extracted during captioning")
    
    # OCR text is already set by captioning_agent
    return {}


def exif_extraction_agent(state: ImageGraphState) -> dict:
    """Extract EXIF metadata from image for authenticity verification"""
    image_path = state["image_path"]
    
    print(f"📷 Extracting EXIF metadata...")
    
    exif_result = extract_exif(image_path)
    
    if exif_result.status == "success" and exif_result.metadata:
        # Convert Pydantic model to dict for storage
        exif_dict = exif_result.metadata.model_dump(exclude_none=True)
        
        print(f"✅ EXIF extracted: {exif_result.metadata.Make or 'Unknown'} {exif_result.metadata.Model or ''}")
        if exif_result.metadata.DateTime:
            print(f"   Captured: {exif_result.metadata.DateTime}")
        if exif_result.metadata.Software:
            print(f"   ⚠️  Edited with: {exif_result.metadata.Software}")
        
        return {"exif_data": exif_dict}
    
    elif exif_result.status == "no_exif":
        print(f"ℹ️  No EXIF data found (may indicate screenshot or edited image)")
        return {"exif_data": None}
    
    else:
        print(f"⚠️  EXIF extraction error: {exif_result.error}")
        return {"exif_data": None}


def consolidate_evidence(state: ImageGraphState) -> dict:
    """Consolidate all image analysis into ImageEvidence model"""
    
    print(f"📦 Consolidating image evidence...")
    
    image_id = state.get("image_id") or f"img_{hash(state['image_path'])}"
    
    evidence = ImageEvidence(
        image_id=image_id,
        image_path=state["image_path"],
        caption=state["caption"],
        detected_entities=state.get("detected_entities", []),
        is_ai_generated=state["is_ai_generated"],
        ai_confidence=state["ai_confidence"],
        forensic_findings=state["forensic_findings"],
        ocr_text=state.get("ocr_text"),
        exif_data=state.get("exif_data"),  # Now includes EXIF data
        reverse_search_results=None  # Can be added later with reverse search
    )
    
    print(f"✅ Evidence consolidated for {image_id}")
    
    return {"image_evidence": evidence}


def image_text_fusion_agent(state: ImageGraphState) -> dict:
    """Fuse image evidence with text claim for cross-modal verification"""
    
    claim_statement = state.get("claim_statement")
    
    if not claim_statement:
        print("⏩ No text claim provided, skipping fusion")
        return {}
    
    print(f"🔗 Fusing image evidence with text claim...")
    
    # Format the fusion prompt with all available data
    fusion_prompt = IMAGE_FUSION_PROMPT.format(
        claim_statement=claim_statement,
        image_caption=state["caption"],
        detected_entities=", ".join(state.get("detected_entities", [])) or "None",
        is_ai_generated="Yes" if state["is_ai_generated"] else "No",
        ai_confidence=f"{state['ai_confidence']:.2f}",
        forensic_findings=state["forensic_findings"],
        ocr_text=state.get("ocr_text") or "None"
    )
    
    response = fusion_model.with_structured_output(ImageFusionResult).invoke(fusion_prompt)
    
    print(f"✅ Fusion result: {response.relation.upper()} (confidence: {response.fusion_confidence:.2f})")
    
    return {"fusion_result": response}


# === 5. Build the graph
graph = StateGraph(ImageGraphState)

# Add nodes
graph.add_node("Captioning", captioning_agent)
graph.add_node("DeepfakeDetection", deepfake_detection_agent)
graph.add_node("ExifExtraction", exif_extraction_agent)
graph.add_node("ConsolidateEvidence", consolidate_evidence)
graph.add_node("ImageTextFusion", image_text_fusion_agent)

# Define edges
graph.add_edge("Captioning", "DeepfakeDetection")
graph.add_edge("DeepfakeDetection", "ExifExtraction")
graph.add_edge("ExifExtraction", "ConsolidateEvidence")
graph.add_edge("ConsolidateEvidence", "ImageTextFusion")
graph.add_edge("ImageTextFusion", END)

# Set entry point
graph.set_entry_point("Captioning")

# Compile the graph
image_pipeline = graph.compile()


# === 6. Standalone execution for testing
if __name__ == "__main__":
    import sys
    
    # Test with sample image
    test_image_path = "/home/arka/Desktop/Hackathons/ihub/pic.jpg"
    
    print("=" * 70)
    print("IMAGE ANALYSIS PIPELINE TEST")
    print("=" * 70)
    
    # Test 1: Image analysis only (no claim)
    print("\n🧪 TEST 1: Image Analysis Only")
    print("-" * 70)
    
    result = image_pipeline.invoke({
        "image_path": test_image_path,
        "image_id": "test_img_001"
    })
    
    print("\n=== Results ===")
    evidence = result["image_evidence"]
    print(f"\n📸 Image: {evidence.image_id}")
    print(f"📝 Caption: {evidence.caption}")
    print(f"🏷️  Entities: {', '.join(evidence.detected_entities or ['None'])}")
    print(f"🤖 AI-Generated: {evidence.is_ai_generated} (confidence: {evidence.ai_confidence:.2%})")
    print(f"🔬 Forensic Findings: {evidence.forensic_findings}")
    if evidence.ocr_text:
        print(f"📄 OCR Text: {evidence.ocr_text}")
    
    # Display EXIF metadata
    if evidence.exif_data:
        print(f"\n📷 EXIF Metadata:")
        if evidence.exif_data.get("Make"):
            print(f"   Camera: {evidence.exif_data.get('Make')} {evidence.exif_data.get('Model', '')}")
        if evidence.exif_data.get("DateTime"):
            print(f"   Date: {evidence.exif_data.get('DateTime')}")
        if evidence.exif_data.get("Software"):
            print(f"   ⚠️  Software: {evidence.exif_data.get('Software')}")
        if evidence.exif_data.get("GPSInfo"):
            print(f"   📍 GPS: Location data available")
    else:
        print(f"\n📷 EXIF: Not available (may indicate screenshot/edited image)")
    
    # Test 2: Image + Text claim fusion
    print("\n" + "=" * 70)
    print("🧪 TEST 2: Image-Text Fusion")
    print("-" * 70)
    
    result_fusion = image_pipeline.invoke({
        "image_path": test_image_path,
        "image_id": "test_img_002",
        "claim_statement": "This photo shows recent flooding in Mumbai during monsoon season"
    })
    
    print("\n=== Fusion Results ===")
    fusion = result_fusion.get("fusion_result")
    if fusion:
        print(f"🔗 Relation: {fusion.relation.upper()}")
        print(f"📊 Confidence: {fusion.fusion_confidence:.2%}")
        print(f"💭 Reasoning: {fusion.reasoning}")
        if fusion.credibility_flags:
            print(f"⚠️  Flags: {', '.join(fusion.credibility_flags)}")
    
    print("\n" + "=" * 70)
