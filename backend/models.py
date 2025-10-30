from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class Claim(BaseModel):
    """A factual claim extracted from multimodal input content."""
    
    id: str = Field(..., description="Unique claim ID, e.g., 'C001'.")
    statement: str = Field(..., description="The extracted factual statement.")
    language: str = Field(..., description="Detected language (ISO code, e.g., 'en', 'hi', 'bn', 'de').")
    source_modality: Literal["text", "audio", "image", "video", "ocr", "caption"] = Field(
        ..., description="Source content modality of this claim."
    )
    type: Optional[str] = Field(
        None, description="Optional tag for claim domain, e.g., 'health', 'political', etc."
    )


class ClaimList(BaseModel):
    """Container for multiple claims extracted from a single content item."""
    
    claims: List[Claim]


class FusedClaim(BaseModel):
    """Multimodal fusion result combining claims from multiple modalities."""
    id: str
    fused_statement: str
    evidence_links: List[str] = Field(default_factory=list, description="Sources of corroboration from RAG/Web.")
    supporting_modalities: List[str]
    confidence_score: float = Field(..., description="Fusion confidence [0-1].")
    consensus_summary: Optional[str] = Field(None, description="Short summary of consensus across modalities.")

class FusedClaimList(BaseModel):
    """Container for multiple fused claims."""
    
    fused_claims: List[FusedClaim]

class TranslationResult(BaseModel):
    """Result of translating content into English."""
    
    translated_text: str = Field(..., description="The translated English text.")
    source_language: str = Field(..., description="Detected source language (ISO code).")

    """
Pydantic models for Image Analysis Pipeline
"""

class ImageEvidence(BaseModel):
    """Structured evidence extracted from image analysis"""
    
    image_id: str = Field(..., description="Unique identifier for the image being processed")
    image_path: str = Field(..., description="Path to the image file")
    caption: str = Field(..., description="Medium-length descriptive caption of what the image shows")
    detected_entities: Optional[List[str]] = Field(
        default=None, 
        description="Entities recognized in caption or content (people, places, objects)"
    )
    is_ai_generated: bool = Field(..., description="True if detected as AI-generated or deepfake")
    ai_confidence: float = Field(..., ge=0, le=1, description="Confidence of AI-generated classification")
    forensic_findings: Optional[str] = Field(
        default=None, 
        description="Optional additional notes about anomalies, blending, manipulation, etc."
    )
    ocr_text: Optional[str] = Field(
        default=None,
        description="Text extracted from image via OCR (if any)"
    )
    exif_data: Optional[dict] = Field(
        default=None,
        description="EXIF metadata extracted from image"
    )
    reverse_search_results: Optional[List[dict]] = Field(
        default=None,
        description="Reverse image search results for verification"
    )


class CaptionResult(BaseModel):
    """Result from image captioning agent"""
    
    caption: str = Field(..., description="Factual description of image content")
    detected_entities: List[str] = Field(
        default_factory=list,
        description="List of entities detected in the image"
    )
    visible_text_summary: Optional[str] = Field(
        default=None,
        description="Any visible text in the image"
    )


class DeepfakeResult(BaseModel):
    """Result from AI generation/deepfake detection agent"""
    
    is_ai_generated: bool = Field(..., description="Whether image appears to be AI-generated")
    ai_confidence: float = Field(..., ge=0, le=1, description="Confidence score for AI detection")
    forensic_findings: str = Field(..., description="Detailed explanation of detection reasoning")
    manipulation_indicators: Optional[List[str]] = Field(
        default=None,
        description="Specific indicators of manipulation found"
    )


class ImageFusionResult(BaseModel):
    """Result of fusing text claims with image evidence"""
    
    relation: Literal["supports", "contradicts", "unrelated", "inconclusive"] = Field(
        ..., description="How the image relates to the text claim"
    )
    fusion_confidence: float = Field(..., ge=0, le=1, description="Confidence in the fusion verdict")
    reasoning: str = Field(..., description="Detailed explanation of the relationship")
    credibility_flags: List[str] = Field(
        default_factory=list,
        description="Any red flags or credibility issues identified"
    )


class OCRResult(BaseModel):
    """Result from OCR extraction"""
    
    extracted_text: str = Field(..., description="Text extracted from the image")
    text_regions: Optional[List[dict]] = Field(
        default=None,
        description="Bounding boxes and coordinates of text regions"
    )
    confidence: float = Field(..., ge=0, le=1, description="OCR confidence score")


class ExifMetadata(BaseModel):
    """EXIF metadata extracted from image"""
    
    Make: Optional[str] = Field(default=None, description="Camera/device manufacturer")
    Model: Optional[str] = Field(default=None, description="Camera/device model")
    DateTime: Optional[str] = Field(default=None, description="Date and time image was taken")
    DateTimeOriginal: Optional[str] = Field(default=None, description="Original date and time")
    DateTimeDigitized: Optional[str] = Field(default=None, description="Date and time digitized")
    Software: Optional[str] = Field(default=None, description="Software used to create/edit image")
    Orientation: Optional[str] = Field(default=None, description="Image orientation")
    XResolution: Optional[str] = Field(default=None, description="Horizontal resolution")
    YResolution: Optional[str] = Field(default=None, description="Vertical resolution")
    Flash: Optional[str] = Field(default=None, description="Flash setting used")
    FocalLength: Optional[str] = Field(default=None, description="Focal length of lens")
    ExposureTime: Optional[str] = Field(default=None, description="Exposure time")
    FNumber: Optional[str] = Field(default=None, description="F-number/aperture")
    ISOSpeedRatings: Optional[str] = Field(default=None, description="ISO speed")
    GPSInfo: Optional[dict] = Field(default=None, description="GPS location data if available")
    raw_metadata: Optional[dict] = Field(default=None, description="Complete raw EXIF data")


class ExifResult(BaseModel):
    """Result from EXIF extraction process"""
    
    status: Literal["success", "no_exif", "error"] = Field(
        ..., description="Status of EXIF extraction"
    )
    metadata: Optional[ExifMetadata] = Field(
        default=None, 
        description="Parsed EXIF metadata"
    )
    error: Optional[str] = Field(
        default=None, 
        description="Error message if extraction failed"
    )


class VerdictResult(BaseModel):
    """Multimodal fact-checking verdict"""
    
    verdict: Literal["LIKELY_TRUE", "LIKELY_FALSE", "MISLEADING", "INSUFFICIENT"] = Field(
        ..., description="Final verdict on the claim"
    )
    confidence: float = Field(
        ..., ge=0, le=1, 
        description="Confidence score for the verdict"
    )
    reasoning: str = Field(
        ..., description="Detailed explanation of the verdict"
    )
    red_flags: List[str] = Field(
        default_factory=list,
        description="List of credibility issues identified"
    )
    recommendation: str = Field(
        ..., description="What fact-checkers should investigate further"
    )
