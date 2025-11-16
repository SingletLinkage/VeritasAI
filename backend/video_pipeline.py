"""
Video Analysis Pipeline for VeritasAI
Handles video fact-checking through synchronized multimodal processing:
- Keyframe extraction and media separation
- Audio transcription and claim extraction
- Visual analysis of keyframes
- Timestamp-synchronized fusion
- Fact-checking with vector store retrieval
"""

from typing import TypedDict, Optional, List, Dict, Any, Tuple
import os
import sys
from pathlib import Path
import asyncio
import tempfile
import hashlib
from datetime import timedelta
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Import existing pipelines
from backend.text_pipeline import text_pipeline
from backend.image_pipeline import captioning_agent, deepfake_detection_agent
from backend.models import ClaimList, VerdictResult
from backend.hybrid_retrieval import retrieve_evidence
from backend.explainability import explain_simply

# Video processing libraries
from moviepy.editor import VideoFileClip
import cv2
import numpy as np
import whisper

load_dotenv()


# === 1. Data Models ===

class Timestamp(BaseModel):
    """Timestamp representation"""
    seconds: float = Field(description="Timestamp in seconds")
    formatted: str = Field(description="Human-readable timestamp (HH:MM:SS)")
    
    @classmethod
    def from_seconds(cls, seconds: float) -> "Timestamp":
        """Create timestamp from seconds"""
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        secs = int(td.total_seconds() % 60)
        formatted = f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return cls(seconds=seconds, formatted=formatted)


class Keyframe(BaseModel):
    """Extracted keyframe with metadata"""
    frame_id: str = Field(description="Unique frame identifier")
    timestamp: Timestamp = Field(description="Timestamp in video")
    frame_path: str = Field(description="Path to saved frame")
    frame_number: int = Field(description="Frame number in video")
    caption: Optional[str] = None
    detected_entities: Optional[List[str]] = None
    is_ai_generated: Optional[bool] = None
    ai_confidence: Optional[float] = None


class AudioSegment(BaseModel):
    """Transcribed audio segment with metadata"""
    segment_id: str = Field(description="Unique segment identifier")
    start_time: Timestamp = Field(description="Start timestamp")
    end_time: Timestamp = Field(description="End timestamp")
    text: str = Field(description="Transcribed text")
    language: Optional[str] = None
    confidence: Optional[float] = None


class VideoClaim(BaseModel):
    """Claim extracted from video with multimodal context"""
    claim_id: str = Field(description="Unique claim identifier")
    claim_text: str = Field(description="The fact-checkable claim")
    source: str = Field(description="Source of claim (audio/visual/both)")
    timestamp: Timestamp = Field(description="Primary timestamp")
    audio_context: Optional[str] = None
    visual_context: Optional[List[str]] = None  # Frame IDs
    keyframes: Optional[List[Keyframe]] = None
    confidence: float = Field(default=0.5, description="Extraction confidence")


class VideoMetadata(BaseModel):
    """Video file metadata"""
    filename: str
    duration: float
    fps: float
    resolution: Tuple[int, int]
    total_frames: int
    audio_channels: Optional[int] = None
    file_size: int
    video_hash: str


class FusedVideoEvidence(BaseModel):
    """Fused evidence from audio and visual sources"""
    claim: VideoClaim
    supporting_frames: List[Keyframe]
    contradicting_frames: List[Keyframe]
    audio_evidence: List[AudioSegment]
    temporal_consistency: float = Field(description="How consistent is evidence across time")
    multimodal_alignment: float = Field(description="How well audio and visual align")


# === 2. State Definition ===

class VideoGraphState(TypedDict):
    # Input
    video_path: str
    
    # Media separation
    audio_path: Optional[str]
    keyframes: Optional[List[Keyframe]]
    video_metadata: Optional[VideoMetadata]
    
    # Audio processing
    audio_segments: Optional[List[AudioSegment]]
    full_transcription: Optional[str]
    detected_language: Optional[str]
    
    # Visual processing
    analyzed_keyframes: Optional[List[Keyframe]]
    
    # Claim extraction
    video_claims: Optional[List[VideoClaim]]
    
    # Fusion
    fused_evidence: Optional[List[FusedVideoEvidence]]
    
    # Fact-checking
    verdict_results: Optional[List[Dict[str, Any]]]
    
    # Final output
    final_report: Optional[Dict[str, Any]]
    error: Optional[str]


# === 3. Media Separation & Keyframe Extraction ===

def extract_video_metadata(video_path: str) -> VideoMetadata:
    """Extract metadata from video file"""
    clip = VideoFileClip(video_path)
    
    # Calculate video hash
    with open(video_path, 'rb') as f:
        video_hash = hashlib.md5(f.read()).hexdigest()
    
    metadata = VideoMetadata(
        filename=Path(video_path).name,
        duration=clip.duration,
        fps=clip.fps,
        resolution=(clip.w, clip.h),
        total_frames=int(clip.duration * clip.fps),
        audio_channels=clip.audio.nchannels if clip.audio else None,
        file_size=os.path.getsize(video_path),
        video_hash=video_hash
    )
    
    clip.close()
    return metadata


def extract_keyframes(
    video_path: str,
    method: str = "uniform",
    interval: float = 2.0,
    threshold: float = 0.3,
    max_frames: int = 30
) -> List[Keyframe]:
    """
    Extract keyframes from video
    
    Args:
        video_path: Path to video file
        method: 'uniform' (time-based) or 'scene' (scene change detection)
        interval: Seconds between frames (for uniform method)
        threshold: Scene change threshold (for scene method)
        max_frames: Maximum number of frames to extract
        
    Returns:
        List of extracted keyframes
    """
    print(f"\n🎬 Extracting keyframes from video...")
    print(f"  Method: {method}, Interval: {interval}s, Max: {max_frames}")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    keyframes = []
    temp_dir = tempfile.mkdtemp(prefix="veritas_keyframes_")
    
    if method == "uniform":
        # Extract frames at uniform intervals
        timestamps = np.arange(0, duration, interval)
        timestamps = timestamps[:max_frames]  # Limit to max_frames
        
        for i, timestamp in enumerate(timestamps):
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                frame_id = f"frame_{i:04d}"
                frame_path = os.path.join(temp_dir, f"{frame_id}.jpg")
                cv2.imwrite(frame_path, frame)
                
                keyframes.append(Keyframe(
                    frame_id=frame_id,
                    timestamp=Timestamp.from_seconds(timestamp),
                    frame_path=frame_path,
                    frame_number=frame_number
                ))
    
    elif method == "scene":
        # Scene change detection using histogram difference
        prev_hist = None
        frame_idx = 0
        
        while len(keyframes) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate histogram
            hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            # Compare with previous frame
            if prev_hist is not None:
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                
                # If difference exceeds threshold, save as keyframe
                if diff < (1 - threshold):
                    timestamp = frame_idx / fps
                    frame_id = f"frame_{len(keyframes):04d}"
                    frame_path = os.path.join(temp_dir, f"{frame_id}.jpg")
                    cv2.imwrite(frame_path, frame)
                    
                    keyframes.append(Keyframe(
                        frame_id=frame_id,
                        timestamp=Timestamp.from_seconds(timestamp),
                        frame_path=frame_path,
                        frame_number=frame_idx
                    ))
            
            prev_hist = hist
            frame_idx += 1
    
    cap.release()
    
    print(f"  ✅ Extracted {len(keyframes)} keyframes")
    return keyframes


def separate_media_node(state: VideoGraphState) -> dict:
    """
    Separate video into audio and keyframes
    Node 1: Media Separation
    """
    video_path = state["video_path"]
    
    print(f"\n🎥 Processing video: {Path(video_path).name}")
    
    if not os.path.exists(video_path):
        return {"error": f"Video file not found: {video_path}"}
    
    
    try:
        # Extract metadata
        print("  📊 Extracting metadata...")
        metadata = extract_video_metadata(video_path)
        print(f"  ✅ Duration: {metadata.duration:.1f}s, FPS: {metadata.fps}, Resolution: {metadata.resolution}")
        
        # Extract audio
        print("  🎵 Extracting audio...")
        temp_audio_dir = tempfile.mkdtemp(prefix="veritas_audio_")
        audio_path = os.path.join(temp_audio_dir, "audio.wav")
        
        clip = VideoFileClip(video_path)
        if clip.audio:
            clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
            print(f"  ✅ Audio extracted: {audio_path}")
        else:
            audio_path = None
            print("  ⚠️ No audio track found")
        clip.close()
        
        # Extract keyframes
        keyframe_method = os.getenv("KEYFRAME_METHOD", "uniform")
        keyframe_interval = float(os.getenv("KEYFRAME_INTERVAL", "2.0"))
        max_keyframes = int(os.getenv("MAX_KEYFRAMES", "30"))
        
        keyframes = extract_keyframes(
            video_path,
            method=keyframe_method,
            interval=keyframe_interval,
            max_frames=max_keyframes
        )
        
        return {
            "video_metadata": metadata,
            "audio_path": audio_path,
            "keyframes": keyframes,
            "error": None
        }
        
    except Exception as e:
        error_msg = f"Media separation failed: {str(e)}"
        print(f"  ❌ {error_msg}")
        return {"error": error_msg}


# === 4. Audio Processing ===

def transcribe_audio_node(state: VideoGraphState) -> dict:
    """
    Transcribe audio with timestamps
    Node 2: Audio Transcription
    """
    audio_path = state.get("audio_path")
    
    if not audio_path:
        print("\n⚠️ No audio to transcribe")
        return {"audio_segments": [], "full_transcription": ""}
    
    print(f"\n🎤 Transcribing audio with Whisper...")
    
    
    try:
        # Load Whisper model
        model_name = os.getenv("WHISPER_MODEL", "base")
        model = whisper.load_model(model_name)
        
        # Transcribe with word-level timestamps
        result = model.transcribe(
            audio_path,
            fp16=False,
            word_timestamps=True,
            verbose=False
        )
        
        # Extract segments with timestamps
        segments = []
        for i, segment in enumerate(result["segments"]):
            audio_seg = AudioSegment(
                segment_id=f"audio_seg_{i:04d}",
                start_time=Timestamp.from_seconds(segment["start"]),
                end_time=Timestamp.from_seconds(segment["end"]),
                text=segment["text"].strip(),
                language=result.get("language"),
                confidence=segment.get("confidence")
            )
            segments.append(audio_seg)
        
        full_transcription = result["text"].strip()
        
        print(f"  ✅ Transcribed {len(segments)} segments")
        print(f"  🌐 Language: {result.get('language')}")
        print(f"  📝 Total text: {len(full_transcription)} chars")
        
        return {
            "audio_segments": segments,
            "full_transcription": full_transcription,
            "detected_language": result.get("language"),
            "error": None
        }
        
    except Exception as e:
        error_msg = f"Transcription failed: {str(e)}"
        print(f"  ❌ {error_msg}")
        return {"error": error_msg}


# === 5. Visual Processing ===

def analyze_keyframes_node(state: VideoGraphState) -> dict:
    """
    Analyze keyframes for visual content
    Node 3: Visual Analysis
    """
    keyframes = state.get("keyframes", [])
    
    if not keyframes:
        print("\n⚠️ No keyframes to analyze")
        return {"analyzed_keyframes": []}
    
    print(f"\n🖼️ Analyzing {len(keyframes)} keyframes...")
    
    analyzed_frames = []
    
    for i, frame in enumerate(keyframes):
        print(f"  [{i+1}/{len(keyframes)}] Processing {frame.frame_id}...")
        
        try:
            # Create temporary state for image pipeline
            from backend.image_pipeline import ImageGraphState
            
            image_state = {
                "image_path": frame.frame_path,
                "image_id": frame.frame_id
            }
            
            # Run captioning
            caption_result = captioning_agent(image_state)
            
            # Run deepfake detection
            deepfake_result = deepfake_detection_agent({**image_state, **caption_result})
            
            # Update frame with analysis
            frame.caption = caption_result.get("caption")
            frame.detected_entities = caption_result.get("detected_entities", [])
            frame.is_ai_generated = deepfake_result.get("is_ai_generated")
            frame.ai_confidence = deepfake_result.get("ai_confidence")
            
            analyzed_frames.append(frame)
            
        except Exception as e:
            print(f"    ⚠️ Analysis failed: {e}")
            analyzed_frames.append(frame)
    
    print(f"  ✅ Analyzed {len(analyzed_frames)} frames")
    
    return {"analyzed_keyframes": analyzed_frames}


# === 6. Claim Extraction ===

def extract_video_claims_node(state: VideoGraphState) -> dict:
    """
    Extract fact-checkable claims from audio and visual content
    Node 4: Claim Extraction
    """
    audio_segments = state.get("audio_segments", [])
    analyzed_frames = state.get("analyzed_keyframes", [])
    full_transcription = state.get("full_transcription", "")
    
    print(f"\n🔍 Extracting claims from video content...")
    
    claims = []
    
    # Extract claims from full transcription using text pipeline
    if full_transcription:
        print(f"  📝 Analyzing transcription ({len(full_transcription)} chars)...")
        
        try:
            # Run text pipeline for claim extraction
            text_result = text_pipeline.invoke({"content": full_transcription})
            
            extracted_claims = text_result.get("claims")
            if extracted_claims and hasattr(extracted_claims, 'claims'):
                for i, claim in enumerate(extracted_claims.claims):
                    # Find corresponding audio segment
                    video_claim = VideoClaim(
                        claim_id=f"claim_{len(claims):04d}",
                        claim_text=claim.claim_statement,
                        source="audio",
                        timestamp=Timestamp.from_seconds(0),  # Default to start
                        audio_context=claim.claim_statement,
                        confidence=0.8
                    )
                    claims.append(video_claim)
            
            print(f"    ✅ Extracted {len(claims)} claims from audio")
            
        except Exception as e:
            print(f"    ⚠️ Audio claim extraction failed: {e}")
    
    # Extract claims from visual content
    visual_claim_count = 0
    for frame in analyzed_frames:
        if frame.caption and len(frame.caption) > 50:
            # If caption contains substantial information, treat as potential claim
            # You can enhance this with more sophisticated visual claim detection
            if frame.detected_entities and len(frame.detected_entities) > 2:
                video_claim = VideoClaim(
                    claim_id=f"claim_{len(claims):04d}",
                    claim_text=frame.caption,
                    source="visual",
                    timestamp=frame.timestamp,
                    visual_context=[frame.frame_id],
                    keyframes=[frame],
                    confidence=0.6
                )
                claims.append(video_claim)
                visual_claim_count += 1
    
    print(f"    ✅ Extracted {visual_claim_count} claims from visuals")
    print(f"  🎯 Total claims: {len(claims)}")
    
    return {"video_claims": claims}


# === 7. Temporal Fusion ===

def fuse_multimodal_evidence_node(state: VideoGraphState) -> dict:
    """
    Fuse audio and visual evidence with temporal alignment
    Node 5: Multimodal Fusion
    """
    claims = state.get("video_claims", [])
    audio_segments = state.get("audio_segments", [])
    analyzed_frames = state.get("analyzed_keyframes", [])
    
    print(f"\n🔀 Fusing multimodal evidence for {len(claims)} claims...")
    
    fused_evidence = []
    
    for claim in claims:
        print(f"  📌 Claim: {claim.claim_text[:60]}...")
        
        # Find temporally aligned evidence
        claim_time = claim.timestamp.seconds
        time_window = 5.0  # 5-second window
        
        # Find supporting/contradicting frames within time window
        supporting_frames = []
        contradicting_frames = []
        
        for frame in analyzed_frames:
            frame_time = frame.timestamp.seconds
            if abs(frame_time - claim_time) <= time_window:
                # Simple heuristic: check if entities match
                if frame.caption and claim.claim_text.lower() in frame.caption.lower():
                    supporting_frames.append(frame)
                else:
                    # More sophisticated matching can be added here
                    pass
        
        # Find related audio segments
        related_audio = []
        for seg in audio_segments:
            if (seg.start_time.seconds <= claim_time <= seg.end_time.seconds):
                related_audio.append(seg)
        
        # Calculate alignment scores
        temporal_consistency = min(1.0, len(supporting_frames) / max(1, len(supporting_frames) + len(contradicting_frames)))
        multimodal_alignment = 1.0 if (supporting_frames and related_audio) else 0.5
        
        fused = FusedVideoEvidence(
            claim=claim,
            supporting_frames=supporting_frames,
            contradicting_frames=contradicting_frames,
            audio_evidence=related_audio,
            temporal_consistency=temporal_consistency,
            multimodal_alignment=multimodal_alignment
        )
        
        fused_evidence.append(fused)
        
        print(f"    ✅ Fused: {len(supporting_frames)} frames, {len(related_audio)} audio segments")
    
    print(f"  🎯 Fused evidence for {len(fused_evidence)} claims")
    
    return {"fused_evidence": fused_evidence}


# === 8. Fact-Checking ===

def fact_check_claims_node(state: VideoGraphState) -> dict:
    """
    Fact-check claims using hybrid retrieval and LLM
    Node 6: Fact-Checking
    """
    fused_evidence = state.get("fused_evidence", [])
    
    print(f"\n✅ Fact-checking {len(fused_evidence)} claims...")
    
    verdict_results = []
    
    from langchain_google_genai import ChatGoogleGenerativeAI
    from backend.prompts import VIDEO_FACT_CHECK_PROMPT
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-thinking-exp-01-21",
        temperature=0.1
    )
    
    for i, evidence in enumerate(fused_evidence):
        claim_text = evidence.claim.claim_text
        print(f"  [{i+1}/{len(fused_evidence)}] Checking: {claim_text[:60]}...")
        
        try:
            # Retrieve external evidence
            retrieved_evidence = retrieve_evidence(claim_text, top_k=3)
            
            # Build context from multimodal evidence
            context_parts = []
            
            # Add audio context
            if evidence.audio_evidence:
                audio_texts = [seg.text for seg in evidence.audio_evidence]
                context_parts.append(f"Audio Context: {' '.join(audio_texts)}")
            
            # Add visual context
            if evidence.supporting_frames:
                visual_desc = [f.caption for f in evidence.supporting_frames if f.caption]
                context_parts.append(f"Visual Context: {' | '.join(visual_desc)}")
            
            # Add retrieved evidence
            if retrieved_evidence:
                evidence_texts = [e.content[:200] for e in retrieved_evidence[:3]]
                context_parts.append(f"External Evidence: {' | '.join(evidence_texts)}")
            
            full_context = "\n".join(context_parts)
            
            # Generate verdict
            prompt = VIDEO_FACT_CHECK_PROMPT.format(
                claim=claim_text,
                context=full_context,
                temporal_consistency=evidence.temporal_consistency,
                multimodal_alignment=evidence.multimodal_alignment
            )
            
            verdict = llm.with_structured_output(VerdictResult).invoke(prompt)
            
            # Generate simple explanation
            simple_exp = explain_simply(
                verdict=verdict.verdict,
                confidence=verdict.confidence,
                reasoning=verdict.reasoning,
                red_flags=verdict.red_flags,
                recommendation=verdict.recommendation,
                claim=claim_text,
                language=state.get("detected_language", "en")
            )
            
            result = {
                "claim": claim_text,
                "timestamp": evidence.claim.timestamp.formatted,
                "source": evidence.claim.source,
                "verdict": verdict.dict(),
                "evidence": [e.dict() for e in retrieved_evidence[:3]],
                "multimodal_context": {
                    "audio_segments": len(evidence.audio_evidence),
                    "supporting_frames": len(evidence.supporting_frames),
                    "temporal_consistency": evidence.temporal_consistency,
                    "multimodal_alignment": evidence.multimodal_alignment
                },
                "easy_explain": simple_exp
            }
            
            verdict_results.append(result)
            
            print(f"    ✅ Verdict: {verdict.verdict} ({verdict.confidence:.0%})")
            
        except Exception as e:
            print(f"    ❌ Fact-check failed: {e}")
            verdict_results.append({
                "claim": claim_text,
                "error": str(e)
            })
    
    print(f"  🎯 Completed fact-checking for {len(verdict_results)} claims")
    
    return {"verdict_results": verdict_results}


# === 9. Final Report Generation ===

def generate_report_node(state: VideoGraphState) -> dict:
    """
    Generate final comprehensive report
    Node 7: Report Generation
    """
    print(f"\n📊 Generating final video analysis report...")
    
    metadata = state.get("video_metadata")
    verdict_results = state.get("verdict_results", [])
    
    # Aggregate statistics
    total_claims = len(verdict_results)
    verdicts_count = {}
    for result in verdict_results:
        if "verdict" in result:
            verdict = result["verdict"].get("verdict", "UNKNOWN")
            verdicts_count[verdict] = verdicts_count.get(verdict, 0) + 1
    
    # Calculate overall credibility score
    credibility_scores = []
    for result in verdict_results:
        if "verdict" in result:
            v = result["verdict"].get("verdict", "")
            conf = result["verdict"].get("confidence", 0)
            
            if v == "LIKELY_TRUE":
                credibility_scores.append(conf)
            elif v == "LIKELY_FALSE":
                credibility_scores.append(1 - conf)
            else:
                credibility_scores.append(0.5)
    
    overall_credibility = sum(credibility_scores) / len(credibility_scores) if credibility_scores else 0.5
    
    report = {
        "video_info": {
            "filename": metadata.filename if metadata else "Unknown",
            "duration": f"{metadata.duration:.1f}s" if metadata else "Unknown",
            "resolution": f"{metadata.resolution[0]}x{metadata.resolution[1]}" if metadata else "Unknown",
            "video_hash": metadata.video_hash if metadata else "Unknown"
        },
        "analysis_summary": {
            "total_claims": total_claims,
            "verdicts_breakdown": verdicts_count,
            "overall_credibility": overall_credibility,
            "credibility_label": (
                "HIGH" if overall_credibility > 0.7 else
                "MEDIUM" if overall_credibility > 0.4 else
                "LOW"
            )
        },
        "claims_analysis": verdict_results,
        "processing_metadata": {
            "keyframes_extracted": len(state.get("analyzed_keyframes", [])),
            "audio_segments": len(state.get("audio_segments", [])),
            "language_detected": state.get("detected_language", "unknown")
        }
    }
    
    print(f"  ✅ Report generated")
    print(f"  📈 Overall credibility: {overall_credibility:.0%} ({report['analysis_summary']['credibility_label']})")
    
    return {"final_report": report}


# === 10. Build the Pipeline Graph ===

def build_video_pipeline() -> StateGraph:
    """Build the video analysis pipeline graph"""
    
    workflow = StateGraph(VideoGraphState)
    
    # Add nodes
    workflow.add_node("MediaSeparation", separate_media_node)
    workflow.add_node("AudioTranscription", transcribe_audio_node)
    workflow.add_node("VisualAnalysis", analyze_keyframes_node)
    workflow.add_node("ClaimExtraction", extract_video_claims_node)
    workflow.add_node("MultimodalFusion", fuse_multimodal_evidence_node)
    workflow.add_node("FactChecking", fact_check_claims_node)
    workflow.add_node("ReportGeneration", generate_report_node)
    
    # Define flow
    workflow.set_entry_point("MediaSeparation")
    
    # Parallel processing of audio and visual
    workflow.add_edge("MediaSeparation", "AudioTranscription")
    workflow.add_edge("MediaSeparation", "VisualAnalysis")
    
    # Both converge to claim extraction
    workflow.add_edge("AudioTranscription", "ClaimExtraction")
    workflow.add_edge("VisualAnalysis", "ClaimExtraction")
    
    # Sequential processing
    workflow.add_edge("ClaimExtraction", "MultimodalFusion")
    workflow.add_edge("MultimodalFusion", "FactChecking")
    workflow.add_edge("FactChecking", "ReportGeneration")
    workflow.add_edge("ReportGeneration", END)
    
    return workflow.compile()


# === 11. Initialize Pipeline ===

video_pipeline = build_video_pipeline()


# === 12. Convenience Function ===

def analyze_video(video_path: str) -> Dict[str, Any]:
    """
    Analyze a video file for misinformation
    
    Args:
        video_path: Path to video file
        
    Returns:
        Analysis report dictionary
    """
    print("=" * 80)
    print("🎥 VERITAS AI - VIDEO ANALYSIS PIPELINE")
    print("=" * 80)
    
    result = video_pipeline.invoke({"video_path": video_path})
    
    if result.get("error"):
        print(f"\n❌ Error: {result['error']}")
        return result
    
    print("\n" + "=" * 80)
    print("✅ VIDEO ANALYSIS COMPLETE")
    print("=" * 80)
    
    return result.get("final_report", result)


# === 13. Testing ===

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python video_pipeline.py <video_path>")
        print("Example: python video_pipeline.py /path/to/video.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        sys.exit(1)
    
    # Run analysis
    report = analyze_video(video_path)
    
    # Save report
    output_path = "video_analysis_report.json"
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n💾 Report saved to: {output_path}")
