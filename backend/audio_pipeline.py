"""
Audio Analysis Pipeline for VeritasAI
Transcribes audio using OpenAI Whisper and analyzes claims
"""

from typing import TypedDict, Optional, Dict, Any
import os
import sys
from pathlib import Path
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# Import Whisper for transcription
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️ Whisper not installed. Run: pip install openai-whisper")

# Import text pipeline for claim analysis
from backend.text_pipeline import text_pipeline

load_dotenv()


# === 1. Define the State Schema
class AudioGraphState(TypedDict):
    audio_path: str
    transcription: Optional[str]
    transcription_confidence: Optional[float]
    audio_duration: Optional[float]
    text_analysis_result: Optional[Dict[str, Any]]
    error: Optional[str]


# === 2. Define Pipeline Nodes

def transcribe_audio_node(state: AudioGraphState) -> dict:
    """
    Transcribe audio file using OpenAI Whisper
    
    Args:
        state: Current graph state with audio_path
        
    Returns:
        Updated state with transcription
    """
    audio_path = state["audio_path"]
    
    print(f"\n🎤 Transcribing audio: {audio_path}")
    
    # Check if file exists
    if not os.path.exists(audio_path):
        return {
            "error": f"Audio file not found: {audio_path}",
            "transcription": None
        }
    
    # Check if Whisper is available
    if not WHISPER_AVAILABLE:
        return {
            "error": "Whisper not installed. Install with: pip install openai-whisper",
            "transcription": None
        }
    
    try:
        # Load Whisper model
        # Options: 'tiny', 'base', 'small', 'medium', 'large'
        # 'base' is a good balance between speed and accuracy
        model_name = os.getenv("WHISPER_MODEL", "base")
        print(f"  📦 Loading Whisper model: {model_name}")
        
        model = whisper.load_model(model_name)
        
        # Transcribe audio
        print(f"  🔄 Transcribing...")
        result = model.transcribe(audio_path, fp16=False)
        
        transcription = result["text"].strip()
        
        # Extract additional metadata if available
        language = result.get("language", "unknown")
        
        print(f"  ✅ Transcription complete")
        print(f"  🌐 Detected language: {language}")
        print(f"  📝 Text length: {len(transcription)} characters")
        
        return {
            "transcription": transcription,
            "error": None
        }
        
    except Exception as e:
        error_msg = f"Transcription failed: {str(e)}"
        print(f"  ❌ {error_msg}")
        return {
            "error": error_msg,
            "transcription": None
        }


def analyze_claims_node(state: AudioGraphState) -> dict:
    """
    Run transcribed text through text analysis pipeline
    
    Args:
        state: Current graph state with transcription
        
    Returns:
        Updated state with text analysis results
    """
    transcription = state.get("transcription")
    
    if not transcription:
        print("\n⚠️ No transcription available, skipping claim analysis")
        return {
            "text_analysis_result": None
        }
    
    print(f"\n🔍 Analyzing claims in transcribed text...")
    print(f"  📝 Text preview: {transcription[:100]}...")
    
    try:
        # Run text pipeline on transcription
        result = text_pipeline.invoke({"content": transcription})
        
        print(f"  ✅ Claim analysis complete")
        
        # Extract key information
        num_claims = len(result.get("claims", {}).claims) if result.get("claims") else 0
        num_fused = len(result.get("fused_claims", {}).fused_claims) if result.get("fused_claims") else 0
        num_evidences = len(result.get("evidence_results", []))
        
        print(f"  📊 Found {num_claims} claims, {num_fused} fused claims")
        print(f"  🔍 Retrieved evidence for {num_evidences} claims")
        
        return {
            "text_analysis_result": result
        }
        
    except Exception as e:
        error_msg = f"Claim analysis failed: {str(e)}"
        print(f"  ❌ {error_msg}")
        return {
            "error": error_msg,
            "text_analysis_result": None
        }


# === 3. Build the Graph

graph = StateGraph(AudioGraphState)

# Add nodes
graph.add_node("TranscribeAudio", transcribe_audio_node)
graph.add_node("AnalyzeClaims", analyze_claims_node)

# Define edges
graph.add_edge("TranscribeAudio", "AnalyzeClaims")
graph.add_edge("AnalyzeClaims", END)

# Set entry point
graph.set_entry_point("TranscribeAudio")

# Compile the graph
audio_pipeline = graph.compile()


# === 4. Convenience Functions

def analyze_audio_file(
    audio_path: str,
    save_results: bool = True,
    output_dir: str = "./audio_results"
) -> Dict[str, Any]:
    """
    Analyze an audio file for misinformation
    
    Args:
        audio_path: Path to audio file
        save_results: Whether to save results to JSON
        output_dir: Directory to save results
        
    Returns:
        Dictionary with transcription and analysis results
    """
    print("=" * 70)
    print("AUDIO ANALYSIS PIPELINE")
    print("=" * 70)
    
    # Run pipeline
    result = audio_pipeline.invoke({"audio_path": audio_path})
    
    # Format output
    output = {
        "audio_file": audio_path,
        "transcription": result.get("transcription"),
        "error": result.get("error"),
        "analysis": None
    }
    
    # Add text analysis if available
    if result.get("text_analysis_result"):
        text_result = result["text_analysis_result"]
        
        output["analysis"] = {
            "source_language": text_result.get("source_language"),
            "extracted_claims": [
                {
                    "statement": claim.statement,
                    "category": claim.category,
                    "checkability_score": claim.checkability_score
                }
                for claim in text_result.get("claims", {}).claims
            ] if text_result.get("claims") else [],
            "fused_claims": [
                {
                    "statement": claim.fused_statement,
                    "original_claim_ids": claim.original_claim_ids,
                    "evidence_links": claim.evidence_links
                }
                for claim in text_result.get("fused_claims", {}).fused_claims
            ] if text_result.get("fused_claims") else [],
            "evidence_results": text_result.get("evidence_results", []),
            "verdict": text_result.get("verdict"),  # Include verdict from text pipeline
            "easy_explain": text_result.get("easy_explain")  # Include simple explanation
        }
    
    # Save to JSON if requested
    if save_results and not result.get("error"):
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            json_path = os.path.join(output_dir, f"{base_name}_analysis.json")
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"\n💾 Results saved to: {json_path}")
            
        except Exception as e:
            print(f"\n⚠️ Failed to save results: {e}")
    
    return output


def transcribe_only(audio_path: str, save_to_json: bool = True) -> str:
    """
    Transcribe audio without running claim analysis
    
    Args:
        audio_path: Path to audio file
        save_to_json: Whether to save transcription to JSON
        
    Returns:
        Transcription text
    """
    if not WHISPER_AVAILABLE:
        raise ImportError("Whisper not installed. Run: pip install openai-whisper")
    
    print(f"🎤 Transcribing: {audio_path}")
    
    # Load model
    model_name = os.getenv("WHISPER_MODEL", "base")
    model = whisper.load_model(model_name)
    
    # Transcribe
    result = model.transcribe(audio_path, fp16=False)
    transcription = result["text"].strip()
    
    print(f"✅ Transcription complete ({len(transcription)} characters)")
    
    # Save to JSON if requested
    if save_to_json:
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        json_path = f"{base_name}_transcription.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({"audio": transcription}, f, ensure_ascii=False, indent=4)
        
        print(f"💾 Saved to: {json_path}")
    
    return transcription


# === 5. Run the pipeline (if executed directly)

if __name__ == "__main__":
    import sys
    
    # Example audio files
    test_audio_files = [
        "files/hello_so_gaye.mp3",
        "files/ytmp3free.cc_taj-mahal-was-an-old-hindu-temple-rss-thinker-on-the-big-fight-youtubemp3free.org.mp3"
    ]
    
    # Use command line argument if provided
    if len(sys.argv) > 1:
        test_audio_files = [sys.argv[1]]
    
    for audio_file in test_audio_files:
        if not os.path.exists(audio_file):
            print(f"⚠️ File not found: {audio_file}")
            continue
        
        # Run full analysis
        result = analyze_audio_file(audio_file)
        
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        
        if result.get("error"):
            print(f"❌ Error: {result['error']}")
        else:
            print(f"📝 Transcription: {result['transcription'][:200]}...")
            
            if result.get("analysis"):
                claims = result["analysis"]["extracted_claims"]
                print(f"\n📊 Claims found: {len(claims)}")
                for i, claim in enumerate(claims, 1):
                    print(f"  {i}. {claim['statement']}")
        
        print("\n")
