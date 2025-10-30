"""
VeritasAI - Multimodal Misinformation Detection System
Beautiful Streamlit Frontend
"""

import streamlit as st
import sys
from pathlib import Path
import tempfile
import json
from datetime import datetime
from typing import Optional, Dict, Any
import time

# Add parent directory to path to import backend modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.multimodal_pipeline import multimodal_pipeline
from backend.text_pipeline import text_pipeline
from backend.image_pipeline import image_pipeline

# Page configuration
st.set_page_config(
    page_title="VeritasAI - Fact Checker",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --info-color: #3b82f6;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .main-header p {
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Verdict cards */
    .verdict-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border-left: 5px solid;
    }
    
    .verdict-true {
        background-color: #d1fae5;
        border-left-color: #10b981;
    }
    
    .verdict-false {
        background-color: #fee2e2;
        border-left-color: #ef4444;
    }
    
    .verdict-misleading {
        background-color: #fef3c7;
        border-left-color: #f59e0b;
    }
    
    .verdict-insufficient {
        background-color: #e0e7ff;
        border-left-color: #6366f1;
    }
    
    /* Evidence cards */
    .evidence-card {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #6366f1;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Progress animation */
    .analysis-progress {
        text-align: center;
        padding: 2rem;
    }
    
    /* Red flags */
    .red-flag {
        background-color: #fef2f2;
        border-left: 3px solid #ef4444;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 6px;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #eff6ff;
        border-left: 3px solid #3b82f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }
    
    /* File uploader */
    .uploadedFile {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
    }
    
    /* EXIF metadata */
    .exif-data {
        background-color: #f1f5f9;
        padding: 1rem;
        border-radius: 8px;
        font-family: monospace;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


def render_header():
    """Render the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🔍 VeritasAI</h1>
        <p>Advanced Multimodal Misinformation Detection System</p>
    </div>
    """, unsafe_allow_html=True)


def get_verdict_color(verdict: str) -> str:
    """Get color for verdict"""
    colors = {
        "LIKELY_TRUE": "#10b981",
        "LIKELY_FALSE": "#ef4444",
        "MISLEADING": "#f59e0b",
        "INSUFFICIENT": "#6366f1"
    }
    return colors.get(verdict, "#6366f1")


def get_verdict_emoji(verdict: str) -> str:
    """Get emoji for verdict"""
    emojis = {
        "LIKELY_TRUE": "✅",
        "LIKELY_FALSE": "❌",
        "MISLEADING": "⚠️",
        "INSUFFICIENT": "❓"
    }
    return emojis.get(verdict, "❓")


def get_verdict_class(verdict: str) -> str:
    """Get CSS class for verdict"""
    classes = {
        "LIKELY_TRUE": "verdict-true",
        "LIKELY_FALSE": "verdict-false",
        "MISLEADING": "verdict-misleading",
        "INSUFFICIENT": "verdict-insufficient"
    }
    return classes.get(verdict, "verdict-insufficient")


def render_verdict_card(verdict_result: Any):
    """Render the main verdict card"""
    # Handle both dict and Pydantic model
    if hasattr(verdict_result, 'verdict'):
        verdict = verdict_result.verdict
        confidence = verdict_result.confidence
        reasoning = verdict_result.reasoning
        red_flags = verdict_result.red_flags if hasattr(verdict_result, 'red_flags') else []
        recommendation = verdict_result.recommendation if hasattr(verdict_result, 'recommendation') else ""
    else:
        verdict = verdict_result.get('verdict', 'INSUFFICIENT')
        confidence = verdict_result.get('confidence', 0)
        reasoning = verdict_result.get('reasoning', 'No reasoning provided')
        red_flags = verdict_result.get('red_flags', [])
        recommendation = verdict_result.get('recommendation', '')
    
    verdict_class = get_verdict_class(verdict)
    verdict_emoji = get_verdict_emoji(verdict)
    
    st.markdown(f"""
    <div class="verdict-card {verdict_class}">
        <h2>{verdict_emoji} {verdict.replace('_', ' ').title()}</h2>
        <h3>Confidence: {confidence:.1%}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Reasoning
    st.markdown("### 💭 Analysis")
    st.markdown(f"""
    <div class="info-box">
        {reasoning}
    </div>
    """, unsafe_allow_html=True)
    
    # Red flags
    if red_flags:
        st.markdown("### 🚩 Red Flags Detected")
        for flag in red_flags:
            st.markdown(f"""
            <div class="red-flag">
                ⚠️ {flag}
            </div>
            """, unsafe_allow_html=True)
    
    # Recommendation
    if recommendation:
        st.markdown("### 💡 Recommendation")
        st.markdown(f"""
        <div class="info-box">
            {recommendation}
        </div>
        """, unsafe_allow_html=True)


def render_evidence(evidence_list: list):
    """Render evidence cards"""
    if not evidence_list:
        st.info("No evidence retrieved")
        return
    
    st.markdown("### 📚 Supporting Evidence")
    
    for i, evidence in enumerate(evidence_list, 1):
        if isinstance(evidence, dict):
            source = evidence.get('source', 'Unknown source')
            content = evidence.get('content', evidence.get('text', 'No content'))
            relevance = evidence.get('relevance_score', evidence.get('score', 0))
        else:
            source = "Evidence source"
            content = str(evidence)
            relevance = 0
        
        with st.expander(f"📄 Evidence {i}: {source[:80]}{'...' if len(source) > 80 else ''}", expanded=(i == 1)):
            st.markdown(f"""
            <div class="evidence-card">
                <p><strong>Source:</strong> {source}</p>
                <p><strong>Relevance:</strong> {relevance:.2%}</p>
                <hr>
                <p>{content}</p>
            </div>
            """, unsafe_allow_html=True)


def render_image_analysis(result: Dict[str, Any]):
    """Render image analysis results"""
    st.markdown("### 🖼️ Image Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Caption
        if result.get('caption'):
            st.markdown("#### 📝 Image Description")
            st.info(result['caption'])
        
        # Deepfake detection
        if result.get('deepfake_result'):
            df_result = result['deepfake_result']
            st.markdown("#### 🔍 Authenticity Check")
            
            is_suspicious = df_result.get('is_suspicious', False)
            confidence = df_result.get('confidence_score', 0)
            
            if is_suspicious:
                st.error(f"⚠️ Suspicious content detected (Confidence: {confidence:.1%})")
            else:
                st.success(f"✅ No manipulation detected (Confidence: {confidence:.1%})")
            
            if df_result.get('red_flags'):
                st.markdown("**Indicators:**")
                for flag in df_result['red_flags']:
                    st.markdown(f"- {flag}")
    
    with col2:
        # EXIF metadata
        if result.get('exif_data'):
            st.markdown("#### 📷 Camera Metadata (EXIF)")
            exif = result['exif_data']
            
            exif_info = []
            if exif.get('Make') or exif.get('Model'):
                exif_info.append(f"**Camera:** {exif.get('Make', '')} {exif.get('Model', '')}")
            if exif.get('DateTime'):
                exif_info.append(f"**Date:** {exif.get('DateTime')}")
            if exif.get('Software'):
                exif_info.append(f"**Software:** {exif.get('Software')}")
            if exif.get('GPS'):
                exif_info.append(f"**GPS:** {exif.get('GPS')}")
            
            if exif_info:
                st.markdown('<div class="exif-data">', unsafe_allow_html=True)
                for info in exif_info:
                    st.markdown(info)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("No EXIF metadata found")


def render_text_analysis(result: Dict[str, Any]):
    """Render text analysis results"""
    st.markdown("### 📝 Text Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        language = result.get('language', 'Unknown')
        st.metric("Language Detected", language.upper())
    
    with col2:
        claims = result.get('claims', [])
        st.metric("Claims Extracted", len(claims))
    
    with col3:
        evidence = result.get('evidence', [])
        st.metric("Evidence Sources", len(evidence))
    
    # Show extracted claims
    if claims:
        st.markdown("#### 🎯 Extracted Claims")
        for i, claim in enumerate(claims, 1):
            claim_text = claim if isinstance(claim, str) else claim.get('claim', str(claim))
            st.markdown(f"{i}. {claim_text}")


def process_text_only(text: str):
    """Process text-only input"""
    with st.spinner("🔍 Analyzing text claim..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Detecting language...")
        progress_bar.progress(20)
        time.sleep(0.3)
        
        status_text.text("Extracting claims...")
        progress_bar.progress(50)
        
        result = text_pipeline.invoke({
            "content": text,
        })
        
        status_text.text("Retrieving evidence...")
        progress_bar.progress(80)
        time.sleep(0.3)
        
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()
    
    return result


def process_multimodal(text: str, image_path: Optional[str] = None):
    """Process text + image input"""
    with st.spinner("🔍 Running multimodal analysis..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Detecting language...")
        progress_bar.progress(15)
        time.sleep(0.3)
        
        status_text.text("Extracting claims...")
        progress_bar.progress(30)
        time.sleep(0.3)
        
        if image_path:
            status_text.text("Analyzing image...")
            progress_bar.progress(50)
            time.sleep(0.5)
            
            status_text.text("Checking for manipulation...")
            progress_bar.progress(70)
            time.sleep(0.5)
        
        result = multimodal_pipeline.invoke({
            "content": text,
            "image_path": image_path,
        })
        
        status_text.text("Consolidating evidence...")
        progress_bar.progress(90)
        time.sleep(0.3)
        
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()
    
    return result


def main():
    """Main application"""
    render_header()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        analysis_mode = st.radio(
            "Analysis Mode",
            ["Text Only", "Text + Image", "Image Only"],
            help="Choose what type of content to analyze"
        )
        
        st.markdown("---")
        
        st.markdown("### ℹ️ About")
        st.info("""
        **VeritasAI** uses advanced AI models to detect misinformation by:
        
        - 🌍 Multi-language support
        - 📝 Claim extraction
        - 🖼️ Image authenticity verification
        - 📷 EXIF metadata analysis
        - 🔍 Evidence retrieval
        - 🤖 Cross-modal fusion
        """)
        
        st.markdown("---")
        st.markdown("### 🎨 Features")
        st.markdown("""
        ✅ Text fact-checking  
        ✅ Deepfake detection  
        ✅ EXIF analysis  
        ✅ Multi-language support  
        ✅ Evidence retrieval  
        ✅ Confidence scoring  
        """)
    
    # Main content area
    st.markdown("## 📋 Input Content")
    
    text_input = None
    image_file = None
    uploaded_image_path = None
    
    # Text input
    if analysis_mode in ["Text Only", "Text + Image"]:
        text_input = st.text_area(
            "Enter text claim to verify:",
            height=150,
            placeholder="Example: 'Breaking news: Major event happened at this location...'",
            help="Enter any claim, news, or statement you want to fact-check"
        )
    
    # Image upload
    if analysis_mode in ["Text + Image", "Image Only"]:
        st.markdown("### 📎 Upload Media")
        
        upload_tab1, upload_tab2, upload_tab3 = st.tabs(["🖼️ Image", "🎥 Video", "🎵 Audio"])
        
        with upload_tab1:
            image_file = st.file_uploader(
                "Upload image",
                type=['jpg', 'jpeg', 'png', 'webp'],
                help="Upload an image to analyze for manipulation and verify against claims"
            )
            
            if image_file:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.image(image_file, caption="Uploaded Image", use_column_width=True)
                
                with col2:
                    st.success("✅ Image uploaded successfully")
                    st.info(f"""
                    **File:** {image_file.name}  
                    **Size:** {image_file.size / 1024:.1f} KB  
                    **Type:** {image_file.type}
                    """)
        
        with upload_tab2:
            video_file = st.file_uploader(
                "Upload video",
                type=['mp4', 'avi', 'mov', 'mkv'],
                help="Upload a video (keyframe extraction coming soon)"
            )
            if video_file:
                st.warning("⚠️ Video analysis coming soon! For now, extract a keyframe and upload as image.")
        
        with upload_tab3:
            audio_file = st.file_uploader(
                "Upload audio",
                type=['mp3', 'wav', 'ogg', 'm4a'],
                help="Upload audio for transcription (coming soon)"
            )
            if audio_file:
                st.warning("⚠️ Audio transcription coming soon!")
    
    # Analyze button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        analyze_button = st.button("🚀 Analyze Content", use_container_width=True)
    
    # Process analysis
    if analyze_button:
        # Validation
        if analysis_mode == "Text Only" and not text_input:
            st.error("❌ Please enter text to analyze")
            return
        
        if analysis_mode == "Image Only" and not image_file:
            st.error("❌ Please upload an image to analyze")
            return
        
        if analysis_mode == "Text + Image":
            if not text_input:
                st.error("❌ Please enter text to analyze")
                return
            if not image_file:
                st.warning("⚠️ No image uploaded. Running text-only analysis...")
        
        # Save uploaded image to temp file
        if image_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(image_file.name).suffix) as tmp_file:
                tmp_file.write(image_file.getvalue())
                uploaded_image_path = tmp_file.name
        
        # Run analysis
        try:
            st.markdown("---")
            st.markdown("## 🔬 Analysis Results")
            
            if analysis_mode == "Text Only":
                result = process_text_only(text_input)
                
                # Show text analysis
                render_text_analysis(result)
                
                # Show evidence
                if result.get('evidence'):
                    render_evidence(result['evidence'])
            
            elif analysis_mode == "Text + Image":
                if not uploaded_image_path:
                    # Fallback to text only
                    result = process_text_only(text_input)
                    render_text_analysis(result)
                    if result.get('evidence'):
                        render_evidence(result['evidence'])
                else:
                    result = process_multimodal(text_input, uploaded_image_path)
                    
                    # Show verdict
                    if result.get('multimodal_verdict'):
                        render_verdict_card(result['multimodal_verdict'])
                    
                    # Tabs for detailed analysis
                    tab1, tab2, tab3 = st.tabs(["📝 Text Analysis", "🖼️ Image Analysis", "📚 Evidence"])
                    
                    with tab1:
                        render_text_analysis(result)
                    
                    with tab2:
                        render_image_analysis(result)
                    
                    with tab3:
                        if result.get('evidence'):
                            render_evidence(result['evidence'])
                        else:
                            st.info("No evidence retrieved for this analysis")
            
            elif analysis_mode == "Image Only":
                if uploaded_image_path:
                    with st.spinner("🔍 Analyzing image..."):
                        result = image_pipeline.invoke({
                            "image_path": uploaded_image_path,
                            "text_claims": [text_input] if text_input else ["Analyze this image"]
                        })
                    
                    render_image_analysis(result)
            
            # Download results
            st.markdown("---")
            st.markdown("### 💾 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # JSON export
                result_json = json.dumps(
                    {k: str(v) for k, v in result.items()},
                    indent=2,
                    default=str
                )
                st.download_button(
                    "📥 Download as JSON",
                    result_json,
                    file_name=f"veritasai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with col2:
                # Text report
                report = f"""
VeritasAI Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*60}
INPUT
{'='*60}
Text: {text_input if text_input else 'N/A'}
Image: {image_file.name if image_file else 'N/A'}

{'='*60}
VERDICT
{'='*60}
{json.dumps({k: str(v) for k, v in result.items()}, indent=2, default=str)}
                """
                
                st.download_button(
                    "📥 Download as TXT",
                    report,
                    file_name=f"veritasai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.exception(e)
        
        finally:
            # Cleanup temp file
            if uploaded_image_path and Path(uploaded_image_path).exists():
                try:
                    Path(uploaded_image_path).unlink()
                except:
                    pass


if __name__ == "__main__":
    main()
