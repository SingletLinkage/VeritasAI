# VeritasAI - Multimodal Misinformation Detection System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-121212?logo=chainlink&logoColor=white)](https://langchain.com/)

A comprehensive AI-powered fact-checking system that analyzes **multimodal content** to detect misinformation using advanced LLMs, RAG (Retrieval-Augmented Generation), and computer vision techniques.


## Overview

**VeritasAI** is a state-of-the-art misinformation detection platform that combines multiple AI technologies to provide comprehensive fact-checking capabilities. The system can:

- ✅ **Analyze text claims** with multi-language support (90+ languages)
- ✅ **Transcribe and verify audio content** using OpenAI Whisper
- ✅ **Detect image manipulations** including deepfakes and AI-generated content
- ✅ **Verify multimodal content** by cross-referencing text and images
- ✅ **Retrieve evidence** from local vector stores and web searches
- ✅ **Generate detailed verdicts** with confidence scores and explanations

### System Pipeline and Architecture

![VeritasAI Architecture](pipeline.jpg)

---
## Key Features

### 🔤 **Text Analysis Pipeline**
- **Multi-language Detection**: Automatic detection of 90+ languages
- **Claim Extraction**: AI-powered extraction of verifiable claims
- **Claim Fusion**: Intelligent merging of redundant claims
- **Evidence Retrieval**: Hybrid search combining local vector store (FAISS) and real web search
- **LLM Reranking**: Advanced evidence ranking using Gemini 2.0 Flash Lite

### 🎤 **Audio Analysis Pipeline**
- **Speech-to-Text**: Automatic transcription using OpenAI Whisper
- **Multi-language Support**: Supports 90+ languages in audio
- **Claim Analysis**: Transcribed audio analyzed through text pipeline
- **Evidence Retrieval**: Full fact-checking on spoken claims

### 🖼️ **Image Analysis Pipeline**
- **EXIF Metadata Extraction**: Camera settings, GPS, timestamps
- **Reverse Image Search**: Find similar images across the web using real SerpAPI
- **Deepfake Detection**: AI-generated content identification
- **Manipulation Detection**: Edit analysis and authenticity scoring
- **Visual Evidence**: Comparison with similar images
  
### 🎥 **Video Analysis Pipeline**
- **Media Separation**: Automatic audio and keyframe extraction
- **Audio Transcription**: Whisper-based speech-to-text with timestamps
- **Visual Analysis**: Keyframe captioning, entity recognition, deepfake detection
- **Temporal Fusion**: Synchronized audio-visual evidence alignment
- **Timeline-based Fact-Checking**: Claims mapped to video timestamps
- **Multimodal Context**: Audio + visual + external evidence verification

### 🔀 **Multimodal Pipeline**
- **Cross-Modal Verification**: Text-image consistency checking
- **Context Analysis**: Relationship between claims and visuals
- **Comprehensive Verdicts**: Holistic misinformation assessment

### 👵👴 **Explainability & Accessibility ("Explain Like I'm 60")**
- **Simple Language**: Grade-5 reading level explanations for all verdicts
- **Respectful Tone**: Culturally appropriate greetings (Uncle/Aunty)
- **Clear Actions**: Easy-to-follow steps for what to do
- **Multi-language**: Explanations in user's native language
- **Accessibility**: Makes fact-checking understandable for users 60+ with limited digital literacy

### 🌐 **Web Interface**
- **Beautiful Streamlit UI**: Modern, responsive design
- **Multiple Analysis Modes**: Text, Text+Image, Audio, Video (coming soon)
- **Real-time Analysis**: Live progress indicators
- **Export Options**: JSON and text report downloads
- **Interactive Results**: Expandable sections and detailed breakdowns
- **Dual Explanations**: Technical + Simple explanations for every verdict

---


## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/SingletLinkage/VeritasAI.git
cd VeritasAI
```

### Step 2: Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

```

### Step 3: Environment Setup

Create a `.env` file in the **project root directory**:

```bash
# Required
GOOGLE_API_KEY=your_google_gemini_api_key_here

# Required for image reverse search
SERPAPI_API_KEY=your_serpapi_key_here
```

### Step 4: Populate Vector Store

```bash
cd backend
python3 populate_vector_store.py
```

---

## 📁 Project Structure

```
ihub/
├── README.md                         # Main documentation
├── requirements.txt                  # Python dependencies
├── pipeline.jpg                      # System architecture diagram
│
├── backend/                          # Core system
│   ├── .env                          # Environment variables
│   │
│   ├── text_pipeline.py              # Text analysis pipeline
│   ├── audio_pipeline.py             # Audio analysis pipeline
│   ├── image_pipeline.py             # Image analysis pipeline
│   ├── multimodal_pipeline.py        # Combined analysis
│   │
│   ├── hybrid_retrieval.py           # Main retrieval orchestrator
│   ├── vector_store.py               # FAISS vector store manager
│   ├── web_search.py                 # Real SerpAPI web search
│   ├── reranker.py                   # LLM-based evidence ranking
│   │
│   ├── models.py                     # Pydantic models (text/fusion)
│   ├── retrieval_models.py           # Pydantic models (evidence)
│   ├── prompts.py                    # LLM prompts
│   │
│   ├── exif_tool.py                  # EXIF metadata extraction
│   ├── ocr_tool.py                   # OCR for images
│   ├── rev_search_tool.py            # Reverse image search
│   ├── explainability.py             # "Explain Like I'm 60" module
│   │
│   ├── populate_vector_store.py      # Data population script
│   ├── visualize_pipeline.py         # Generate pipeline diagrams
│   ├── video_pipeline_graph.mmd      # Video pipeline diagram 
│   │
│   ├── web_scrappers/                # Scraped fact-checking data
│   │   ├── who_scrapper.py           # WHO scraper script
│   │   ├── fact_check_scraper.py     # FactCheck.org scraper
│   │   ├── pti_html_parser.py        # PTI parser script
│   │   └── rbi_scrapper.py           # RBI scraper script
│   │
│   └── data/                         # Data storage
│       └── vector_store/             # FAISS index storage
│           ├── index.faiss           # Vector index (generated)
│           └── index.pkl             # Metadata (generated)
│
└── frontend/                         # Web interface
    ├── app.py                        # Streamlit application
    └── requirements.txt              # Frontend dependencies

```
---

## 🚀 Running the Application

```bash
# Make sure you're in the project root directory
cd /path/to/directory

# Run the Streamlit app
streamlit run frontend/app.py
```

The app will open in your browser at `http://localhost:8501`

--- 

## Team Members
- Arka Mukhopadhyay
- Paridhi Mittal
- Piyush Dwivedi
- Yug Goyal

## License
This project was created as part of the IIT Mandi iHub Multimodality Hackathon.
