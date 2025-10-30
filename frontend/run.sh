#!/bin/bash

# VeritasAI Frontend Launcher
# This script sets up and runs the Streamlit frontend

set -e

echo "🔍 VeritasAI Frontend Launcher"
echo "=============================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we're in the frontend directory
if [ ! -f "app.py" ]; then
    echo -e "${RED}❌ Error: app.py not found${NC}"
    echo "Please run this script from the frontend/ directory"
    exit 1
fi

# Check for .env file in parent code_arka directory
if [ ! -f "../code_arka/.env" ]; then
    echo -e "${YELLOW}⚠️  Warning: .env file not found in ../code_arka/${NC}"
    echo ""
    echo "Please create a .env file with:"
    echo "GOOGLE_API_KEY=your_api_key_here"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}📦 Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${GREEN}🔧 Activating virtual environment...${NC}"
source venv/bin/activate

# Install/update dependencies
echo -e "${GREEN}📥 Installing dependencies...${NC}"
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Check if Streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo -e "${RED}❌ Streamlit installation failed${NC}"
    exit 1
fi

# Verify backend imports
echo -e "${GREEN}🔍 Verifying backend pipelines...${NC}"
python3 -c "
import sys
sys.path.append('../code_arka')
try:
    from multimodal_pipeline import multimodal_pipeline
    from text_pipeline import text_pipeline
    from image_pipeline import image_pipeline
    print('✅ All pipelines imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
" || exit 1

echo ""
echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo "=============================="
echo "🚀 Starting VeritasAI Frontend"
echo "=============================="
echo ""
echo "The app will open in your browser at:"
echo "👉 http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run Streamlit
streamlit run app.py
