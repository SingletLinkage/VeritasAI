"""
VeritasAI Frontend - Test Script
Tests the frontend setup without launching the full UI
"""

import sys
from pathlib import Path

# Add parent directory to path to access backend module
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test all required imports"""
    print("🧪 Testing Frontend Imports...")
    print("="*60)
    
    tests = []
    
    # Test Streamlit
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__}")
        tests.append(True)
    except ImportError as e:
        print(f"❌ Streamlit: {e}")
        tests.append(False)
    
    # Test Pillow
    try:
        from PIL import Image
        print(f"✅ Pillow (PIL)")
        tests.append(True)
    except ImportError as e:
        print(f"❌ Pillow: {e}")
        tests.append(False)
    
    # Test pipelines
    try:
        from backend.multimodal_pipeline import multimodal_pipeline
        print("✅ Multimodal Pipeline")
        tests.append(True)
    except ImportError as e:
        print(f"❌ Multimodal Pipeline: {e}")
        tests.append(False)
        from traceback import print_exc
        print_exc()
    
    try:
        from backend.text_pipeline import text_pipeline
        print("✅ Text Pipeline")
        tests.append(True)
    except ImportError as e:
        print(f"❌ Text Pipeline: {e}")
        tests.append(False)
    
    try:
        from backend.image_pipeline import image_pipeline
        print("✅ Image Pipeline")
        tests.append(True)
    except ImportError as e:
        print(f"❌ Image Pipeline: {e}")
        tests.append(False)
    
    # Test models
    try:
        from backend.models import VerdictResult
        print("✅ Pydantic Models")
        tests.append(True)
    except ImportError as e:
        print(f"❌ Models: {e}")
        tests.append(False)
    
    print("="*60)
    
    if all(tests):
        print("\n✅ All tests passed!")
        print("\n🚀 Ready to launch frontend:")
        print("   streamlit run app.py")
        return True
    else:
        print(f"\n❌ {tests.count(False)} test(s) failed")
        print("\n💡 Fix issues above before launching")
        return False


def check_environment():
    """Check environment setup"""
    print("\n🔧 Checking Environment...")
    print("="*60)
    
    # Check .env file - now in backend folder
    env_path = Path(__file__).parent.parent / "backend" / ".env"
    if not env_path.exists():
        # Try old location as fallback
        env_path = Path(__file__).parent.parent / "code_arka" / ".env"
    
    if env_path.exists():
        print(f"✅ .env file found: {env_path}")
        
        # Check for API key
        with open(env_path) as f:
            content = f.read()
            if "GOOGLE_API_KEY" in content:
                print("✅ GOOGLE_API_KEY configured")
            else:
                print("⚠️  GOOGLE_API_KEY not found in .env")
    else:
        print(f"⚠️  .env file not found: {env_path}")
        print("   Create it with: GOOGLE_API_KEY=your_key")
    
    # Check config
    config_path = Path(__file__).parent / ".streamlit" / "config.toml"
    if config_path.exists():
        print(f"✅ Streamlit config found: {config_path}")
    else:
        print(f"⚠️  Streamlit config not found (using defaults)")
    
    print("="*60)


def show_usage():
    """Show usage instructions"""
    print("\n📚 Usage Instructions")
    print("="*60)
    print("""
1. Install dependencies:
   pip install streamlit pillow

2. Configure API key:
   echo "GOOGLE_API_KEY=your_key" > ../code_arka/.env

3. Launch frontend:
   streamlit run app.py
   
4. Open browser:
   http://localhost:8501

5. Analyze content:
   - Enter text claim
   - Upload image (optional)
   - Click "Analyze Content"
   - Review verdict and evidence
    """)
    print("="*60)


def main():
    """Main test function"""
    print("\n" + "="*60)
    print("🔍 VeritasAI Frontend Test Suite")
    print("="*60 + "\n")
    
    # Run tests
    imports_ok = test_imports()
    check_environment()
    
    if imports_ok:
        show_usage()
        print("\n✅ Frontend is ready to launch!")
    else:
        print("\n❌ Please fix errors before launching")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
