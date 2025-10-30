"""
Pipeline Visualization Tool
Uses LangGraph's native visualization methods to generate diagrams
"""
from pathlib import Path
from typing import Optional


def visualize_pipeline(pipeline, pipeline_name: str, output_dir: str = "."):
    """
    Visualize a LangGraph pipeline using built-in methods
    
    Args:
        pipeline: Compiled LangGraph pipeline
        pipeline_name: Name for the output files
        output_dir: Directory to save visualization files
    
    Returns:
        Dictionary with paths to generated files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    print(f"\n{'='*60}")
    print(f"🎨 Visualizing: {pipeline_name}")
    print(f"{'='*60}\n")
    
    # Get the graph
    graph = pipeline.get_graph()
    
    # 1. Generate Mermaid PNG
    try:
        png_path = output_path / f"{pipeline_name}_graph.png"
        png_data = graph.draw_mermaid_png()
        
        with open(png_path, 'wb') as f:
            f.write(png_data)
        
        print(f"✅ PNG diagram saved: {png_path}")
        results['png'] = str(png_path)
    except Exception as e:
        print(f"⚠️  PNG generation failed: {e}")
        print(f"   Install with: pip install pyppeteer")
    
    # 2. Generate Mermaid code
    try:
        mermaid_code = graph.draw_mermaid()
        mermaid_path = output_path / f"{pipeline_name}_graph.mmd"
        
        with open(mermaid_path, 'w') as f:
            f.write(mermaid_code)
        
        print(f"✅ Mermaid code saved: {mermaid_path}")
        print(f"   View at: https://mermaid.live/")
        results['mermaid'] = str(mermaid_path)
    except Exception as e:
        print(f"⚠️  Mermaid generation failed: {e}")
    
    # 3. Generate ASCII representation
    try:
        ascii_repr = graph.draw_ascii()
        ascii_path = output_path / f"{pipeline_name}_graph.txt"
        
        with open(ascii_path, 'w') as f:
            f.write(ascii_repr)
        
        print(f"✅ ASCII diagram saved: {ascii_path}")
        results['ascii'] = str(ascii_path)
        
        # Also print to console
        print(f"\n{'='*60}")
        print(f"ASCII Diagram:")
        print(f"{'='*60}")
        print(ascii_repr)
        
    except Exception as e:
        print(f"⚠️  ASCII generation failed: {e}")
    
    print(f"\n{'='*60}\n")
    
    return results


def visualize_all():
    """
    Visualize all pipelines (text, image, multimodal)
    """
    print("\n" + "="*60)
    print("🎨 PIPELINE VISUALIZATION TOOL")
    print("="*60)
    
    results = {}
    
    # 1. Text Pipeline
    try:
        print("\n📝 Loading text_pipeline...")
        from text_pipeline import text_pipeline
        results['text'] = visualize_pipeline(text_pipeline, "text_pipeline")
    except Exception as e:
        print(f"❌ Failed to visualize text pipeline: {e}")
    
    # 2. Image Pipeline
    try:
        print("\n🖼️  Loading image_pipeline...")
        from image_pipeline import image_pipeline
        results['image'] = visualize_pipeline(image_pipeline, "image_pipeline")
    except Exception as e:
        print(f"❌ Failed to visualize image pipeline: {e}")
    
    # 3. Multimodal Pipeline
    try:
        print("\n🔗 Loading multimodal_pipeline...")
        from multimodal_pipeline import multimodal_pipeline
        results['multimodal'] = visualize_pipeline(multimodal_pipeline, "multimodal_pipeline")
    except Exception as e:
        print(f"❌ Failed to visualize multimodal pipeline: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 VISUALIZATION SUMMARY")
    print("="*60)
    
    for pipeline_name, files in results.items():
        print(f"\n{pipeline_name.upper()} Pipeline:")
        for file_type, file_path in files.items():
            print(f"  • {file_type}: {file_path}")
    
    print("\n" + "="*60)
    print("✅ Visualization complete!")
    print("="*60)
    
    print("\n💡 To view:")
    print("  • PNG files: Open with any image viewer")
    print("  • Mermaid files: Copy to https://mermaid.live/")
    print("  • ASCII files: View in terminal with 'cat'")
    print()
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        pipeline_type = sys.argv[1]
        
        if pipeline_type == "text":
            from text_pipeline import text_pipeline
            visualize_pipeline(text_pipeline, "text_pipeline")
        
        elif pipeline_type == "image":
            from image_pipeline import image_pipeline
            visualize_pipeline(image_pipeline, "image_pipeline")
        
        elif pipeline_type == "multimodal":
            from multimodal_pipeline import multimodal_pipeline
            visualize_pipeline(multimodal_pipeline, "multimodal_pipeline")
        
        elif pipeline_type == "all":
            visualize_all()
        
        else:
            print(f"❌ Unknown pipeline: {pipeline_type}")
            print("Usage: python visualize_pipeline.py [text|image|multimodal|all]")
    
    else:
        # Default: visualize all
        visualize_all()
