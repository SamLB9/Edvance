#!/usr/bin/env python3
"""
Performance monitoring script for Edvance Study Coach.
Run this to test the performance improvements.
"""

import time
import subprocess
import sys
from pathlib import Path

def measure_page_load_time():
    """Measure how long it takes to load the main page"""
    print("🚀 Testing page load performance...")
    
    start_time = time.time()
    
    # Import the main app to test initialization time
    try:
        from app import main, ensure_vectorstore_loaded, list_notes
        init_time = time.time() - start_time
        print(f"✅ App initialization: {init_time:.2f}s")
        
        # Test vector store loading
        start_time = time.time()
        ensure_vectorstore_loaded()
        vs_time = time.time() - start_time
        print(f"✅ Vector store loading: {vs_time:.2f}s")
        
        # Test notes listing
        start_time = time.time()
        notes = list_notes()
        notes_time = time.time() - start_time
        print(f"✅ Notes listing: {notes_time:.2f}s")
        
        total_time = init_time + vs_time + notes_time
        print(f"🎯 Total core operations: {total_time:.2f}s")
        
        if total_time < 2.0:
            print("🌟 Excellent performance!")
        elif total_time < 5.0:
            print("✅ Good performance")
        else:
            print("⚠️ Performance could be improved")
            
    except Exception as e:
        print(f"❌ Error during performance test: {e}")

def check_dependencies():
    """Check if all dependencies are properly installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'streamlit',
        'langchain',
        'langchain-openai',
        'langchain-chroma',
        'chromadb',
        'pypdf',
        'tiktoken'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
    else:
        print("\n✅ All dependencies are installed!")

def main():
    print("📊 Edvance Study Coach Performance Monitor")
    print("=" * 50)
    
    check_dependencies()
    print()
    measure_page_load_time()
    
    print("\n" + "=" * 50)
    print("💡 Performance Tips:")
    print("- Use the virtual environment: source .venv/bin/activate")
    print("- Clear browser cache if pages load slowly")
    print("- Close other applications to free up memory")
    print("- Use smaller PDF files for faster processing")

if __name__ == "__main__":
    main()
