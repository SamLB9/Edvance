#!/usr/bin/env python3
"""
Test that timestamped filenames work and each generation creates a unique file.
"""

import os
import sys
import time
sys.path.append('.')

from src.summary_engine import generate_enhanced_summary_with_pdf

def test_timestamped_filenames():
    """Test that each generation creates a unique file even with the same focus."""
    print("🧪 Testing Timestamped Filenames...")
    
    # Test data
    context = """
    Bayes' Theorem is a fundamental concept in probability theory.
    It describes the probability of an event based on prior knowledge.
    The formula is: P(A|B) = P(B|A) * P(A) / P(B)
    """
    
    focus = "Bayes Theorem Test"
    pdf_name = "test_notes.pdf"
    summary_type = "Comprehensive"
    
    # First generation
    print("\n📝 First Generation...")
    try:
        result1 = generate_enhanced_summary_with_pdf(
            context=context,
            focus=focus,
            pdf_name=pdf_name,
            summary_type=summary_type
        )
        
        if result1["status"] == "success" and result1.get("pdf_path"):
            print(f"✅ First PDF generated: {result1['pdf_filename']}")
            first_pdf = result1['pdf_path']
            first_size = os.path.getsize(first_pdf) if os.path.exists(first_pdf) else 0
            print(f"📏 First PDF size: {first_size} bytes")
        else:
            print("❌ First generation failed")
            return False
            
    except Exception as e:
        print(f"❌ First generation error: {e}")
        return False
    
    # Wait a moment to ensure different timestamp
    print("⏳ Waiting 2 seconds for different timestamp...")
    time.sleep(2)
    
    # Second generation with same focus
    print("\n📝 Second Generation (same focus)...")
    try:
        result2 = generate_enhanced_summary_with_pdf(
            context=context,
            focus=focus,
            pdf_name=pdf_name,
            summary_type=summary_type
        )
        
        if result2["status"] == "success" and result2.get("pdf_path"):
            print(f"✅ Second PDF generated: {result2['pdf_filename']}")
            second_pdf = result2['pdf_path']
            second_size = os.path.getsize(second_pdf) if os.path.exists(second_pdf) else 0
            print(f"📏 Second PDF size: {second_size} bytes")
        else:
            print("❌ Second generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Second generation error: {e}")
        return False
    
    # Check if files are different
    if first_pdf != second_pdf:
        print(f"✅ Different files generated:")
        print(f"   First:  {os.path.basename(first_pdf)}")
        print(f"   Second: {os.path.basename(second_pdf)}")
        
        # Check if both files exist
        if os.path.exists(first_pdf) and os.path.exists(second_pdf):
            print("✅ Both files exist independently")
            return True
        else:
            print("❌ One or both files missing")
            return False
    else:
        print("❌ Same file path generated twice")
        return False

if __name__ == "__main__":
    print("🚀 Testing Timestamped Filenames...")
    success = test_timestamped_filenames()
    
    if success:
        print("\n🎉 Timestamped Filenames Test PASSED!")
        print("Each generation now creates a unique file!")
        print("No more showing old PDFs when using the same focus!")
    else:
        print("\n💥 Timestamped Filenames Test FAILED!")
        print("Check the error messages above.")
