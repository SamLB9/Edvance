#!/usr/bin/env python3
"""
Test logo integration in LaTeX generation.
"""

import os
from src.dynamic_latex_generator import ensure_logo_inclusion

def test_logo_integration():
    """Test logo integration function."""
    print("🧪 Testing Logo Integration...")
    
    # Sample LaTeX code (similar to what GPT generates)
    sample_latex = r"""
\documentclass[11pt,a4paper]{article}
\usepackage{graphicx}
\begin{document}
\begin{titlepage}
  \centering
  \vspace*{12mm}
  \brandlogo
  \vspace{8mm}
  {\huge\bfseries Course Notes Summary\par}
\end{titlepage}
\end{document}
"""
    
    logo_path = "16.png"
    
    print(f"Original LaTeX length: {len(sample_latex)}")
    print(f"Contains \\brandlogo: {'\\brandlogo' in sample_latex}")
    print(f"Contains \\includegraphics: {'\\includegraphics' in sample_latex}")
    
    # Test logo integration
    modified_latex = ensure_logo_inclusion(sample_latex, logo_path)
    
    print(f"\nModified LaTeX length: {len(modified_latex)}")
    print(f"Contains \\brandlogo: {'\\brandlogo' in modified_latex}")
    print(f"Contains \\includegraphics: {'\\includegraphics' in modified_latex}")
    print(f"Contains logo path: {logo_path in modified_latex}")
    
    if '\\includegraphics' in modified_latex and logo_path in modified_latex:
        print("✅ Logo integration successful!")
        return True
    else:
        print("❌ Logo integration failed!")
        return False

if __name__ == "__main__":
    success = test_logo_integration()
    if success:
        print("\n🎉 Logo Integration Test PASSED!")
    else:
        print("\n💥 Logo Integration Test FAILED!")

