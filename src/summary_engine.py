import json
import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from .config import OPENAI_MODEL
from .dynamic_latex_generator import generate_latex_code


def generate_enhanced_summary_with_pdf(context: str, focus: str, pdf_name: str, summary_type: str = "Comprehensive") -> Dict[str, Any]:
    """
    Generate a summary and create a professional PDF version.
    
    Args:
        context: Retrieved context from the vector store
        focus: The specific focus area or topic for the summary
        pdf_name: Name of the PDF being summarized
        summary_type: Type of summary to generate
        
    Returns:
        Dictionary containing the summary, PDF path, and metadata
    """
    # Generate the summary first
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.3)
    
    prompt = f"""
    You are an expert study coach helping a student create focused course notes.
    
    CONTEXT FROM {pdf_name}:
    {context}
    
    FOCUS AREA: {focus}
    
    Please create a comprehensive, well-structured summary that:
    1. Focuses specifically on the requested topic: {focus}
    2. Organize logically with clear hierarchical sections
    3. Includes key definitions, concepts, and examples
    4. Highlights important relationships and connections
    5. Use clarity tools:
        - Bullet points (“- …”) for unordered lists
        - Numbered lists (“1 - …”) for ordered steps
        - Short, precise sentences
    6. Tone: Academic yet accessible, like notes prepared for quick revision.

    Avoid using too many "-" at the beginning of the lines.
    
    ### Output format:
    Return a **structured textual summary** (not LaTeX) with:
    - Sections and subsections labeled as shown above
    - Lists formatted exactly as described
    - Equations and formulas written in plain text (e.g., P(A|B) = P(B|A)P(A)/P(B))

    Do not add a title before the first section,commentary, explanations of your reasoning, or extra formatting beyond what is requested.
    """
    
    try:
        response = llm.invoke(prompt)
        summary = response.content
        
        summary_result = {
            "summary": summary,
            "focus": focus,
            "pdf_name": pdf_name,
            "context_length": len(context),
            "summary_length": len(summary),
            "status": "success",
            "type": summary_type
        }
        
        # Add timestamp
        from datetime import datetime
        summary_result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate PDF
        pdf_path = generate_pdf_summary(summary_result)
        summary_result["pdf_path"] = pdf_path
        summary_result["pdf_filename"] = os.path.basename(pdf_path)
        
    except Exception as e:
        summary_result = {
            "summary": f"Error generating summary: {str(e)}",
            "focus": focus,
            "pdf_name": pdf_name,
            "context_length": len(context),
            "summary_length": 0,
            "status": "error",
            "error": str(e),
            "type": summary_type
        }
        summary_result["pdf_error"] = str(e)
        summary_result["pdf_path"] = None
        print(f"PDF generation failed: {e}")
    
    return summary_result


def generate_pdf_summary(summary_data: Dict[str, Any], output_dir: str = "generated_summaries") -> str:
    """
    Generate a professional PDF summary using LaTeX.
    
    Args:
        summary_data: Dictionary containing summary information
        output_dir: Directory to save the PDF
        
    Returns:
        Path to the generated PDF file
    """
    try:
        # Use absolute output directory to avoid path issues across reruns
        abs_output_dir = os.path.abspath(output_dir)
        os.makedirs(abs_output_dir, exist_ok=True)
        
        # Build target filename: summary_{pdf_name}_{focus}.pdf with de-dupe suffixes
        def _clean(s: str) -> str:
            return (
                str(s)
                .strip()
                .replace(" ", "_")
                .replace("/", "_")
                .replace("\\", "_")
            )

        pdf_name_clean = _clean(summary_data.get("pdf_name", "document")).replace(".pdf", "")
        focus_clean = _clean(summary_data.get("focus", "General_Summary"))
        base_name = f"summary_{pdf_name_clean}_{focus_clean}"

        candidate = base_name + ".pdf"
        candidate_path = os.path.join(abs_output_dir, candidate)
        if os.path.exists(candidate_path):
            idx = 1
            while True:
                with_suffix = f"{base_name}_{idx}.pdf"
                with_suffix_path = os.path.join(abs_output_dir, with_suffix)
                if not os.path.exists(with_suffix_path):
                    candidate = with_suffix
                    candidate_path = with_suffix_path
                    break
                idx += 1

        pdf_filename = candidate
        pdf_path = candidate_path
        
        # Generate LaTeX code dynamically using GPT
        try:
            # Prepare logo: copy into output directory so LaTeX can find it during compilation
            logo_path = None
            if os.path.exists("16.png"):
                logo_dest = os.path.join(abs_output_dir, "16.png")
                shutil.copy2("16.png", logo_dest)
                logo_path = logo_dest
            
            raw_latex_content, latex_content = generate_latex_code(summary_data, logo_path)
            
        except Exception as e:
            print(f"Error generating LaTeX with GPT: {e}")
            raise Exception(f"Error generating LaTeX with GPT: {e}")

        # Create LaTeX file
        tex_filename = f"temp_{base_name}.tex"
        tex_file_path = os.path.join(abs_output_dir, tex_filename)
        
        try:
            # Write LaTeX content to file
            with open(tex_file_path, 'w', encoding='utf-8') as f:
                f.write(latex_content)
            
            # Compile LaTeX to PDF (single pass for speed)
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', tex_filename],
                capture_output=True,
                text=True,
                cwd=abs_output_dir
            )
            
            # Check if PDF was generated
            pdf_generated = False
            if os.path.exists(pdf_path):
                pdf_generated = True
            else:
                # Check for temp PDF file
                temp_pdf_path = os.path.join(abs_output_dir, f"temp_{base_name}.pdf")
                if os.path.exists(temp_pdf_path):
                    shutil.move(temp_pdf_path, pdf_path)
                    pdf_generated = True
            
            if not pdf_generated:
                raise Exception("PDF was not generated. Check LaTeX compilation logs for errors.")

        except Exception as e:
            raise Exception(f"Error during PDF generation: {e}")
        
        # Clean up temporary files only after successful PDF generation
        try:
            # Clean up LaTeX source file
            if os.path.exists(tex_file_path):
                os.unlink(tex_file_path)
            # Clean up LaTeX auxiliary files
            for ext in ['.aux', '.log', '.out', '.toc']:
                aux_file = tex_file_path.replace('.tex', ext)
                if os.path.exists(aux_file):
                    os.unlink(aux_file)
        except Exception:
            pass  # Ignore cleanup errors
        
    except Exception as e:
        raise Exception(f"Error generating PDF: {str(e)}")

    return pdf_path


def test_latex_compilation(output_dir: str = "generated_summaries") -> Dict[str, Any]:
    """
    Test if LaTeX compilation is working correctly.
    
    Args:
        output_dir: Directory to test compilation in
        
    Returns:
        Dictionary with test results
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Simple test LaTeX document
        test_tex = r"""
\documentclass{article}
\begin{document}
\title{LaTeX Test}
\author{Study Coach}
\maketitle
\section{Test Section}
This is a test document to verify LaTeX compilation works.
\end{document}
"""
        
        test_tex_path = os.path.join(output_dir, "test.tex")
        test_pdf_path = os.path.join(output_dir, "test.pdf")
        
        # Write test file
        with open(test_tex_path, 'w', encoding='utf-8') as f:
            f.write(test_tex)
        
        # Try to compile
        result = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', 'test.tex'],
            capture_output=True,
            text=True,
            cwd=output_dir
        )
        
        success = os.path.exists(test_pdf_path)
        
        # Clean up
        try:
            if os.path.exists(test_tex_path):
                os.unlink(test_tex_path)
            if os.path.exists(test_pdf_path):
                os.unlink(test_pdf_path)
            # Clean auxiliary files
            for ext in ['.aux', '.log', '.out']:
                aux_file = os.path.join(output_dir, f"test{ext}")
                if os.path.exists(aux_file):
                    os.unlink(aux_file)
        except Exception:
            pass
        
        return {
            "success": success,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "return_code": -1,
            "stdout": "",
            "stderr": ""
        }