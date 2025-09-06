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
        
        # Generate filename with timestamp to ensure uniqueness
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        focus_clean = summary_data["focus"].replace(" ", "_").replace("/", "_").replace("\\", "_")
        pdf_filename = f"summary_{focus_clean}_{timestamp}.pdf"
        pdf_path = os.path.join(abs_output_dir, pdf_filename)
        
        # Also check for the temp filename that LaTeX actually creates
        temp_pdf_filename = f"temp_{focus_clean}_{timestamp}.pdf"
        temp_pdf_path = os.path.join(abs_output_dir, temp_pdf_filename)
        
        # Save raw summary text for debug purposes
        summary_txt_filename = f"summary_{timestamp}.txt"
        summary_txt_path = os.path.join(abs_output_dir, summary_txt_filename)
        try:
            with open(summary_txt_path, 'w', encoding='utf-8') as f:
                f.write(summary_data.get('summary', 'No summary available'))
            print(f"📝 Debug: Raw summary saved to {summary_txt_path}")
        except Exception as e:
            print(f"⚠️ Could not save summary text file: {e}")

        # Generate LaTeX code dynamically using GPT
        print("🔧 Generating LaTeX code with GPT...")
        try:
            # Prepare logo: copy into output directory so LaTeX can find it during compilation
            logo_path = None
            if os.path.exists("16.png"):
                logo_dest = os.path.join(abs_output_dir, "16.png")
                shutil.copy2("16.png", logo_dest)
                logo_path = logo_dest
                print(f"✅ Copied logo to output dir: {logo_dest}")
            
            raw_latex_content, latex_content = generate_latex_code(summary_data, logo_path)
            print(f"✅ LaTeX code generated successfully. Length: {len(latex_content)}")
            
            # Save LaTeX code before cleanup for debug purposes
            before_cleanup_tex_filename = f"before_cleanup_summary_{focus_clean}_{timestamp}.tex"
            before_cleanup_tex_path = os.path.join(abs_output_dir, before_cleanup_tex_filename)
            try:
                with open(before_cleanup_tex_path, 'w', encoding='utf-8') as f:
                    f.write(raw_latex_content)
                print(f"📝 Debug: LaTeX before cleanup saved to {before_cleanup_tex_path}")
            except Exception as e:
                print(f"⚠️ Could not save before-cleanup LaTeX file: {e}")
            
        except Exception as e:
            print(f"Error generating LaTeX with GPT: {e}")
            raise Exception(f"Error generating LaTeX with GPT: {e}")

        # Create LaTeX file
        tex_filename = f"temp_{focus_clean}_{timestamp}.tex"
        tex_file_path = os.path.join(abs_output_dir, tex_filename)
        
        try:
            # Write LaTeX content to file
            with open(tex_file_path, 'w', encoding='utf-8') as f:
                f.write(latex_content)
            
            # Debug: Save a copy of the LaTeX content for inspection
            debug_tex_path = os.path.join(abs_output_dir, f"debug_{focus_clean}_{timestamp}.tex")
            with open(debug_tex_path, 'w', encoding='utf-8') as f:
                f.write(latex_content)
            print(f"📝 Debug LaTeX saved to: {debug_tex_path}")
            
            # Compile LaTeX to PDF (two passes), write logs to files for debugging
            def run_pdflatex(pass_num: int) -> subprocess.CompletedProcess:
                print(f"▶️ Running pdflatex pass {pass_num}...")
                result_local = subprocess.run(
                    ['pdflatex', '-interaction=nonstopmode', tex_filename],
                    capture_output=True,
                    text=False,
                    cwd=abs_output_dir
                )
                log_base = os.path.join(abs_output_dir, tex_filename.replace('.tex', f'_pass{pass_num}'))
                try:
                    with open(log_base + '.stdout.txt', 'wb') as lf:
                        lf.write(result_local.stdout or b'')
                    with open(log_base + '.stderr.txt', 'wb') as lf:
                        lf.write(result_local.stderr or b'')
                    print(f"🪵 Saved pdflatex logs to {log_base}.[stdout|stderr].txt")
                except Exception as log_err:
                    print(f"⚠️ Could not save pdflatex logs: {log_err}")
                try:
                    stdout_preview = (result_local.stdout or b'').decode('utf-8', errors='replace')
                    stderr_preview = (result_local.stderr or b'').decode('utf-8', errors='replace')
                except Exception:
                    stdout_preview, stderr_preview = '', ''
                print(f"pdflatex pass {pass_num} rc={result_local.returncode}")
                if stdout_preview:
                    print(f"stdout preview: {stdout_preview[:200]}...")
                if stderr_preview:
                    print(f"stderr preview: {stderr_preview[:200]}...")
                return result_local

            # Run pdflatex twice for proper cross-references
            result1 = run_pdflatex(1)
            result2 = run_pdflatex(2)
            
            # Check if PDF was generated
            pdf_generated = False
            if os.path.exists(pdf_path):
                pdf_generated = True
                print(f"✅ PDF generated: {pdf_path}")
            elif os.path.exists(temp_pdf_path):
                # Sometimes LaTeX creates files with temp_ prefix
                shutil.move(temp_pdf_path, pdf_path)
                pdf_generated = True
                print(f"PDF generated with temp name: {temp_pdf_path}")
                print(f"PDF renamed to: {pdf_path}")
            else:
                # Check for any PDF files with similar names
                import glob
                pattern = os.path.join(abs_output_dir, f"*{focus_clean}*.pdf")
                pdf_files = glob.glob(pattern)
                if pdf_files:
                    # Use the most recent one
                    latest_pdf = max(pdf_files, key=os.path.getctime)
                    shutil.move(latest_pdf, pdf_path)
                    pdf_generated = True
                    print(f"Found and moved existing PDF: {latest_pdf} -> {pdf_path}")
            
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
            print("✅ Temporary files cleaned up")
        except Exception as cleanup_error:
            print(f"Warning: Could not clean up temporary files: {cleanup_error}")
        
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