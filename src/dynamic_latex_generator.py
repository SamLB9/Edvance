"""
Dynamic LaTeX generator using GPT to create customizable LaTeX code.
"""

import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from .config import OPENAI_MODEL

#CONVERSION RULES:
#1. Every standalone heading becomes \\section{{heading}} or \\subsection{{heading}}
#2. Every line starting with "-" becomes \\item inside \\begin{{itemize}} ... \\end{{itemize}}
#3. Every line starting with "1.", "2.", etc. becomes \\item inside \\begin{{enumerate}} ... \\end{{enumerate}}
#4. Mathematical expressions use $...$ or $$...$$


def generate_latex_code(summary_data: Dict[str, Any], logo_path: str = None) -> tuple[str, str]:
    """
    Generate LaTeX code using GPT for maximum customization.
    
    Args:
        summary_data: Dictionary containing summary information
        logo_path: Optional path to business logo image
        
    Returns:
        Tuple of (raw_latex_code, cleaned_latex_code)
    """
    
    # Prepare the prompt for GPT - CONTENT ONLY (no title page)
    prompt = f"""
    You are a LaTeX expert. Convert the following text content into LaTeX format for the BODY of a document.
    
    TEXT TO CONVERT:
    {summary_data.get('summary', 'No content available')}
    
    CONVERSION RULES:
    1. Every standalone heading becomes \\section{{heading}} or \\subsection{{heading}}
    2. Mathematical expressions use $...$ or $$...$$
    
    EXAMPLE:
    Input: "Purpose\\n- Item 1\\n- Item 2\\n1. Numbered item"
    Output: "\\section{{Purpose}}\\n\\begin{{itemize}}\\n\\item Item 1\\n\\item Item 2\\n\\end{{itemize}}\\n\\begin{{enumerate}}\\n\\item Numbered item\\n\\end{{enumerate}}"
    
    IMPORTANT: Generate ONLY the document body content. Do NOT include:
    - \\documentclass
    - \\usepackage commands
    - \\begin{{document}} or \\end{{document}}
    - Title page (\\begin{{titlepage}})
    - Preamble or setup commands
    
    Generate ONLY the LaTeX content that goes between \\begin{{document}} and \\end{{document}}, no explanations.
    """
    
    try:
        # Initialize GPT model
        llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.5)
        
        # Debug: Print what we're sending to the LLM
        print(f"🔍 DEBUG: Summary data keys: {list(summary_data.keys())}")
        print(f"🔍 DEBUG: Summary content length: {len(summary_data.get('summary', ''))}")
        print(f"🔍 DEBUG: Summary content preview: {summary_data.get('summary', '')[:200]}...")
        
        # Generate LaTeX code
        response = llm.invoke(prompt)
        raw_latex_content = response.content
        
        # Debug: Print what the LLM returned
        print(f"🔍 DEBUG: LLM response length: {len(raw_latex_content)}")
        print(f"🔍 DEBUG: LLM response preview: {raw_latex_content[:300]}...")
        
        # Clean up the response (remove any markdown formatting)
        latex_content = raw_latex_content
        if latex_content.startswith('```latex'):
            latex_content = latex_content.split('```latex')[1]
        if latex_content.startswith('```'):
            latex_content = latex_content.split('```')[1]
        if latex_content.endswith('```'):
            latex_content = latex_content[:-3]
        
        latex_content = latex_content.strip()
        
        # NO CLEANUP - Use raw LLM output to see what it generates
        print("📝 Using raw LLM output without any cleanup")
        # Just normalize color names to ensure basic compatibility
        latex_content = normalize_color_names(latex_content)

        # Now create the complete document with standardized title page
        complete_latex = create_complete_document(latex_content, summary_data, logo_path)
        
        return raw_latex_content, complete_latex
        
    except Exception as e:
        print(f"Error generating LaTeX with GPT: {e}")
        # Hard failure shouldn't happen often, but create a minimal master document from the plain summary
        body = summary_data.get('summary', 'No content available')
        complete_latex = create_complete_document(body, summary_data, logo_path)
        return complete_latex, complete_latex


def create_complete_document(body_content: str, summary_data: Dict[str, Any], logo_path: str = None) -> str:
    """
    Create a complete LaTeX document with standardized title page and content.
    
    Args:
        body_content: The LaTeX content for the document body
        summary_data: Dictionary containing summary information
        logo_path: Optional path to business logo image
        
    Returns:
        Complete LaTeX document as string
    """
    focus = summary_data.get('focus', 'Unknown')
    pdf_name = summary_data.get('pdf_name', 'Unknown')
    generated = summary_data.get('timestamp', 'Unknown')
    summary_type = summary_data.get('type', 'Comprehensive')
    # Prepare a stable PDF document title so viewers show a friendly name
    def _clean(s: str) -> str:
        return str(s).strip().replace(' ', '_').replace('/', '_').replace('\\', '_')
    pdf_name_clean = _clean(os.path.splitext(pdf_name)[0])
    focus_clean = _clean(focus)
    pdf_doc_title = f"summary_{pdf_name_clean}_{focus_clean}"

    # Standardized preamble (same every time)
    preamble = r"""
\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage[a4paper,margin=25mm]{geometry}
\usepackage{xcolor}
\definecolor{primary}{HTML}{0B6FA4}
\definecolor{secondary}{HTML}{3A6B8A}
\definecolor{accent}{HTML}{6E7B8C}
\definecolor{lightgray}{HTML}{F3F5F7}
\usepackage{graphicx}
\usepackage{float}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{mathtools}
\usepackage{enumitem}
\usepackage{booktabs}
\usepackage{array}
\usepackage{tcolorbox}
\tcbuselibrary{skins,breakable}
\usepackage{sectsty}
\allsectionsfont{\color{primary}\sffamily}
\usepackage{titlesec}
\titleformat{\section}{\Large\bfseries\sffamily\color{primary}}{\thesection}{1em}{}
\titleformat{\subsection}{\large\bfseries\sffamily\color{secondary}}{\thesubsection}{0.8em}{}
\titleformat{\subsubsection}{\normalsize\bfseries\sffamily\color{accent}}{\thesubsubsection}{0.6em}{}
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0.6pt}
\renewcommand{\footrulewidth}{0.4pt}
\renewcommand{\headrule}{\color{lightgray}\hrule height \headrulewidth \vspace{-\headrulewidth}}
\fancyhead[L]{\sffamily\small Course Notes Summary}
\fancyhead[C]{\sffamily\small Focus: """ + focus + r"""}
\fancyhead[R]{\sffamily\small Source: """ + pdf_name + r"""}
\fancyfoot[L]{\sffamily\small Generated: """ + generated + r"""}
\fancyfoot[C]{}
\fancyfoot[R]{\sffamily\small \thepage}
\usepackage{hyperref}
\hypersetup{pdftitle={""" + pdf_doc_title + r"""}, pdfauthor={Edvance}, colorlinks=true, linkcolor=primary, urlcolor=secondary, citecolor=primary}
\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{2}
\raggedbottom
\usepackage{titling}
\pretitle{\begin{center}\vspace*{1cm}\sffamily}
\posttitle{\par\end{center}\vskip 0.2em}
\preauthor{}\postauthor{}
\predate{}\postdate{}
% Helper macros used by model output
\newcommand{\Prob}{\mathrm{P}}
\newcommand{\given}{\,\mid\,}
"""

    # Standardized title page (same layout every time)
    titlepage = r"""
\begin{titlepage}
  \thispagestyle{empty}
  \begin{center}
    \vspace*{1cm}"""

    # Add logo if available
    if logo_path and os.path.exists(logo_path):
        logo_filename = os.path.basename(logo_path)
        titlepage += f"""
    \\begin{{center}}
    \\includegraphics[width=0.3\\textwidth]{{{logo_filename}}}
    \\end{{center}}
    \\vspace{{1em}}"""

    titlepage += f"""
    {{\\sffamily\\Huge\\bfseries\\color{{primary}} Course Notes Summary \\par}}
    \\vspace{{0.8em}}
    {{\\sffamily\\Large\\color{{secondary}} Focus Area: {focus} \\par}}
    \\vspace{{1.5em}}
    \\begin{{tcolorbox}}[enhanced,width=0.75\\textwidth,colback=lightgray,colframe=primary,boxrule=0.8pt,sharp corners,left=6pt,right=6pt,top=6pt,bottom=6pt]\\vspace{{0.5em}}
      \\sffamily
      \\begin{{tabular}}{{@{{}}p{{0.28\\textwidth}} p{{0.62\\textwidth}}@{{}}}}
        \\textbf{{Source: }} & {pdf_name} \\\\
        \\textbf{{Generated:}} & {generated} \\\\
        \\textbf{{Summary Type:}} & {summary_type} \\\\
      \\end{{tabular}}
    \\end{{tcolorbox}}

    \\vspace{{1.5em}}
    {{\\small\\sffamily Prepared for quick revision and reference\\par}}
    \\vfill

    {{\\small\\sffamily \\color{{accent}} Use this sheet as a step-by-step guide when solving problems.}}
    \\vspace{{1.8cm}}
  \\end{{center}}
\\end{{titlepage}}
"""

    # Combine everything
    complete_document = preamble + "\n\\begin{document}\n\n" + titlepage + "\n" + body_content + "\n\n\\end{document}\n"
    
    return complete_document


def normalize_color_names(body: str) -> str:
    """Map LLM-invented color names to predefined ones."""
    color_map = {
        "blueMain": "primary",
        "blueAccent": "secondary", 
        "blueLight": "accent",
        "grayLight": "lightgray"
    }
    for old_name, new_name in color_map.items():
        body = body.replace(f"\\color{{{old_name}}}", f"\\color{{{new_name}}}")
    return body


def ensure_logo_inclusion(latex_code: str, logo_path: str) -> str:
    """Ensure logo is included in the LaTeX document."""
    logo_filename = os.path.basename(logo_path)
    
    # Check if logo is already included
    if f"\\includegraphics" in latex_code and logo_filename in latex_code:
        return latex_code
    
    # Add logo to title page if not present
    if "\\begin{titlepage}" in latex_code and "\\includegraphics" not in latex_code:
        logo_inclusion = f"\\includegraphics[width=0.3\\textwidth]{{{logo_filename}}}"
        latex_code = latex_code.replace("\\begin{titlepage}", f"\\begin{{titlepage}}\n{logo_inclusion}")
    
    return latex_code


def assemble_master_document(body: str, summary_data: Dict[str, Any], logo_path: str | None) -> str:
    """Wrap a body of LaTeX into a stable, known-good master document with our styling and title page."""
    return create_complete_document(body, summary_data, logo_path)