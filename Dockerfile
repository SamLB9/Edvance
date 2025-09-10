 # Dockerfile for Streamlit app with LaTeX (pdflatex)
 FROM python:3.11-slim

 ENV DEBIAN_FRONTEND=noninteractive \
     PYTHONDONTWRITEBYTECODE=1 \
     PYTHONUNBUFFERED=1 \
     PIP_NO_CACHE_DIR=1
 
# System packages (LaTeX for PDF generation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    texlive-latex-base texlive-latex-recommended texlive-latex-extra \
    texlive-fonts-recommended texlive-fonts-extra \
    texlive-lang-english \
    && rm -rf /var/lib/apt/lists/*
 
 WORKDIR /app
 
 # Install Python dependencies first (better layer caching)
 COPY requirements.txt /app/requirements.txt
 RUN pip install -r /app/requirements.txt
 
 # Copy the rest of the app
 COPY . /app/
 
 # Run Streamlit on provided $PORT (Railway/Render)
 CMD streamlit run app.py --server.port ${PORT:-8000} --server.address 0.0.0.0
 
 
 