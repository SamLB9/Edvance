"""
Performance optimization configuration for the Edvance Study Coach app.
"""

# Cache settings
PDF_PREVIEW_CACHE_LIMIT = 5  # Maximum number of PDF previews to cache
NOTES_LIST_CACHE_TTL = 300   # Notes list cache time-to-live in seconds

# Vector store settings
DEFAULT_RETRIEVAL_K = 6      # Default number of documents to retrieve
MAX_RETRIEVAL_K = 20         # Maximum number of documents to retrieve

# LaTeX compilation settings
LATEX_SINGLE_PASS = True     # Use single-pass LaTeX compilation for speed
LATEX_DEBUG_LOGS = False     # Disable debug logging for LaTeX compilation

# Session state cleanup
SESSION_CLEANUP_INTERVAL = 10  # Clean up session state every N page loads
MAX_SESSION_KEYS = 50         # Maximum number of keys to keep in session state

# File operation settings
FILE_READ_CHUNK_SIZE = 8192   # Chunk size for reading large files
MAX_FILE_PREVIEW_SIZE = 5000  # Maximum characters to preview in text files

# LLM settings
LLM_TEMPERATURE_FAST = 0.1    # Lower temperature for faster, more deterministic responses
LLM_MAX_TOKENS_FAST = 2000    # Limit tokens for faster responses

# UI optimization
STREAMLIT_CACHE_TTL = 3600    # Streamlit cache time-to-live in seconds
DISABLE_DEBUG_OUTPUT = True   # Disable debug print statements
