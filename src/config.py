import os
from pathlib import Path
from dotenv import load_dotenv

# Get the project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

# Load .env file from project root explicitly
if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE, override=True)
else:
    # Fallback: try loading from current directory
    load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    OPENAI_API_KEY = OPENAI_API_KEY.strip()  # Remove any whitespace

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

if not OPENAI_API_KEY:
    raise ValueError(
        f"OPENAI_API_KEY is missing. Set it in .env file at {ENV_FILE}. "
        f"Current working directory: {os.getcwd()}. "
        f"ENV file exists: {ENV_FILE.exists()}"
    )

# Validate API key format (should start with sk-)
if not OPENAI_API_KEY.startswith("sk-"):
    raise ValueError(
        f"OPENAI_API_KEY format appears incorrect. It should start with 'sk-'. "
        f"Current key starts with: {OPENAI_API_KEY[:5] if len(OPENAI_API_KEY) >= 5 else 'too short'}"
    )