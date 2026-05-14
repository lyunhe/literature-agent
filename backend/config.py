import yaml
from .paths import DB_PATH, LIBRARY_DIR, LIBRARY_PDF_DIR
import os

# Load API keys and config from local env.yaml when present.
_env_path = os.path.join(os.path.dirname(__file__), "..", "env.yaml")
if os.path.exists(_env_path):
    with open(_env_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
else:
    cfg = {}

# --- API Keys ---
IEEE_API_KEY = cfg.get("api_keys", {}).get("ieee_xplore", "")
SS_API_KEY   = cfg.get("api_keys", {}).get("semantic_scholar", "")

# --- OpenAI / LLM Config ---
OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    cfg.get("api_keys", {}).get("openai_key", "")
)
OPENAI_BASE_URL = os.getenv(
    "OPENAI_API_BASE_URL",
    cfg.get("openai", {}).get("base_url", "https://api.openai.com/v1")
)
MODEL = cfg.get("openai", {}).get("model", "gpt-4o")

# --- LLM Generation Params ---
TEMPERATURE = cfg.get("llm", {}).get("temperature", 0.6)
MAX_TOKENS   = cfg.get("llm", {}).get("max_tokens", 4096)
TIME_OUT     = cfg.get("llm", {}).get("time_out", 600)
MAX_RETRIES  = cfg.get("llm", {}).get("max_retries", 5)

# --- Library Paths ---
LIBRARY_DIR = str(LIBRARY_DIR)
LIBRARY_PDF_DIR = str(LIBRARY_PDF_DIR)
DB_PATH = str(DB_PATH)
