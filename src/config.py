"""Central settings, read once from the environment (.env locally, env vars on
EC2). Nothing secret is ever hardcoded or logged here."""
import os

from dotenv import load_dotenv

load_dotenv()

# --- LLM (see llm.py) --------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # anthropic | ollama | gemini
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5")  # needs ANTHROPIC_API_KEY
# "default" opts into server-side refusal fallbacks; "off" disables them.
CLAUDE_FALLBACKS = os.getenv("CLAUDE_FALLBACKS", "default")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # needs GOOGLE_API_KEY (free tier)
# Cache LLM responses on disk keyed by prompt hash -- the plan's mitigation
# for API rate limits while iterating on the same test patients.
LLM_CACHE = os.getenv("LLM_CACHE", "false").lower() == "true"

# --- Storage -----------------------------------------------------------------
# With S3_BUCKET set, patient data, long-term memory and logs live in S3
# (the EC2 instance's IAM role grants access). Without it, everything lives
# under LOCAL_DATA_DIR so the app runs on a laptop with no AWS account.
S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_PREFIX = os.getenv("S3_PREFIX", "patient-followup")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
LOCAL_DATA_DIR = os.getenv(
    "LOCAL_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data")
)

# --- RAG ---------------------------------------------------------------------
VECTOR_STORE = os.getenv("VECTOR_STORE", "chroma")  # chroma | faiss
VECTOR_DIR = os.getenv(
    "VECTOR_DIR", os.path.join(os.path.dirname(__file__), "..", "vectorstore")
)
# "default" = ChromaDB's bundled all-MiniLM-L6-v2 (ONNX, runs locally, no API
# key). "fake" = deterministic hash embeddings, used by the test suite.
EMBEDDINGS = os.getenv("EMBEDDINGS", "default")
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "8"))
