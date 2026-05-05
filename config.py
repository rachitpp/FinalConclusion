# =============================================================
# CRAG Pipeline — Central Configuration
#
# All static, tuneable values live here.
# rag_query.py imports everything from this module.
# =============================================================

# ------------------------------------------------------------------
# Google Cloud / Vertex AI
# ------------------------------------------------------------------
GCP_LOCATION = "us-central1"

# ------------------------------------------------------------------
# ChromaDB
# ------------------------------------------------------------------
CHROMA_DIR = "chromaDb"

# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------
MAIN_MODEL      = "gemini-2.5-flash"
FAST_MODEL      = "gemini-2.5-flash"   # swap to "gemini-2.0-flash-lite" when available in your region
EMBEDDING_MODEL = "text-embedding-004"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ------------------------------------------------------------------
# LLM generation parameters
# ------------------------------------------------------------------
LLM_TEMPERATURE         = 0.2   # main generation model
LLM_MAX_OUTPUT_TOKENS   = 2048  # main + structured models
LLM_FAST_MAX_TOKENS     = 512   # fast model (binary tasks only)

# ------------------------------------------------------------------
# Retriever
# ------------------------------------------------------------------
VECTOR_K          = 5     # docs returned by vector retriever
VECTOR_FETCH_K    = 20    # candidate pool for MMR
VECTOR_LAMBDA     = 0.5   # MMR diversity factor (0 = max diversity, 1 = max relevance)

BM25_K            = 5     # docs returned by BM25 retriever

ENSEMBLE_WEIGHTS  = [0.4, 0.6]   # [BM25, vector] — must sum to 1.0

CROSS_ENCODER_TOP_N = 3   # docs kept after cross-encoder reranking

# ------------------------------------------------------------------
# Web search
# ------------------------------------------------------------------
WEB_SEARCH_MAX_RESULTS = 3

# ------------------------------------------------------------------
# Faithfulness thresholds
# ------------------------------------------------------------------
FAITHFULNESS_PASS_THRESHOLD    = 0.85  # ≥85% claims grounded → return as-is
FAITHFULNESS_RETRY_THRESHOLD   = 0.50  # 50–84% → self-correct once → re-check
FAITHFULNESS_RECHECK_THRESHOLD = 0.70  # corrected answer must reach ≥70%

# ------------------------------------------------------------------
# Claim verification
# ------------------------------------------------------------------
MIN_CLAIM_WORDS   = 4   # claims shorter than this are dropped as non-verifiable
MAX_CLAIM_WORKERS = 6   # parallel workers for claim checking (API rate-limit safe)
MAX_CHUNK_WORKERS = 6   # parallel workers for chunk grading

# ------------------------------------------------------------------
# Fallback message
# ------------------------------------------------------------------
HALLUCINATION_FALLBACK = (
    "I was unable to produce a fully verified answer from the available documents. "
    "Please rephrase your question or consult the source material directly."
)
