import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.retrievers import BM25Retriever
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from pydantic import BaseModel, Field


# =============================================================
# Config (synced with config.py)
# =============================================================

GCP_PROJECT     = os.environ.get("GOOGLE_CLOUD_PROJECT")
GCP_LOCATION    = "us-central1"                               # config.GCP_LOCATION
CHROMA_DIR      = "chromaDb"                                  # config.CHROMA_DIR

MAIN_MODEL          = "gemini-2.5-flash"                      # config.MAIN_MODEL
FAST_MODEL          = "gemini-2.5-flash"                      # config.FAST_MODEL
EMBEDDING_MODEL     = "text-embedding-004"                    # config.EMBEDDING_MODEL
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2" # config.CROSS_ENCODER_MODEL

LLM_TEMPERATURE       = 0.2                                   # config.LLM_TEMPERATURE
LLM_MAX_OUTPUT_TOKENS = 2048                                  # config.LLM_MAX_OUTPUT_TOKENS
LLM_FAST_MAX_TOKENS   = 512                                   # config.LLM_FAST_MAX_TOKENS

VECTOR_K            = 5                                       # config.VECTOR_K
VECTOR_FETCH_K      = 20                                      # config.VECTOR_FETCH_K
VECTOR_LAMBDA       = 0.5                                     # config.VECTOR_LAMBDA
BM25_K              = 5                                       # config.BM25_K
ENSEMBLE_WEIGHTS    = [0.4, 0.6]                              # config.ENSEMBLE_WEIGHTS
CROSS_ENCODER_TOP_N = 3                                       # config.CROSS_ENCODER_TOP_N

WEB_SEARCH_MAX_RESULTS = 3                                    # config.WEB_SEARCH_MAX_RESULTS

FAITHFULNESS_PASS_THRESHOLD    = 0.85                         # config.FAITHFULNESS_PASS_THRESHOLD
FAITHFULNESS_RETRY_THRESHOLD   = 0.50                         # config.FAITHFULNESS_RETRY_THRESHOLD
FAITHFULNESS_RECHECK_THRESHOLD = 0.70                         # config.FAITHFULNESS_RECHECK_THRESHOLD

MIN_CLAIM_WORDS = 4                                           # config.MIN_CLAIM_WORDS

HALLUCINATION_FALLBACK = (                                    # config.HALLUCINATION_FALLBACK
    "I was unable to produce a fully verified answer from the available documents. "
    "Please rephrase your question or consult the source material directly."
)


# =============================================================
# Models
# =============================================================

llm = ChatGoogleGenerativeAI(
    model=MAIN_MODEL,
    project=GCP_PROJECT,
    location=GCP_LOCATION,
    vertexai=True,
    temperature=LLM_TEMPERATURE,
    max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
    streaming=True,
)

llm_fast = ChatGoogleGenerativeAI(
    model=FAST_MODEL,
    project=GCP_PROJECT,
    location=GCP_LOCATION,
    vertexai=True,
    temperature=0,
    max_output_tokens=LLM_FAST_MAX_TOKENS,
)


# =============================================================
# Retriever
# synced with create_db.py:
#   - same CHROMA_DIR ("chromaDb")
#   - same EMBEDDING_MODEL ("text-embedding-004")
#   - same GCP_PROJECT, GCP_LOCATION, vertexai=True
# =============================================================

embedding_model = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    project=GCP_PROJECT,
    location=GCP_LOCATION,
    vertexai=True,
)

# Load the ChromaDB that create_db.py built
vector_store = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_model,
)

# Pull raw documents out of ChromaDB so BM25 can index them
all_docs_data = vector_store.get(include=["documents", "metadatas"])
all_docs = [
    Document(page_content=text, metadata=meta or {})
    for text, meta in zip(all_docs_data["documents"], all_docs_data["metadatas"])
]

# Layer 1 — keyword search
bm25 = BM25Retriever.from_documents(all_docs)
bm25.k = BM25_K

# Layer 2 — semantic search
vector_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": VECTOR_K, "fetch_k": VECTOR_FETCH_K, "lambda_mult": VECTOR_LAMBDA},
)

# Layer 3 — combine both
ensemble = EnsembleRetriever(
    retrievers=[bm25, vector_retriever],
    weights=ENSEMBLE_WEIGHTS,
)

# Layer 4 — rerank the combined results
cross_encoder = HuggingFaceCrossEncoder(model_name=CROSS_ENCODER_MODEL)
retriever = ContextualCompressionRetriever(
    base_compressor=CrossEncoderReranker(model=cross_encoder, top_n=CROSS_ENCODER_TOP_N),
    base_retriever=ensemble,
)

web_search = TavilySearchResults(max_results=WEB_SEARCH_MAX_RESULTS)


# =============================================================
# Structured Output Schemas
# =============================================================

class GradeResult(BaseModel):
    score: str = Field(description="'yes' if relevant, 'no' if not")

class ExtractedClaims(BaseModel):
    claims: list[str] = Field(description="Individual factual claims from the answer")

class ClaimVerdict(BaseModel):
    supported: bool = Field(description="True if context supports this claim")


# =============================================================
# Chains
# =============================================================

grader_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Is this document chunk relevant to answering the question? Reply 'yes' or 'no'."),
        ("human", "Chunk:\n{document}\n\nQuestion: {question}"),
    ])
    | llm_fast.with_structured_output(GradeResult)
)

rewrite_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Rewrite the question as a short web search query. Output only the query, nothing else."),
        ("human", "{question}"),
    ])
    | llm_fast | StrOutputParser()
)

claim_extractor_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Extract every individual factual claim from this answer as a list. Each must be a single, atomic, verifiable statement."),
        ("human", "Answer:\n{answer}"),
    ])
    | llm_fast.with_structured_output(ExtractedClaims)
)

claim_checker_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Does the context explicitly support this claim? True if yes, False if absent or contradicted."),
        ("human", "Context:\n{context}\n\nClaim: {claim}"),
    ])
    | llm_fast.with_structured_output(ClaimVerdict)
)

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. Be concise. If the answer is not in the context, say so."),
    ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
])

self_correction_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Rewrite the answer so every claim is directly supported by the context. Fix or remove anything unsupported."),
        ("human", (
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Previous answer:\n{previous_answer}\n\n"
            "Unsupported claims to fix or remove:\n{unsupported_claims}"
        )),
    ])
    | llm_fast | StrOutputParser()
)


# =============================================================
# Helper Functions
# =============================================================

def format_docs(docs: list) -> str:
    """Join all chunks into one string to pass as context."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def grade_chunks(docs: list, question: str) -> list:
    """Ask the LLM if each chunk is relevant. Return only the relevant ones."""
    relevant = []
    for doc in docs:
        result = grader_chain.invoke({"document": doc.page_content, "question": question})
        if result.score.lower() == "yes":
            relevant.append(doc)
    return relevant


def web_search_docs(question: str) -> list:
    """Rewrite question as search query, search the web, return results as Documents."""
    rewritten = rewrite_chain.invoke({"question": question}).strip() or question
    print(f"Web search query: '{rewritten}'")
    try:
        results = web_search.invoke(rewritten)
    except Exception as e:
        print(f"Web search failed: {e}")
        return []
    return [
        Document(
            page_content=r.get("content", ""),
            metadata={"source": r.get("url", ""), "page": "web"}
        )
        for r in results if r.get("content")
    ]


def verify_claims(answer: str, context: str):
    """
    Extract claims from the answer and check each one against the context.
    Returns (score, unsupported_claims).
    Score = fraction of claims that are supported.
    """
    extracted = claim_extractor_chain.invoke({"answer": answer})

    # Drop very short claims — too vague to verify (synced with config.MIN_CLAIM_WORDS = 4)
    claims = [c for c in extracted.claims if len(c.split()) >= MIN_CLAIM_WORDS]

    if not claims:
        return 1.0, []

    unsupported = []
    for claim in claims:
        verdict = claim_checker_chain.invoke({"context": context, "claim": claim})
        if not verdict.supported:
            unsupported.append(claim)

    score = (len(claims) - len(unsupported)) / len(claims)
    return score, unsupported


# =============================================================
# Main CRAG Pipeline
# =============================================================

def run_crag_query(question: str) -> str:

    # ----------------------------------------------------------
    # 1. Retrieve — pull candidate chunks from ChromaDB
    # ----------------------------------------------------------
    docs = retriever.invoke(question)

    # ----------------------------------------------------------
    # 2. Grade — keep only chunks relevant to the question
    # ----------------------------------------------------------
    relevant_docs = grade_chunks(docs, question)

    # ----------------------------------------------------------
    # 3. Route — decide where to get the final context from
    # ----------------------------------------------------------
    if relevant_docs and len(relevant_docs) == len(docs):
        # Every chunk passed grading — local knowledge is sufficient
        print("All chunks relevant — using local docs only.")
        final_docs = relevant_docs

    elif not relevant_docs:
        # Nothing passed — local DB has no useful info, go to web
        print("No relevant chunks — falling back to web search.")
        web_docs   = web_search_docs(question)
        final_docs = grade_chunks(web_docs, question)

    else:
        # Some passed, some didn't — supplement local with web
        print(f"{len(relevant_docs)} relevant chunk(s) — supplementing with web search.")
        web_docs   = web_search_docs(question)
        final_docs = relevant_docs + grade_chunks(web_docs, question)

    # Guard: nothing usable after all routing attempts
    if not final_docs:
        return HALLUCINATION_FALLBACK

    # ----------------------------------------------------------
    # 4. Generate — stream answer from the final context
    # ----------------------------------------------------------
    context = format_docs(final_docs)
    answer  = ""
    print("\nAnswer:")
    for chunk in llm.stream(answer_prompt.invoke({"context": context, "question": question})):
        print(chunk.content, end="", flush=True)
        answer += chunk.content
    print()

    # ----------------------------------------------------------
    # 5. Verify — check every claim against the context
    # ----------------------------------------------------------
    score, unsupported = verify_claims(answer, context)
    print(f"\nFaithfulness: {score:.0%}")

    # Case 1: faithful enough — return as-is
    if score >= FAITHFULNESS_PASS_THRESHOLD:        # 0.85
        return answer

    # Case 2: partially faithful — try to self-correct
    if score >= FAITHFULNESS_RETRY_THRESHOLD:       # 0.50
        print("Partially faithful — attempting self-correction...")
        unsupported_list = "\n".join(f"- {c}" for c in unsupported)
        corrected = self_correction_chain.invoke({
            "context": context,
            "question": question,
            "previous_answer": answer,
            "unsupported_claims": unsupported_list,
        })
        re_score, _ = verify_claims(corrected, context)
        print(f"Re-check score: {re_score:.0%}")
        if re_score >= FAITHFULNESS_RECHECK_THRESHOLD:  # 0.70
            return corrected

    # Case 3: too many hallucinations — safe fallback
    return HALLUCINATION_FALLBACK


# =============================================================
# Entry Point
# =============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("CRAG Pipeline Ready")
    print("Type your question. Enter '0' to exit.")
    print("=" * 50)

    while True:
        question = input("\nYou: ").strip()
        if not question:
            continue
        if question == "0":
            print("Goodbye!")
            break
        print(run_crag_query(question))
