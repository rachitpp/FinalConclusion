# =============================================================
# Corrective RAG (CRAG) Pipeline
# Flow: Retrieve → Grade → Route → Generate → Verify Faithfulness
# =============================================================

from dotenv import load_dotenv
load_dotenv()

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.tools.tavily_search import TavilySearchResults
from langsmith import traceable
from pydantic import BaseModel, Field


# =============================================================
# Config
# =============================================================

GCP_PROJECT   = os.environ.get("GOOGLE_CLOUD_PROJECT")
GCP_LOCATION  = "us-central1"
CHROMA_DIR    = "chromaDb"

FAITHFULNESS_PASS_THRESHOLD  = 0.85   # ≥85% claims grounded → return as-is
FAITHFULNESS_RETRY_THRESHOLD = 0.50   # 50–84% → self-correct once → re-check
                                       # <50%   → hard fallback

HALLUCINATION_FALLBACK = (
    "I was unable to produce a fully verified answer from the available documents. "
    "Please rephrase your question or consult the source material directly."
)

MAX_CLAIM_WORKERS  = 6    # parallel workers for claim verification (API rate-limit safe)
MAX_CHUNK_WORKERS  = 6    # parallel workers for chunk grading


# =============================================================
# Models
# =============================================================

# Main model — used for answer generation (streaming) and claim extraction
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    project=GCP_PROJECT, location=GCP_LOCATION,
    vertexai=True, temperature=0.2, max_output_tokens=1024, streaming=True,
)

# Same model, non-streaming — required for structured output calls
llm_structured = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    project=GCP_PROJECT, location=GCP_LOCATION,
    vertexai=True, temperature=0, max_output_tokens=1024, streaming=False,
)

# Lightweight model — used for cheap binary tasks: chunk grading, query rewriting, claim checking
llm_fast = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    project=GCP_PROJECT, location=GCP_LOCATION,
    vertexai=True, temperature=0,
)


# =============================================================
# Retriever (Hybrid: BM25 + Vector MMR + Cross-Encoder Reranker)
# =============================================================

embedding_model = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",
    project=GCP_PROJECT, location=GCP_LOCATION, vertexai=True,
)

vector_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_model)

vector_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
)

def _load_bm25_docs(store: Chroma) -> list[Document]:
    """Load all documents from ChromaDB and wrap them as Document objects for BM25."""
    data = store.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(data["documents"], data["metadatas"])
    ]

bm25_retriever      = BM25Retriever.from_documents(_load_bm25_docs(vector_store))
bm25_retriever.k    = 5

ensemble_retriever  = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever], weights=[0.4, 0.6]
)

cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
retriever     = ContextualCompressionRetriever(
    base_compressor=CrossEncoderReranker(model=cross_encoder, top_n=3),
    base_retriever=ensemble_retriever,
)


# =============================================================
# Structured Output Schemas
# =============================================================

class GradeResult(BaseModel):
    score:      str   = Field(description="'yes' if the chunk helps answer the question, 'no' otherwise.")
    confidence: float = Field(description="Confidence in the decision, 0.0 (unsure) to 1.0 (certain).")
    reason:     str   = Field(description="One sentence explaining the decision.")

class ExtractedClaims(BaseModel):
    claims: list[str] = Field(description=(
        "Individual factual claims from the answer. Each must be a single, atomic, "
        "verifiable statement. Exclude hedges and meta-phrases."
    ))

class ClaimVerdict(BaseModel):
    supported: bool = Field(description=(
        "True if the context explicitly supports this claim. "
        "False if absent, contradicted, or only implied."
    ))
    reason: str = Field(description="One sentence citing context evidence for or against the claim.")


# =============================================================
# Prompts & Chains
# =============================================================

# --- Answer generation ---
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a precise and helpful AI assistant.
- Answer ONLY from the provided context.
- Be concise and structured.
- If the answer is not in the context, say: "I could not find the answer in the provided documents."
- Do not hallucinate. Cite relevant details where useful."""),
    ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
])

# --- Chunk relevance grader ---
grader_chain = (
    ChatPromptTemplate.from_messages([
        ("system", """You are a relevance grader. Decide if a document chunk helps answer a question.
Say 'yes' if useful, 'no' if off-topic. Be strict."""),
        ("human", "Document chunk:\n{document}\n\nQuestion: {question}\n\nIs this chunk relevant?"),
    ])
    | llm_fast.with_structured_output(GradeResult)
)

# --- Query rewriter (for web search fallback) ---
rewrite_chain = (
    ChatPromptTemplate.from_messages([
        ("system", """You are a search query optimizer. Rewrite the user's question into a concise,
keyword-rich search query. Remove conversational fillers. Output ONLY the rewritten query."""),
        ("human", "Original question: {question}\n\nRewritten search query:"),
    ])
    | llm_fast
    | StrOutputParser()
)

# --- Claim extractor ---
claim_extractor_chain = (
    ChatPromptTemplate.from_messages([
        ("system", """You are a claim extractor. Extract every individual factual claim from an answer.
Each claim must be a single, atomic, verifiable statement. Exclude hedges and meta-phrases."""),
        ("human", "Answer:\n{answer}\n\nExtract all individual factual claims:"),
    ])
    | llm_structured.with_structured_output(ExtractedClaims)
)

# --- Per-claim fact checker ---
claim_checker_chain = (
    ChatPromptTemplate.from_messages([
        ("system", """You are a fact verifier. Check if a claim is explicitly supported by the context.
True = context clearly states it. False = absent, contradicted, or only implied."""),
        ("human", "CONTEXT:\n{context}\n\nCLAIM:\n{claim}\n\nIs this claim supported?"),
    ])
    | llm_fast.with_structured_output(ClaimVerdict)
)

# --- Self-correction ---
self_correction_chain = (
    ChatPromptTemplate.from_messages([
        ("system", """You are a precise AI assistant. Your previous answer had ungrounded claims.
Produce a corrected answer using ONLY the provided context. Do not re-introduce the listed claims."""),
        ("human", (
            "CONTEXT:\n{context}\n\n"
            "QUESTION:\n{question}\n\n"
            "PREVIOUS ANSWER:\n{previous_answer}\n\n"
            "UNSUPPORTED CLAIMS TO REMOVE:\n{unsupported_claims}\n\n"
            "Corrected answer:"
        )),
    ])
    | llm
    | StrOutputParser()
)

# --- Web search ---
web_search = TavilySearchResults(max_results=3)


# =============================================================
# Helper Functions
# =============================================================

def format_docs(docs: list[Document]) -> str:
    """Format docs with source/page labels for the answer prompt."""
    return "\n\n---\n\n".join(
        f"[Chunk {i} | {doc.metadata.get('source', 'unknown')}, p.{doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for i, doc in enumerate(docs, 1)
    )


def stream_answer(prompt_input, label: str = "AI Assistant") -> str:
    """Stream LLM response to stdout and return the full answer string."""
    output = []
    print(f"\n{label}:")
    for chunk in llm.stream(prompt_input):
        content = chunk.content if hasattr(chunk, "content") else str(chunk)
        for ch in content:
            print(ch, end="", flush=True)
        output.append(content)
    print()
    return "".join(output)


def _grade_one_chunk(indexed_doc: tuple[int, Document], question: str) -> tuple[int, Document, GradeResult]:
    """Grade a single (index, doc) pair. Safe to call from a thread pool."""
    i, doc = indexed_doc
    result = grader_chain.invoke({"document": doc.page_content, "question": question})
    return i, doc, result


def grade_chunks(docs: list[Document], question: str) -> tuple[list[Document], bool, bool]:
    """
    Grade each retrieved chunk for relevance in parallel.

    Chunks are graded concurrently then re-sorted by original index so log
    output always appears in the same order as retrieval rank.

    Returns (relevant_docs, any_relevant, any_irrelevant).
    """
    with ThreadPoolExecutor(max_workers=min(len(docs), MAX_CHUNK_WORKERS)) as pool:
        # Submit all grading jobs and collect results ordered by chunk index
        graded = sorted(
            pool.map(lambda pair: _grade_one_chunk(pair, question), enumerate(docs)),
            key=lambda x: x[0],
        )

    relevant, any_relevant, any_irrelevant = [], False, False
    for i, doc, grade in graded:
        is_relevant = grade.score.lower() == "yes" and grade.confidence >= 0.5
        icon = "✓" if is_relevant else "✗"
        print(f"  Chunk {i+1}: {'RELEVANT' if is_relevant else 'NOT RELEVANT'} {icon}  "
              f"(conf={grade.confidence:.2f}) — {grade.reason}")
        if is_relevant:
            relevant.append(doc)
            any_relevant = True
        else:
            any_irrelevant = True

    return relevant, any_relevant, any_irrelevant


def web_search_docs(question: str) -> list[Document]:
    """Rewrite query, search the web, and return results as Documents."""
    rewritten = rewrite_chain.invoke({"question": question})
    print(f"[CRAG] Rewritten search query: '{rewritten}'")
    return [
        Document(page_content=r["content"], metadata={"source": r["url"], "page": "web"})
        for r in web_search.invoke(rewritten)
    ]


def verify_claims(answer: str, raw_context: str) -> tuple[list, list, float]:
    """
    Extract claims from the answer and verify each against raw context in parallel.
    Returns (supported, unsupported, score).
    """
    extracted = claim_extractor_chain.invoke({"answer": answer})
    claims    = extracted.claims

    if not claims:
        return [], [], 1.0  # meta-answer ("I don't know") — treat as faithful

    def _check(claim):
        verdict = claim_checker_chain.invoke({"context": raw_context, "claim": claim})
        return claim, verdict

    verdicts = []
    with ThreadPoolExecutor(max_workers=min(len(claims), MAX_CLAIM_WORKERS)) as pool:
        verdicts = list(pool.map(_check, claims))

    supported   = [(c, v) for c, v in verdicts if v.supported]
    unsupported = [(c, v) for c, v in verdicts if not v.supported]
    score       = len(supported) / len(claims)

    for claim, verdict in supported:
        print(f"  ✓ {claim[:90]}{'...' if len(claim) > 90 else ''}\n     → {verdict.reason}")
    for claim, verdict in unsupported:
        print(f"  ✗ {claim[:90]}{'...' if len(claim) > 90 else ''}\n     → {verdict.reason}")

    return supported, unsupported, score


# =============================================================
# Main CRAG Pipeline
# =============================================================

@traceable(name="crag_query")
def run_crag_query(question: str) -> str:
    """
    CRAG pipeline:
      1. Retrieve  — hybrid BM25 + Vector MMR + cross-encoder reranker
      2. Grade     — parallel relevance grading per chunk
      3. Route     — all good / mixed / all bad → local / hybrid / web
      4. Generate  — stream answer from final context
      5. Verify    — per-claim faithfulness check → pass / self-correct / fallback
    """

    # 1. Retrieve
    print("\n[CRAG] Retrieving chunks...")
    docs = retriever.invoke(question)

    # 2. Grade
    print(f"[CRAG] Grading {len(docs)} chunk(s)...")
    relevant_docs, any_relevant, any_irrelevant = grade_chunks(docs, question)

    # 3. Route
    if any_relevant and not any_irrelevant:
        print("[CRAG] All chunks relevant → local knowledge only.")
        final_docs = relevant_docs
    elif not any_relevant:
        print("[CRAG] All chunks irrelevant → web search fallback.")
        final_docs = web_search_docs(question)
    else:
        print(f"[CRAG] Mixed ({len(relevant_docs)} relevant) → local + web supplement.")
        final_docs = relevant_docs + web_search_docs(question)

    # 4. Generate
    print(f"[CRAG] Generating answer from {len(final_docs)} document(s)...")
    context      = format_docs(final_docs)
    raw_context  = "\n\n".join(doc.page_content for doc in final_docs)
    final_prompt = answer_prompt.invoke({"context": context, "question": question})
    answer       = stream_answer(final_prompt)

    # 5. Faithfulness check
    print("\n[Faithfulness] Verifying claims...")
    supported, unsupported, score = verify_claims(answer, raw_context)
    total = len(supported) + len(unsupported)
    print(f"[Faithfulness] Score: {len(supported)}/{total} ({score:.0%})")

    if score >= FAITHFULNESS_PASS_THRESHOLD:
        print(f"[Faithfulness] ✓ PASSED — returning answer.")
        return answer

    if score >= FAITHFULNESS_RETRY_THRESHOLD:
        print(f"[Faithfulness] ⚠ PARTIAL — self-correcting...")
        unsupported_list = "\n".join(f"- {c}" for c, _ in unsupported)
        corrected = stream_answer(
            self_correction_chain.invoke({
                "context": raw_context, "question": question,
                "previous_answer": answer, "unsupported_claims": unsupported_list,
            }),
            label="AI Assistant (self-corrected)"
        )

        # Re-check corrected answer (one retry max)
        print("\n[Faithfulness] Re-verifying corrected answer...")
        _, _, re_score = verify_claims(corrected, raw_context)
        if re_score >= FAITHFULNESS_RETRY_THRESHOLD:
            print(f"[Faithfulness] ✓ Re-check PASSED ({re_score:.0%})")
            return corrected

    print(f"[Faithfulness] ✗ FAILED — returning safe fallback.")
    print(f"\nAI Assistant (safe fallback):\n{HALLUCINATION_FALLBACK}")
    return HALLUCINATION_FALLBACK


# =============================================================
# Entry Point
# =============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CRAG System Ready")
    print("Retrieval : Hybrid BM25 + Vector MMR + Cross-Encoder Reranker")
    print("Grading   : Parallel chunk relevance + web fallback (Tavily)")
    print("Verify    : Per-claim faithfulness + self-correction loop")
    print("LLM       : Vertex AI Gemini 2.5 Flash")
    print("Tracing   : LangSmith  |  Type '0' to exit")
    print("=" * 60)

    while True:
        user_input = input("\nUser:\n").strip()
        if not user_input:
            continue
        if user_input == "0":
            print("Goodbye!")
            break
        run_crag_query(user_input)
