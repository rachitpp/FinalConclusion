# =============================================================
# Corrective RAG (CRAG) Pipeline — Fully Revised
# Flow: Retrieve → Grade → Route → Generate → Verify Faithfulness
#
# FIXES APPLIED:
#   [F01] Web search docs are now graded before use
#   [F02] Empty final_docs handled before generation
#   [F03] Claim verdict order is now deterministic (sorted by original index)
#   [F04] Self-correction uses llm_structured (non-streaming) — no silent buffering
#   [F05] Model name constants extracted — no more hardcoded strings ×3
#   [F06] Module-level DB load wrapped in lazy init — safe to import without live DB
#   [F07] All pipeline output uses logger.info — no bare print() calls
#   [F08] Claim extractor output sanitised — trivially short claims dropped
#   [F09] Web search failure is explicit — empty result raises warning, pipeline aborts gracefully
#   [F10] Rewritten query validated before use — empty string guarded
#   [F11] Structured return value (CRAGResult dataclass) instead of bare string
#   [F12] BM25 retriever built lazily and cached — not rebuilt on every import
#   [F13] Thread pools created once at module level — not per-query
#   [F14] raw_context replaced with labelled context for faithfulness checking
#   [F15] Doc deduplication applied when combining local + web results
#   [F16] Thread pools registered with atexit for graceful shutdown on process exit
#   [F17] stream_answer label moved from logger.info to print() — consistent with content channel
# =============================================================

from __future__ import annotations

import atexit
import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

from dotenv import load_dotenv

load_dotenv() 

import config

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langsmith import traceable
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================
# Config
# =============================================================

GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
if not GCP_PROJECT:
    raise EnvironmentError(
        "GOOGLE_CLOUD_PROJECT is not set. Add it to your .env file."
    )

GCP_LOCATION = config.GCP_LOCATION
CHROMA_DIR   = config.CHROMA_DIR

MAIN_MODEL = config.MAIN_MODEL
FAST_MODEL = config.FAST_MODEL

FAITHFULNESS_PASS_THRESHOLD    = config.FAITHFULNESS_PASS_THRESHOLD
FAITHFULNESS_RETRY_THRESHOLD   = config.FAITHFULNESS_RETRY_THRESHOLD
FAITHFULNESS_RECHECK_THRESHOLD = config.FAITHFULNESS_RECHECK_THRESHOLD

HALLUCINATION_FALLBACK = config.HALLUCINATION_FALLBACK

MAX_CLAIM_WORKERS = config.MAX_CLAIM_WORKERS
MAX_CHUNK_WORKERS = config.MAX_CHUNK_WORKERS

MIN_CLAIM_WORDS = config.MIN_CLAIM_WORDS


# =============================================================
# Structured Return Value  [F11]
# =============================================================

@dataclass
class CRAGResult:
    """Structured result returned by run_crag_query."""
    answer:              str
    faithfulness_score:  float
    used_web_search:     bool
    source_docs:         list[Document] = field(default_factory=list)
    self_corrected:      bool = False
    fallback_used:       bool = False


# =============================================================
# Models
# =============================================================

# Main model — used for answer generation
llm = ChatGoogleGenerativeAI(
    model=MAIN_MODEL,
    project=GCP_PROJECT, location=GCP_LOCATION,
    vertexai=True, temperature=config.LLM_TEMPERATURE, max_output_tokens=config.LLM_MAX_OUTPUT_TOKENS, streaming=True,
)

# Non-streaming — for structured output calls (claim extraction, self-correction)
# [F04] Self-correction now uses this so output is not silently buffered
llm_structured = ChatGoogleGenerativeAI(
    model=MAIN_MODEL,
    project=GCP_PROJECT, location=GCP_LOCATION,
    vertexai=True, temperature=0, max_output_tokens=config.LLM_MAX_OUTPUT_TOKENS, streaming=False,
)

# Fast model — high-frequency binary tasks: chunk grading, query rewriting, claim checking
llm_fast = ChatGoogleGenerativeAI(
    model=FAST_MODEL,
    project=GCP_PROJECT, location=GCP_LOCATION,
    vertexai=True, temperature=0, max_output_tokens=config.LLM_FAST_MAX_TOKENS,
)


# =============================================================
# Retriever — Lazy Init  [F06] [F12]
# =============================================================

embedding_model = GoogleGenerativeAIEmbeddings(
    model=config.EMBEDDING_MODEL,
    project=GCP_PROJECT, location=GCP_LOCATION, vertexai=True,
)

# These are populated on first call to _get_retriever(), not at import time.
_retriever_cache: Optional[ContextualCompressionRetriever] = None
_vector_store_cache: Optional[Chroma] = None


def _load_bm25_docs(store: Chroma) -> list[Document]:
    """Load all documents from ChromaDB as Document objects for BM25."""
    data = store.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(data["documents"], data["metadatas"])
    ]


def _get_retriever() -> ContextualCompressionRetriever:
    """
    Build and cache the hybrid retriever.  [F06] [F12]
    Called lazily on first query so importing this module doesn't require a live DB.
    """
    global _retriever_cache, _vector_store_cache
    if _retriever_cache is not None:
        return _retriever_cache

    logger.info("[CRAG] Initialising retriever...")
    vector_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_model)
    _vector_store_cache = vector_store

    vector_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": config.VECTOR_K, "fetch_k": config.VECTOR_FETCH_K, "lambda_mult": config.VECTOR_LAMBDA},
    )

    bm25_docs = _load_bm25_docs(vector_store)
    if not bm25_docs:
        raise RuntimeError(
            f"ChromaDB at '{CHROMA_DIR}' is empty. Run db_creation.py first."
        )

    bm25_retriever   = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = config.BM25_K

    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], weights=config.ENSEMBLE_WEIGHTS
    )

    cross_encoder = HuggingFaceCrossEncoder(
        model_name=config.CROSS_ENCODER_MODEL
    )
    _retriever_cache = ContextualCompressionRetriever(
        base_compressor=CrossEncoderReranker(model=cross_encoder, top_n=config.CROSS_ENCODER_TOP_N),
        base_retriever=ensemble,
    )
    logger.info("[CRAG] Retriever ready.")
    return _retriever_cache


# =============================================================
# Persistent Thread Pools  [F13] [F16]
# =============================================================

_chunk_pool = ThreadPoolExecutor(max_workers=MAX_CHUNK_WORKERS, thread_name_prefix="chunk")
_claim_pool = ThreadPoolExecutor(max_workers=MAX_CLAIM_WORKERS, thread_name_prefix="claim")

# [F16] Register graceful shutdown so threads don't linger after the process exits.
# wait=False means the process won't block waiting for in-flight LLM API calls to finish.
atexit.register(_chunk_pool.shutdown, wait=False)
atexit.register(_claim_pool.shutdown, wait=False)



# =============================================================
# Structured Output Schemas
# =============================================================

class GradeResult(BaseModel):
    score:      str   = Field(description="'yes' if the chunk helps answer the question, 'no' otherwise.")
    confidence: float = Field(description="Confidence in the decision, 0.0–1.0.")
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
    reason: str = Field(description="One sentence citing context evidence for or against.")


# =============================================================
# Prompts & Chains
# =============================================================

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a precise and helpful AI assistant.
- Answer ONLY from the provided context.
- Be concise and structured.
- If the answer is not in the context, say: "I could not find the answer in the provided documents."
- Do not hallucinate. Cite relevant details where useful."""),
    ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
])

grader_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a relevance grader. Say 'yes' if the chunk helps answer the question, 'no' if off-topic. Be strict."),
        ("human", "Document chunk:\n{document}\n\nQuestion: {question}\n\nIs this chunk relevant?"),
    ])
    | llm_fast.with_structured_output(GradeResult)
)

rewrite_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a search query optimizer. Rewrite the user's question into a concise, keyword-rich search query. Output ONLY the rewritten query, nothing else."),
        ("human", "Original question: {question}\n\nRewritten search query:"),
    ])
    | llm_fast
    | StrOutputParser()
)

claim_extractor_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a claim extractor. Extract every individual factual claim from an answer. Each must be single, atomic, verifiable. Exclude hedges and meta-phrases."),
        ("human", "Answer:\n{answer}\n\nExtract all individual factual claims:"),
    ])
    | llm_structured.with_structured_output(ExtractedClaims)
)

claim_checker_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a fact verifier. True = context clearly states the claim. False = absent, contradicted, or only implied."),
        ("human", "CONTEXT:\n{context}\n\nCLAIM:\n{claim}\n\nIs this claim supported?"),
    ])
    | llm_fast.with_structured_output(ClaimVerdict)
)

# [F04] Self-correction uses llm_structured (non-streaming) — result is explicit, not silently buffered
self_correction_chain = (
    ChatPromptTemplate.from_messages([
        ("system", """You are a precise AI assistant performing a factual correction.
Your previous answer contained claims NOT supported by the provided context.
Rewrite the answer so every statement is directly supported by the context.
For each unsupported claim listed below, either replace it with what the context
actually says, or omit it entirely. Do not invent new information."""),
        ("human", (
            "CONTEXT:\n{context}\n\n"
            "QUESTION:\n{question}\n\n"
            "PREVIOUS ANSWER:\n{previous_answer}\n\n"
            "UNSUPPORTED CLAIMS TO CORRECT OR REMOVE:\n{unsupported_claims}\n\n"
            "Corrected answer (use only what the context directly supports):"
        )),
    ])
    | llm_structured
    | StrOutputParser()
)

web_search = TavilySearchResults(max_results=config.WEB_SEARCH_MAX_RESULTS)


# =============================================================
# Helpers
# =============================================================

def format_docs(docs: list[Document]) -> str:
    """Format docs with source/page labels — used both in the answer prompt
    and passed to verify_claims so the fact-checker can cite sources.  [F14]"""
    return "\n\n---\n\n".join(
        f"[Chunk {i} | {doc.metadata.get('source', 'unknown')}, p.{doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for i, doc in enumerate(docs, 1)
    )


def deduplicate_docs(docs: list[Document]) -> list[Document]:
    """Remove docs with duplicate page_content, preserving order.  [F15]"""
    seen: set[str] = set()
    unique: list[Document] = []
    for doc in docs:
        key = doc.page_content.strip()
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def stream_answer(prompt_input, label: str = "AI Assistant") -> str:
    """Stream LLM response to stdout and return the full string.

    Both the label and streamed content use print() intentionally — this is
    real-time UX output that must reach the terminal regardless of the logging
    level set in production. Using logger for the label but print() for content
    would cause the label to silently disappear at WARNING level while the answer
    still appeared, which is confusing. Keeping both on the same channel (stdout
    via print) ensures they are always seen together.  [F17]
    """
    output: list[str] = []
    print(f"\n{label}:")  # [F17] matches content output channel — both print(), not split logger/print
    for chunk in llm.stream(prompt_input):
        content = chunk.content if hasattr(chunk, "content") else str(chunk)
        print(content, end="", flush=True)
        output.append(content)
    print()
    return "".join(output)


# ------------------------------------------------------------------
# Chunk grading
# ------------------------------------------------------------------

def _grade_one_chunk(i: int, doc: Document, question: str) -> tuple[int, Document, GradeResult]:
    """Grade a single chunk. Safe to call from a thread pool."""
    try:
        result = grader_chain.invoke({"document": doc.page_content, "question": question})
        return i, doc, result
    except Exception as exc:
        logger.warning("Chunk grading failed for chunk %d: %s", i, exc)
        return i, doc, GradeResult(score="no", confidence=0.0, reason=f"Grading error: {exc}")


def grade_chunks(
    docs: list[Document], question: str, label: str = "local"
) -> tuple[list[Document], bool, bool]:
    """
    Grade each chunk for relevance in parallel using the persistent pool.
    Results are re-sorted by original index for deterministic log output.

    Returns (relevant_docs, any_relevant, any_irrelevant).
    """
    # [F02 guard] Empty input — nothing to grade
    if not docs:
        return [], False, False

    fn = partial(_grade_one_chunk, question=question)
    graded: list[tuple[int, Document, GradeResult]] = []

    futures: dict[Future, int] = {
        _chunk_pool.submit(fn, i, doc): i for i, doc in enumerate(docs)
    }
    for future in as_completed(futures):
        try:
            graded.append(future.result())
        except Exception as exc:
            i = futures[future]
            logger.warning("Future for chunk %d raised: %s", i, exc)
            graded.append((i, docs[i], GradeResult(score="no", confidence=0.0, reason=f"Future error: {exc}")))

    # [F03] Sort by original retrieval rank — deterministic output
    graded.sort(key=lambda x: x[0])

    relevant, any_relevant, any_irrelevant = [], False, False
    for i, doc, grade in graded:
        is_rel = grade.score.lower() == "yes" and grade.confidence >= 0.5
        icon   = "✓" if is_rel else "✗"
        logger.info(
            "  [%s] Chunk %d: %s %s  (conf=%.2f) — %s",
            label, i + 1, "RELEVANT" if is_rel else "NOT RELEVANT", icon,
            grade.confidence, grade.reason,
        )
        if is_rel:
            relevant.append(doc)
            any_relevant = True
        else:
            any_irrelevant = True

    return relevant, any_relevant, any_irrelevant


# ------------------------------------------------------------------
# Web search
# ------------------------------------------------------------------

def web_search_docs(question: str) -> list[Document]:
    """
    Rewrite query, search the web, return results as Documents.
    Returns an empty list (with a warning) on failure.  [F09] [F10]
    """
    rewritten = rewrite_chain.invoke({"question": question}).strip()

    # [F10] Guard against empty rewrite
    if not rewritten:
        logger.warning("[CRAG] Query rewriter returned empty string — using original question.")
        rewritten = question

    logger.info("[CRAG] Rewritten search query: '%s'", rewritten)

    try:
        results = web_search.invoke(rewritten)
    except Exception as exc:
        logger.warning("[CRAG] Web search failed: %s", exc)
        return []

    docs: list[Document] = []
    for r in results:
        content = r.get("content") or r.get("snippet") or ""
        url     = r.get("url", "unknown")
        if content:
            docs.append(Document(page_content=content, metadata={"source": url, "page": "web"}))
        else:
            logger.warning("Tavily result missing content, skipping: %s", r)

    # [F09] Explicit warning when search yields nothing
    if not docs:
        logger.warning("[CRAG] Web search returned no usable results for: '%s'", rewritten)

    return docs


# ------------------------------------------------------------------
# Claim verification
# ------------------------------------------------------------------

def _check_claim(claim: str, context: str) -> tuple[str, ClaimVerdict]:
    """Verify a single claim against context. Safe to call from a thread pool."""
    try:
        verdict = claim_checker_chain.invoke({"context": context, "claim": claim})
        return claim, verdict
    except Exception as exc:
        logger.warning("Claim check failed for '%s': %s", claim[:60], exc)
        return claim, ClaimVerdict(supported=False, reason=f"Verification error: {exc}")


def verify_claims(
    answer: str, context: str
) -> tuple[list[tuple[str, ClaimVerdict]], list[tuple[str, ClaimVerdict]], float]:
    """
    Extract claims from answer, verify each against context in parallel.
    Uses the labelled context (with source markers) so the checker can cite sources.  [F14]

    Returns (supported, unsupported, score).
    """
    extracted = claim_extractor_chain.invoke({"answer": answer})

    # [F08] Drop trivially short / non-verifiable claims
    claims = [
        c for c in extracted.claims
        if len(c.split()) >= MIN_CLAIM_WORDS
    ]

    if not claims:
        logger.info("[Faithfulness] No verifiable claims extracted — treating as faithful.")
        return [], [], 1.0

    fn = partial(_check_claim, context=context)
    raw_verdicts: list[tuple[str, ClaimVerdict, int]] = []   # (claim, verdict, original_index)

    futures: dict[Future, tuple[str, int]] = {
        _claim_pool.submit(fn, claim): (claim, idx)
        for idx, claim in enumerate(claims)
    }
    for future in as_completed(futures):
        claim, idx = futures[future]
        try:
            _, verdict = future.result()
            raw_verdicts.append((claim, verdict, idx))
        except Exception as exc:
            logger.warning("Future for claim '%s' raised: %s", claim[:60], exc)
            raw_verdicts.append((claim, ClaimVerdict(supported=False, reason=f"Future error: {exc}"), idx))

    # [F03] Restore original claim order for deterministic logs
    raw_verdicts.sort(key=lambda x: x[2])

    verdicts = [(c, v) for c, v, _ in raw_verdicts]
    supported   = [(c, v) for c, v in verdicts if v.supported]
    unsupported = [(c, v) for c, v in verdicts if not v.supported]
    score       = len(supported) / len(claims)

    for claim, verdict in supported:
        logger.info("  ✓ %s\n     → %s", claim[:90] + ("..." if len(claim) > 90 else ""), verdict.reason)
    for claim, verdict in unsupported:
        logger.info("  ✗ %s\n     → %s", claim[:90] + ("..." if len(claim) > 90 else ""), verdict.reason)

    return supported, unsupported, score


# =============================================================
# Main CRAG Pipeline
# =============================================================

@traceable(name="crag_query", tags=["crag", "production"])
def run_crag_query(question: str) -> CRAGResult:
    """
    CRAG pipeline:
      1. Retrieve  — hybrid BM25 + Vector MMR + cross-encoder reranker
      2. Grade     — parallel relevance grading per chunk
      3. Route     — all good / mixed / all bad → local / hybrid / web
      4. Generate  — stream answer from final context
      5. Verify    — per-claim faithfulness check → pass / self-correct / fallback

    Returns a CRAGResult with the answer, faithfulness score, and metadata.  [F11]
    """
    retriever    = _get_retriever()   # lazy init  [F06]
    used_web     = False
    self_corrected = False

    # ------------------------------------------------------------------
    # 1. Retrieve
    # ------------------------------------------------------------------
    logger.info("[CRAG] Retrieving chunks...")
    docs = retriever.invoke(question)

    # ------------------------------------------------------------------
    # 2. Grade local docs
    # ------------------------------------------------------------------
    logger.info("[CRAG] Grading %d local chunk(s)...", len(docs))
    relevant_docs, any_relevant, any_irrelevant = grade_chunks(docs, question, label="local")

    # ------------------------------------------------------------------
    # 3. Route
    # ------------------------------------------------------------------
    if any_relevant and not any_irrelevant:
        logger.info("[CRAG] All chunks relevant → local knowledge only.")
        final_docs = relevant_docs

    elif not any_relevant:
        logger.info("[CRAG] All chunks irrelevant → web search fallback.")
        used_web   = True
        web_docs   = web_search_docs(question)

        # [F01] Grade web results before using them
        if web_docs:
            logger.info("[CRAG] Grading %d web chunk(s)...", len(web_docs))
            web_relevant, _, _ = grade_chunks(web_docs, question, label="web")
            final_docs = web_relevant
        else:
            final_docs = []

    else:
        logger.info("[CRAG] Mixed (%d relevant) → local + web supplement.", len(relevant_docs))
        used_web = True
        web_docs = web_search_docs(question)

        # [F01] Grade web results before combining
        if web_docs:
            logger.info("[CRAG] Grading %d web chunk(s)...", len(web_docs))
            web_relevant, _, _ = grade_chunks(web_docs, question, label="web")
            combined = relevant_docs + web_relevant
        else:
            combined = relevant_docs

        # [F15] Deduplicate before sending to LLM
        final_docs = deduplicate_docs(combined)

    # ------------------------------------------------------------------
    # [F02] Guard: nothing to generate from
    # ------------------------------------------------------------------
    if not final_docs:
        logger.warning("[CRAG] No usable documents after grading — returning fallback.")
        return CRAGResult(
            answer=HALLUCINATION_FALLBACK,
            faithfulness_score=0.0,
            used_web_search=used_web,
            source_docs=[],
            fallback_used=True,
        )

    # ------------------------------------------------------------------
    # 4. Generate
    # ------------------------------------------------------------------
    logger.info("[CRAG] Generating answer from %d document(s)...", len(final_docs))

    # [F14] labelled_context is used BOTH for the answer prompt AND faithfulness checking
    labelled_context = format_docs(final_docs)
    final_prompt     = answer_prompt.invoke({"context": labelled_context, "question": question})
    answer           = stream_answer(final_prompt)

    # ------------------------------------------------------------------
    # 5. Faithfulness check
    # ------------------------------------------------------------------
    logger.info("[Faithfulness] Verifying claims...")

    # [F14] Pass labelled_context (with source markers) to verify_claims
    supported, unsupported, score = verify_claims(answer, labelled_context)
    total = len(supported) + len(unsupported)
    logger.info("[Faithfulness] Score: %d/%d (%.0f%%)", len(supported), total, score * 100)

    if score >= FAITHFULNESS_PASS_THRESHOLD:
        logger.info("[Faithfulness] ✓ PASSED — returning answer.")
        return CRAGResult(
            answer=answer,
            faithfulness_score=score,
            used_web_search=used_web,
            source_docs=final_docs,
        )

    if score >= FAITHFULNESS_RETRY_THRESHOLD:
        logger.info("[Faithfulness] ⚠ PARTIAL — self-correcting...")
        unsupported_list = "\n".join(f"- {c}" for c, _ in unsupported)

        # [F04] Self-correction via llm_structured — result is explicit, not silently buffered
        corrected = self_correction_chain.invoke({
            "context": labelled_context,
            "question": question,
            "previous_answer": answer,
            "unsupported_claims": unsupported_list,
        })
        logger.info("\nAI Assistant (self-corrected):\n%s", corrected)
        self_corrected = True

        logger.info("[Faithfulness] Re-verifying corrected answer...")
        _, _, re_score = verify_claims(corrected, labelled_context)
        logger.info("[Faithfulness] Re-check score: %.0f%%", re_score * 100)

        if re_score >= FAITHFULNESS_RECHECK_THRESHOLD:
            logger.info("[Faithfulness] ✓ Re-check PASSED.")
            return CRAGResult(
                answer=corrected,
                faithfulness_score=re_score,
                used_web_search=used_web,
                source_docs=final_docs,
                self_corrected=True,
            )

    logger.warning("[Faithfulness] ✗ FAILED — returning safe fallback.")
    logger.info("\nAI Assistant (safe fallback):\n%s", HALLUCINATION_FALLBACK)
    return CRAGResult(
        answer=HALLUCINATION_FALLBACK,
        faithfulness_score=score,
        used_web_search=used_web,
        source_docs=final_docs,
        self_corrected=self_corrected,
        fallback_used=True,
    )


# =============================================================
# Entry Point
# =============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    print("=" * 60)
    print("CRAG System Ready")
    print("Retrieval : Hybrid BM25 + Vector MMR + Cross-Encoder Reranker")
    print("Grading   : Parallel chunk relevance + web fallback (Tavily)")
    print("Verify    : Per-claim faithfulness + self-correction loop")
    print(f"LLM       : Vertex AI {MAIN_MODEL}")
    print("Tracing   : LangSmith  |  Type '0' to exit")
    print("=" * 60)

    while True:
        user_input = input("\nUser:\n").strip()
        if not user_input:
            continue
        if user_input == "0":
            print("Goodbye!")
            break

        result = run_crag_query(user_input)

        print("\n--- Result Metadata ---")
        print(f"  Faithfulness : {result.faithfulness_score:.0%}")
        print(f"  Web search   : {'yes' if result.used_web_search else 'no'}")
        print(f"  Self-corrected: {'yes' if result.self_corrected else 'no'}")
        print(f"  Fallback used : {'yes' if result.fallback_used else 'no'}")
        print(f"  Source docs  : {len(result.source_docs)}")