"""
crag_pipeline.py
================
Corrective RAG (CRAG) pipeline with RAGAS as the single evaluation standard.

What changed from the original
--------------------------------
- verify_claims()   → replaced by ragas.metrics.faithfulness        (same logic, standardised)
- grade_chunks()    → replaced by ragas.metrics.context_precision    (same logic, standardised)
- Self-correction   → still active, but now triggered by RAGAS faithfulness score
- answer_relevancy  → new metric, was never checked before
- context_recall    → available when ground_truth is supplied

Run modes
---------
  python crag_pipeline.py               # interactive Q&A
  python crag_pipeline.py --batch       # batch eval on TEST_CASES at bottom of file

Install
-------
  pip install ragas datasets langchain-community langchain-google-genai langchain-chroma
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

# ── LangChain ─────────────────────────────────────────────────────────────────
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

# ── RAGAS ─────────────────────────────────────────────────────────────────────
from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

GCP_PROJECT  = os.environ.get("GOOGLE_CLOUD_PROJECT")
GCP_LOCATION = "us-central1"
CHROMA_DIR   = "chromaDb"

# Models
MAIN_MODEL          = "gemini-2.5-flash"
EMBEDDING_MODEL     = "text-embedding-004"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# LLM
LLM_TEMPERATURE       = 0.2
LLM_MAX_OUTPUT_TOKENS = 2048
LLM_FAST_MAX_TOKENS   = 512

# Retrieval
VECTOR_K         = 5
VECTOR_FETCH_K   = 20
VECTOR_LAMBDA    = 0.5
BM25_K           = 5
ENSEMBLE_WEIGHTS = [0.4, 0.6]
RERANKER_TOP_N   = 3

# Web search
WEB_SEARCH_MAX_RESULTS = 3

# RAGAS thresholds  (replaces the old manual FAITHFULNESS_* thresholds)
RAGAS_PASS_THRESHOLD    = 0.85   # answer accepted as-is
RAGAS_RETRY_THRESHOLD   = 0.50   # attempt self-correction
RAGAS_RECHECK_THRESHOLD = 0.70   # accept corrected answer

HALLUCINATION_FALLBACK = (
    "I was unable to produce a fully verified answer from the available documents. "
    "Please rephrase your question or consult the source material directly."
)


# ═══════════════════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════════════════

def _gemini(max_tokens: int, temperature: float = 0) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=MAIN_MODEL,
        project=GCP_PROJECT,
        location=GCP_LOCATION,
        vertexai=True,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )

llm        = _gemini(LLM_MAX_OUTPUT_TOKENS, LLM_TEMPERATURE)  # streaming answer generation
llm_fast   = _gemini(LLM_FAST_MAX_TOKENS)                     # routing / rewriting / correction

embedding_model = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    project=GCP_PROJECT,
    location=GCP_LOCATION,
    vertexai=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# RAGAS setup  (single source of truth for all evaluation)
# ═══════════════════════════════════════════════════════════════════════════════

_ragas_llm        = LangchainLLMWrapper(llm_fast)
_ragas_embeddings = LangchainEmbeddingsWrapper(embedding_model)

# Metrics that don't need a ground-truth reference answer
METRICS_NO_GT   = [faithfulness, answer_relevancy, context_precision]

# Metrics that additionally need a ground-truth reference answer
METRICS_WITH_GT = [faithfulness, answer_relevancy, context_precision, context_recall]

for _m in METRICS_WITH_GT:
    _m.llm        = _ragas_llm
    _m.embeddings = _ragas_embeddings


# ═══════════════════════════════════════════════════════════════════════════════
# Retriever stack
# ═══════════════════════════════════════════════════════════════════════════════

vector_store = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_model,
)

# Pull all docs so BM25 can index them
_stored      = vector_store.get(include=["documents", "metadatas"])
_all_docs    = [
    Document(page_content=text, metadata=meta or {})
    for text, meta in zip(_stored["documents"], _stored["metadatas"])
]

bm25 = BM25Retriever.from_documents(_all_docs)
bm25.k = BM25_K

vector_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": VECTOR_K, "fetch_k": VECTOR_FETCH_K, "lambda_mult": VECTOR_LAMBDA},
)

_ensemble = EnsembleRetriever(
    retrievers=[bm25, vector_retriever],
    weights=ENSEMBLE_WEIGHTS,
)

retriever = ContextualCompressionRetriever(
    base_compressor=CrossEncoderReranker(
        model=HuggingFaceCrossEncoder(model_name=CROSS_ENCODER_MODEL),
        top_n=RERANKER_TOP_N,
    ),
    base_retriever=_ensemble,
)

web_search = TavilySearchResults(max_results=WEB_SEARCH_MAX_RESULTS)


# ═══════════════════════════════════════════════════════════════════════════════
# Chains
# ═══════════════════════════════════════════════════════════════════════════════

rewrite_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Rewrite the question as a short web-search query. Output only the query."),
        ("human", "{question}"),
    ])
    | llm_fast | StrOutputParser()
)

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. Be concise. "
               "If the answer is not in the context, say so explicitly."),
    ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
])

correction_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Rewrite the answer so every claim is directly supported by the context. "
                   "Fix or remove anything that is not grounded in the context."),
        ("human", (
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Previous answer:\n{previous_answer}"
        )),
    ])
    | llm_fast | StrOutputParser()
)


# ═══════════════════════════════════════════════════════════════════════════════
# RAGAS evaluation helpers
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RAGASResult:
    """Holds RAGAS scores for one Q&A turn."""
    faithfulness:      float
    answer_relevancy:  float
    context_precision: float
    context_recall:    Optional[float] = None

    @property
    def average(self) -> float:
        scores = [self.faithfulness, self.answer_relevancy, self.context_precision]
        if self.context_recall is not None:
            scores.append(self.context_recall)
        return sum(scores) / len(scores)

    def grade(self) -> str:
        avg = self.average
        if avg >= 0.75:
            return "✅  Good"
        if avg >= 0.50:
            return "⚠️   Mediocre"
        return "❌  Poor"

    def pretty(self) -> str:
        lines = [
            "┌─ RAGAS Scores ────────────────────────────────┐",
            f"│  Faithfulness      {self.faithfulness:.2f}   "
              "— answer grounded in context?    │",
            f"│  Answer Relevancy  {self.answer_relevancy:.2f}   "
              "— answer actually about question? │",
            f"│  Context Precision {self.context_precision:.2f}   "
              "— retrieved chunks on-point?      │",
        ]
        if self.context_recall is not None:
            lines.append(
                f"│  Context Recall    {self.context_recall:.2f}   "
                "— retrieval covered all needed?   │"
            )
        lines += [
            f"│  Overall avg       {self.average:.2f}   {self.grade():<27}│",
            "└───────────────────────────────────────────────┘",
        ]
        return "\n".join(lines)


def _evaluate_single(
    question: str,
    answer:   str,
    contexts: list[str],
    ground_truth: Optional[str] = None,
) -> RAGASResult:
    """
    Run RAGAS on one Q&A pair.
    context_recall is only computed when ground_truth is provided.
    """
    data: dict = {
        "question": [question],
        "answer":   [answer],
        "contexts": [contexts],
    }
    metrics = METRICS_NO_GT

    if ground_truth:
        data["ground_truth"] = [ground_truth]
        metrics = METRICS_WITH_GT

    result = ragas_evaluate(
        Dataset.from_dict(data),
        metrics=metrics,
        raise_exceptions=False,
    )

    return RAGASResult(
        faithfulness      = float(result["faithfulness"]),
        answer_relevancy  = float(result["answer_relevancy"]),
        context_precision = float(result["context_precision"]),
        context_recall    = float(result["context_recall"]) if ground_truth else None,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Retrieval helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def _web_search(question: str) -> list[Document]:
    """Rewrite question → search → return results as Documents."""
    query = rewrite_chain.invoke({"question": question}).strip() or question
    print(f"  Web query: '{query}'")
    try:
        results = web_search.invoke(query)
    except Exception as exc:
        print(f"  Web search failed: {exc}")
        return []
    return [
        Document(
            page_content=r.get("content", ""),
            metadata={"source": r.get("url", ""), "page": "web"},
        )
        for r in results if r.get("content")
    ]


def _route_docs(docs: list[Document], question: str) -> list[Document]:
    """
    RAGAS context_precision score replaces the old grade_chunks() loop.

    Strategy
    --------
    - Score the retrieved docs with RAGAS context_precision.
    - High precision  → local docs are sufficient.
    - Low precision   → supplement (or fully replace) with web search.
    """
    if not docs:
        print("  No docs retrieved — going straight to web search.")
        return _web_search(question)

    # Quick single-doc precision probe (cheap; one RAGAS call)
    probe = _evaluate_single(
        question = question,
        answer   = _format_docs(docs),   # use raw context as a stand-in answer
        contexts = [doc.page_content for doc in docs],
    )
    precision = probe.context_precision
    print(f"  Context precision probe: {precision:.2f}")

    if precision >= 0.70:
        print("  Precision sufficient — using local docs only.")
        return docs

    print("  Precision low — supplementing with web search.")
    web_docs = _web_search(question)
    return docs + web_docs if web_docs else docs


# ═══════════════════════════════════════════════════════════════════════════════
# Answer generation + RAGAS-gated correction
# ═══════════════════════════════════════════════════════════════════════════════

def _generate(context: str, question: str) -> str:
    """Stream the answer and return the full string."""
    answer = ""
    print("\nAnswer:")
    for chunk in llm.stream(answer_prompt.invoke({"context": context, "question": question})):
        print(chunk.content, end="", flush=True)
        answer += chunk.content
    print()
    return answer


def _verified_answer(
    question: str,
    answer:   str,
    context:  str,
    contexts: list[str],
) -> tuple[str, RAGASResult]:
    """
    Use RAGAS faithfulness to decide whether to accept, correct, or reject the answer.
    Returns the final answer string and its RAGAS scores.

    Replaces: verify_claims() + self_correction_chain logic from the original.
    """
    scores = _evaluate_single(question, answer, contexts)
    faith  = scores.faithfulness

    # ── Case 1: faithful enough ───────────────────────────────────────────────
    if faith >= RAGAS_PASS_THRESHOLD:
        return answer, scores

    # ── Case 2: partially faithful → try self-correction ─────────────────────
    if faith >= RAGAS_RETRY_THRESHOLD:
        print(f"  Faithfulness {faith:.2f} — attempting self-correction …")
        corrected = correction_chain.invoke({
            "context":         context,
            "question":        question,
            "previous_answer": answer,
        })
        re_scores = _evaluate_single(question, corrected, contexts)
        if re_scores.faithfulness >= RAGAS_RECHECK_THRESHOLD:
            print(f"  Corrected faithfulness {re_scores.faithfulness:.2f} — accepted.")
            return corrected, re_scores
        print(f"  Corrected faithfulness {re_scores.faithfulness:.2f} — still too low.")

    # ── Case 3: hallucination — safe fallback ─────────────────────────────────
    return HALLUCINATION_FALLBACK, scores


# ═══════════════════════════════════════════════════════════════════════════════
# Public API — single query
# ═══════════════════════════════════════════════════════════════════════════════

def run_query(
    question:     str,
    ground_truth: Optional[str] = None,
) -> tuple[str, RAGASResult]:
    """
    Full CRAG pipeline with RAGAS evaluation.

    Parameters
    ----------
    question     : the user's question
    ground_truth : optional reference answer — enables context_recall metric

    Returns
    -------
    (answer, RAGASResult)
    """
    print(f"\n{'═' * 52}")
    print(f"  Q: {question}")
    print(f"{'═' * 52}")

    # 1. Retrieve
    raw_docs = retriever.invoke(question)

    # 2. Route  (RAGAS context_precision replaces grade_chunks)
    final_docs = _route_docs(raw_docs, question)

    if not final_docs:
        fallback_scores = RAGASResult(0.0, 0.0, 0.0)
        return HALLUCINATION_FALLBACK, fallback_scores

    context  = _format_docs(final_docs)
    contexts = [doc.page_content for doc in final_docs]

    # 3. Generate
    raw_answer = _generate(context, question)

    # 4. Verify + correct  (RAGAS faithfulness replaces verify_claims)
    final_answer, scores = _verified_answer(question, raw_answer, context, contexts)

    # 5. If ground_truth supplied, re-run RAGAS to get context_recall too
    if ground_truth:
        scores = _evaluate_single(question, final_answer, contexts, ground_truth)

    return final_answer, scores


# ═══════════════════════════════════════════════════════════════════════════════
# Public API — batch evaluation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TestCase:
    question:     str
    answer:       str
    contexts:     list[str]
    ground_truth: Optional[str] = None


def evaluate_batch(test_cases: list[TestCase]) -> pd.DataFrame:
    """
    Run RAGAS over a pre-built test set (answers already generated).
    Useful for offline / CI evaluation without running the full pipeline.

    Returns a pandas DataFrame — one row per test case.
    """
    has_gt = all(tc.ground_truth for tc in test_cases)
    data: dict = {
        "question": [tc.question for tc in test_cases],
        "answer":   [tc.answer   for tc in test_cases],
        "contexts": [tc.contexts for tc in test_cases],
    }
    if has_gt:
        data["ground_truth"] = [tc.ground_truth for tc in test_cases]

    result = ragas_evaluate(
        Dataset.from_dict(data),
        metrics=METRICS_WITH_GT if has_gt else METRICS_NO_GT,
        raise_exceptions=False,
    )
    return result.to_pandas()


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

# Sample test set — replace / extend with your own cases
TEST_CASES: list[TestCase] = [
    TestCase(
        question     = "What is Corrective RAG?",
        answer       = "CRAG adds a correction step to standard RAG by grading retrieved chunks "
                       "and falling back to web search when local knowledge is insufficient.",
        contexts     = [
            "CRAG introduces a lightweight evaluator to assess retrieved documents.",
            "When chunks are irrelevant, CRAG falls back to web search.",
        ],
        ground_truth = "CRAG is a RAG variant that evaluates retrieved documents and uses "
                       "web search as a fallback when local knowledge is insufficient.",
    ),
    TestCase(
        question = "What embedding model does the pipeline use?",
        answer   = "The pipeline uses Google's text-embedding-004 model.",
        contexts = ["EMBEDDING_MODEL = 'text-embedding-004'"],
    ),
]


if __name__ == "__main__":

    # ── Batch mode ────────────────────────────────────────────────────────────
    if "--batch" in sys.argv:
        print("Running RAGAS batch evaluation …\n")
        df = evaluate_batch(TEST_CASES)
        print(df.to_string(index=False))
        sys.exit(0)

    # ── Interactive mode ──────────────────────────────────────────────────────
    print("CRAG + RAGAS  │  type '0' to exit")
    while True:
        q = input("\nYou: ").strip()
        if not q:
            continue
        if q == "0":
            print("Goodbye!")
            break

        answer, scores = run_query(q)
        print(f"\n{scores.pretty()}")
