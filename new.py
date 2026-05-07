import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field


# =============================================================
# Config values (from config.py)
# =============================================================

GCP_PROJECT     = os.environ.get("GOOGLE_CLOUD_PROJECT")
GCP_LOCATION    = "us-central1"
CHROMA_DIR      = "chromaDb"

MAIN_MODEL          = "gemini-2.5-flash"
FAST_MODEL          = "gemini-2.5-flash"
EMBEDDING_MODEL     = "text-embedding-004"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

LLM_TEMPERATURE       = 0.2
LLM_MAX_OUTPUT_TOKENS = 2048
LLM_FAST_MAX_TOKENS   = 512

VECTOR_K        = 5
VECTOR_FETCH_K  = 20
VECTOR_LAMBDA   = 0.5
BM25_K          = 5
ENSEMBLE_WEIGHTS    = [0.4, 0.6]
CROSS_ENCODER_TOP_N = 3

WEB_SEARCH_MAX_RESULTS = 3

FAITHFULNESS_PASS_THRESHOLD    = 0.85
FAITHFULNESS_RETRY_THRESHOLD   = 0.50
FAITHFULNESS_RECHECK_THRESHOLD = 0.70

MIN_CLAIM_WORDS = 4

HALLUCINATION_FALLBACK = (
    "I was unable to produce a fully verified answer from the available documents. "
    "Please rephrase your question or consult the source material directly."
)


# =============================================================
# DB Creation (from create_db.py)
# =============================================================

def load_documents(pdf_path: str) -> list:
    """Load a single PDF or a directory of PDFs."""
    if os.path.isdir(pdf_path):
        loader = DirectoryLoader(pdf_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    else:
        loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} page(s) from '{pdf_path}'")
    return documents


def split_documents(documents: list) -> list:
    """Split pages into smaller chunks so the LLM gets focused context."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunk(s)")
    return chunks


def create_vector_store(chunks: list) -> Chroma:
    """Embed each chunk with Vertex AI and store in ChromaDB."""
    embedding_model = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        project=GCP_PROJECT,
        location=GCP_LOCATION,
        vertexai=True,
    )

    BATCH_SIZE = 200  # Vertex AI max per call
    vector_store = None

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        print(f"Embedding batch {i // BATCH_SIZE + 1} ({len(batch)} chunk(s))...")
        if vector_store is None:
            vector_store = Chroma.from_documents(
                documents=batch,
                embedding=embedding_model,
                persist_directory=CHROMA_DIR,
                collection_metadata={"hnsw:space": "cosine"},
            )
        else:
            vector_store.add_documents(batch)

    print(f"Stored {len(chunks)} chunk(s) in '{CHROMA_DIR}'")
    return vector_store


def build_db(pdf_path: str):
    """Run the full DB creation pipeline: load → split → embed → store."""
    docs   = load_documents(pdf_path)
    chunks = split_documents(docs)
    store  = create_vector_store(chunks)
    print(f"ChromaDB contains {store._collection.count()} document(s). Ready.")
    return store


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
# =============================================================

embedding_model = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    project=GCP_PROJECT,
    location=GCP_LOCATION,
    vertexai=True,
)

# Load existing ChromaDB
vector_store = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_model,
)

# BM25 needs the raw documents pulled out of ChromaDB
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
        Document(page_content=r.get("content", ""), metadata={"source": r.get("url", ""), "page": "web"})
        for r in results if r.get("content")
    ]


def verify_claims(answer: str, context: str):
    """
    Extract claims from the answer and check each one against the context.
    Returns (score, unsupported_claims).
    score = fraction of claims that are supported.
    """
    extracted = claim_extractor_chain.invoke({"answer": answer})

    # Drop very short claims — they are usually not verifiable
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
        print(f"{len(relevant_docs)} relevant chunk(s) found — supplementing with web search.")
        web_docs    = web_search_docs(question)
        graded_web  = grade_chunks(web_docs, question)
        final_docs  = relevant_docs + graded_web

    # Guard: if still nothing after all that, return fallback
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

    # Case 1: answer is faithful enough — return it
    if score >= FAITHFULNESS_PASS_THRESHOLD:
        return answer

    # Case 2: partially faithful — try to self-correct
    if score >= FAITHFULNESS_RETRY_THRESHOLD:
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
        if re_score >= FAITHFULNESS_RECHECK_THRESHOLD:
            return corrected

    # Case 3: too many hallucinations — return safe fallback
    return HALLUCINATION_FALLBACK


# =============================================================
# Entry Point
# =============================================================

if __name__ == "__main__":
    import sys

    # If a PDF path is passed as argument, build the DB first
    # Usage: python pipeline.py MachineLearning.pdf
    if len(sys.argv) > 1:
        build_db(sys.argv[1])
        print("\nDB built. Starting query loop...\n")

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
        result = run_crag_query(question)
        print(f"\nFinal answer:\n{result}")
