# =============================================================
# RAG Pipeline - Database Creation (Upgraded)
# Vertex AI Embeddings + LangSmith Tracing
# Steps: Load PDF → Split → Embed (Vertex AI) → Store in ChromaDB
# =============================================================

# --- Environment Setup ---
from dotenv import load_dotenv
load_dotenv()

import os

# --- LangSmith Tracing Setup ---
# Add these to your .env file:
#   LANGCHAIN_TRACING_V2=true
#   LANGCHAIN_API_KEY=your_langsmith_api_key
#   LANGCHAIN_PROJECT=your_project_name        (e.g. "rag-pipeline")
#   GOOGLE_APPLICATION_CREDENTIALS=path/to/your-service-account-key.json
#   GOOGLE_CLOUD_PROJECT=your-gcp-project-id

# --- Imports ---
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings    # ← Vertex AI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langsmith import traceable                                    # ← LangSmith


# =============================================================
# Step 1: Load PDF(s)
# =============================================================

@traceable(name="load_documents")   # Every decorated fn is traced in LangSmith
def load_documents(pdf_path: str) -> list:
    """Load a single PDF or a directory of PDFs."""
    if os.path.isdir(pdf_path):
        # Load all PDFs from a folder
        loader = DirectoryLoader(pdf_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    else:
        loader = PyPDFLoader(pdf_path)

    documents = loader.load()
    print(f"[Loader] Loaded {len(documents)} page(s) from '{pdf_path}'")
    return documents


# =============================================================
# Step 2: Split into Chunks
# =============================================================

@traceable(name="split_documents")
def split_documents(documents: list) -> list:
    """
    Chunk documents using RecursiveCharacterTextSplitter.

    chunk_size=1000   → each chunk is at most 1000 characters
    chunk_overlap=200 → 200 characters shared between adjacent chunks
                        so context is not abruptly cut
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],  # tries these in order
    )
    chunks = splitter.split_documents(documents)
    print(f"[Splitter] Created {len(chunks)} chunk(s)")
    return chunks


# =============================================================
# Step 3: Embed and Store with Vertex AI
# =============================================================

@traceable(name="create_vector_store")
def create_vector_store(chunks: list, persist_dir: str = "chromaDb") -> Chroma:
    """
    Embed each chunk with Vertex AI text-embedding-004 and
    persist the vectors in ChromaDB.

    Vertex AI auth is picked up automatically from:
      GOOGLE_APPLICATION_CREDENTIALS env var  (service account JSON path)
      or Application Default Credentials      (gcloud auth application-default login)
    """
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        location="us-central1",
        vertexai=True,
    )

    BATCH_SIZE = 200  # Vertex AI allows max 250 instances per prediction call

    vector_store = None
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        print(f"[VectorStore] Embedding batch {i // BATCH_SIZE + 1} ({len(batch)} chunk(s))...")
        if vector_store is None:
            vector_store = Chroma.from_documents(
                documents=batch,
                embedding=embedding_model,
                persist_directory=persist_dir,
                collection_metadata={"hnsw:space": "cosine"},
            )
        else:
            vector_store.add_documents(batch)

    print(f"[VectorStore] Stored {len(chunks)} chunk(s) in '{persist_dir}'")
    return vector_store


# =============================================================
# Step 4: Verify the store
# =============================================================

@traceable(name="verify_store")
def verify_store(vector_store: Chroma) -> None:
    """Quick sanity check — count stored documents."""
    count = vector_store._collection.count()
    print(f"[Verify] ChromaDB contains {count} document(s). Ready for querying.")


# =============================================================
# Entrypoint
# =============================================================

if __name__ == "__main__":
    PDF_PATH   = "MachineLearning.pdf"  # single PDF or a folder path
    PERSIST_DIR = "chromaDb"

    docs   = load_documents(PDF_PATH)
    chunks = split_documents(docs)
    store  = create_vector_store(chunks, PERSIST_DIR)
    verify_store(store)

    print("\n✓ Database creation complete. Run rag_query.py to start querying.")