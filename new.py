import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.retrievers import BM25Retriever
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from pydantic import BaseModel, Field

# ── Models ──────────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7, streaming=True)
llm_fast = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# ── Retriever ────────────────────────────────────────────────────────────
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# BM25 needs the raw documents
all_docs_data = vector_store.get(include=["documents", "metadatas"])
all_docs = [
    Document(page_content=text, metadata=meta or {})
    for text, meta in zip(all_docs_data["documents"], all_docs_data["metadatas"])
]

bm25 = BM25Retriever.from_documents(all_docs)
bm25.k = 4
vector_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20})

ensemble = EnsembleRetriever(retrievers=[bm25, vector_retriever], weights=[0.5, 0.5])

cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
retriever = ContextualCompressionRetriever(
    base_compressor=CrossEncoderReranker(model=cross_encoder, top_n=5),
    base_retriever=ensemble,
)

web_search = TavilySearchResults(max_results=3)

# ── Structured output schemas ────────────────────────────────────────────
class GradeResult(BaseModel):
    score: str = Field(description="'yes' if relevant, 'no' if not")

class ExtractedClaims(BaseModel):
    claims: list[str] = Field(description="Individual factual claims from the answer")

class ClaimVerdict(BaseModel):
    supported: bool = Field(description="True if context supports this claim")

# ── Chains ───────────────────────────────────────────────────────────────
grader_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Is this document chunk relevant to the question? Reply 'yes' or 'no'."),
        ("human", "Chunk:\n{document}\n\nQuestion: {question}"),
    ])
    | llm_fast.with_structured_output(GradeResult)
)

rewrite_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Rewrite the question as a short web search query. Output only the query."),
        ("human", "{question}"),
    ])
    | llm_fast | StrOutputParser()
)

claim_extractor_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Extract every individual factual claim from this answer as a list."),
        ("human", "Answer:\n{answer}"),
    ])
    | llm_fast.with_structured_output(ExtractedClaims)
)

claim_checker_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Does the context support this claim? True or False."),
        ("human", "Context:\n{context}\n\nClaim: {claim}"),
    ])
    | llm_fast.with_structured_output(ClaimVerdict)
)

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say so."),
    ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
])

self_correction_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Rewrite the answer so every claim is supported by the context. Remove or fix anything unsupported."),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}\n\nPrevious answer:\n{previous_answer}\n\nUnsupported claims to fix:\n{unsupported_claims}"),
    ])
    | llm_fast | StrOutputParser()
)

# ── Helper functions ─────────────────────────────────────────────────────
def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def grade_chunks(docs, question):
    """Return only the relevant chunks."""
    relevant = []
    for doc in docs:
        result = grader_chain.invoke({"document": doc.page_content, "question": question})
        if result.score.lower() == "yes":
            relevant.append(doc)
    return relevant

def web_search_docs(question):
    """Rewrite query, search the web, return results as Documents."""
    rewritten = rewrite_chain.invoke({"question": question}).strip() or question
    results = web_search.invoke(rewritten)
    return [
        Document(page_content=r.get("content", ""), metadata={"source": r.get("url", "")})
        for r in results if r.get("content")
    ]

def verify_claims(answer, context):
    """Check each claim in the answer against the context. Return (score, unsupported_claims)."""
    claims = claim_extractor_chain.invoke({"answer": answer}).claims
    if not claims:
        return 1.0, []

    unsupported = []
    for claim in claims:
        verdict = claim_checker_chain.invoke({"context": context, "claim": claim})
        if not verdict.supported:
            unsupported.append(claim)

    score = (len(claims) - len(unsupported)) / len(claims)
    return score, unsupported

# ── Main pipeline ─────────────────────────────────────────────────────────
def run_crag_query(question):
    # 1. Retrieve
    docs = retriever.invoke(question)

    # 2. Grade local docs
    relevant_docs = grade_chunks(docs, question)

    # 3. Route based on how many came back relevant
    if len(relevant_docs) == len(docs) and docs:
        # All relevant — use local only
        final_docs = relevant_docs
    elif not relevant_docs:
        # None relevant — go to web
        web_docs = web_search_docs(question)
        final_docs = grade_chunks(web_docs, question)
    else:
        # Mixed — combine local + web
        web_docs = web_search_docs(question)
        graded_web = grade_chunks(web_docs, question)
        final_docs = relevant_docs + graded_web

    if not final_docs:
        return "I couldn't find relevant information to answer your question."

    # 4. Generate
    context = format_docs(final_docs)
    answer = ""
    print("\nAnswer:")
    for chunk in llm.stream(answer_prompt.invoke({"context": context, "question": question})):
        print(chunk.content, end="", flush=True)
        answer += chunk.content
    print()

    # 5. Verify faithfulness
    score, unsupported = verify_claims(answer, context)
    print(f"\nFaithfulness: {score:.0%}")

    if score >= 0.8:
        return answer

    if score >= 0.5:
        # Try to self-correct
        unsupported_list = "\n".join(f"- {c}" for c in unsupported)
        corrected = self_correction_chain.invoke({
            "context": context,
            "question": question,
            "previous_answer": answer,
            "unsupported_claims": unsupported_list,
        })
        re_score, _ = verify_claims(corrected, context)
        if re_score >= 0.8:
            return corrected

    return "I'm not confident enough in my answer to share it. Please consult a primary source."


# ── Run ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    while True:
        question = input("\nYou: ").strip()
        if question == "0":
            break
        print(run_crag_query(question))
