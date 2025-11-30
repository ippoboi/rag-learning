from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.load import dumps, loads
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


# ---------------------------------------------------------------------------
# Environment & configuration
# ---------------------------------------------------------------------------

load_dotenv()

# Mirror the environment handling from the notebook so LangSmith and Groq work
for key in [
    "LANGSMITH_TRACING",
    "LANGSMITH_ENDPOINT",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "GROQ_API_KEY",
]:
    value = os.getenv(key)
    if value is not None:
        os.environ[key] = value


BASE_DIR = Path(__file__).resolve().parent.parent
VECTORDB_DIR = BASE_DIR / "chroma_fastapi"


# ---------------------------------------------------------------------------
# Vector stores (Chroma) & retrievers
# ---------------------------------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# These should point at the persisted collections created by the notebook.
# If you haven't run the notebook yet, these will be empty but still work.
dense_vs = Chroma(
    collection_name="fastapi_dense_docs",
    embedding_function=embeddings,
    persist_directory=str(VECTORDB_DIR / "dense"),
)

faq_vs = Chroma(
    collection_name="fastapi_faq_docs",
    embedding_function=embeddings,
    persist_directory=str(VECTORDB_DIR / "faq"),
)

dense_retriever = dense_vs.as_retriever(search_kwargs={"k": 4})
faq_retriever = faq_vs.as_retriever(search_kwargs={"k": 2})


# ---------------------------------------------------------------------------
# Routing logic (copied from the notebook)
# ---------------------------------------------------------------------------

def route_query(query: str) -> Dict[str, str]:
    """
    Route a query to the appropriate corpus(es) using rule-based logic.

    Returns a dict with key \"corpus\" set to \"dense_docs\", \"faq_docs\", or \"both\".
    """
    query_lower = query.lower()

    # FAQ indicators: STRONG error/debugging signals only (faq_docs is limited)
    strong_faq_keywords = [
        "500",
        "404",
        "403",
        "401",
        "400",  # HTTP error codes
        "stack trace",
        "traceback",  # Specific error indicators
        "doesn't work",
        "not working",
        "broken",  # Clear problem statements
        "troubleshoot",
        "debug",  # Explicit debugging intent
    ]

    # Moderate FAQ indicators: may use "both" for these
    moderate_faq_keywords = [
        "error",
        "exception",
        "failed",
        "failure",
        "crash",
        "issue",
        "bug",
        "problem",
    ]

    # Dense docs indicators: tutorials, guides, concepts, best practices
    dense_keywords = [
        "tutorial",
        "how to",
        "how do",
        "guide",
        "best practices",
        "best practice",
        "architecture",
        "design",
        "deployment",
        "security",
        "dependency injection",
        "background tasks",
        "middleware",
        "async",
        "asynchronous",
        "concurrency",
        "testing",
        "cors",
        "authentication",
        "authorization",
        "validation",
        "pydantic",
        "openapi",
        "swagger",
        "websocket",
        "websockets",
        "project structure",
        "structure",
        "conventions",
        "patterns",
    ]

    strong_faq_score = sum(
        1 for keyword in strong_faq_keywords if keyword in query_lower
    )
    moderate_faq_score = sum(
        1 for keyword in moderate_faq_keywords if keyword in query_lower
    )
    dense_score = sum(1 for keyword in dense_keywords if keyword in query_lower)

    query_length = len(query.split())
    is_short_query = query_length <= 3
    is_long_query = query_length > 10

    if strong_faq_score > 0:
        return {"corpus": "faq_docs"}

    if dense_score > 0 and dense_score > moderate_faq_score:
        return {"corpus": "dense_docs"}

    if moderate_faq_score > 0 and dense_score > 0:
        return {"corpus": "both"}

    if moderate_faq_score > 0 and is_long_query:
        return {"corpus": "both"}

    if moderate_faq_score > 0:
        return {"corpus": "dense_docs"}

    if is_short_query:
        return {"corpus": "dense_docs"}

    return {"corpus": "dense_docs"}


def get_retriever_for_query(query: str):
    routing = route_query(query)
    corpus = routing["corpus"]

    if corpus == "dense_docs":
        return dense_retriever
    if corpus == "faq_docs":
        return faq_retriever
    return [dense_retriever, faq_retriever]


# ---------------------------------------------------------------------------
# Multi-query reformulation & retrieval
# ---------------------------------------------------------------------------

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

multi_query_template = """You are an AI language model assistant. Your task is to generate three 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""

prompt_perspectives = ChatPromptTemplate.from_template(multi_query_template)

generate_queries = (
    prompt_perspectives
    | llm
    | StrOutputParser()
    | (lambda x: [q.strip() for q in x.split("\n") if q.strip()][:3])
)


def get_unique_union(documents: List[List[Any]]):
    """Unique union of retrieved docs from a list of lists."""
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]


def retrieve_multi_corpus(queries: List[str]):
    """Retrieve documents for multiple query variants across routed corpora."""
    all_results: List[List[Any]] = []
    for q in queries:
        retriever_or_list = get_retriever_for_query(q)
        docs_for_q: List[Any] = []
        if isinstance(retriever_or_list, list):
            for r in retriever_or_list:
                docs_for_q.extend(r.invoke(q))
        else:
            docs_for_q.extend(retriever_or_list.invoke(q))
        all_results.append(docs_for_q)
    return all_results


retrieval_chain = (
    generate_queries
    | RunnableLambda(retrieve_multi_corpus)
    | get_unique_union
)


# ---------------------------------------------------------------------------
# Final RAG chain: context + question -> answer text
# ---------------------------------------------------------------------------

answer_template = """Answer the following question based on this context:

{context}

Question: {question}
"""

answer_prompt = ChatPromptTemplate.from_template(answer_template)

final_rag_chain = (
    {
        "context": retrieval_chain,
        "question": lambda inp: inp["question"],
    }
    | answer_prompt
    | llm
    | StrOutputParser()
)


def answer_question(question: str) -> Dict[str, Any]:
    """
    Run the full RAG pipeline for a user question.

    Returns a dict with:
    - \"answer\": str
    - \"sources\": list of metadata dicts (file paths, categories, etc.)
    """
    if not question.strip():
        raise ValueError("Question must not be empty.")

    # Generate the final answer
    answer: str = final_rag_chain.invoke({"question": question})

    # Also expose the retrieval sources for the UI
    docs = retrieval_chain.invoke({"question": question})
    sources: Dict[str, Dict[str, Any]] = {}
    for d in docs:
        meta = d.metadata or {}
        source_path = meta.get("source")
        if not source_path:
            continue
        sources[source_path] = {
            "source": source_path,
            "corpus": meta.get("corpus"),
            "section": meta.get("section"),
            "category_path": meta.get("category_path"),
            "top_level_category": meta.get("top_level_category"),
            "subcategory": meta.get("subcategory"),
        }

    return {
        "answer": answer,
        "sources": list(sources.values()),
    }


__all__ = ["answer_question"]


