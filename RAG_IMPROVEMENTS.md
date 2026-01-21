# RAG Implementation Review & Improvements

A comprehensive analysis of this RAG implementation with actionable improvements for future learning.

---

## Table of Contents

- [What You Did Well](#what-you-did-well)
- [Areas for Improvement](#areas-for-improvement)
- [Advanced Techniques to Explore](#advanced-techniques-to-explore)
- [Recommended Learning Path](#recommended-learning-path)
- [Code Improvement Summary](#code-improvement-summary)
- [Resources](#resources)

---

## What You Did Well

### 1. Multi-Corpus Architecture

Your separation of dense docs (tutorials) from FAQ docs (troubleshooting) is a smart design pattern. This is similar to what production systems do — different document types often need different retrieval strategies.

### 2. Metadata Enrichment

Extracting navigation metadata from `mkdocs.yml` is excellent. Many RAG learners skip this, but metadata is crucial for:

- Better filtering
- Source attribution
- Hybrid search strategies

### 3. Multi-Query Reformulation

Using LLM to generate query variants is a solid technique to overcome embedding limitations. This addresses the "vocabulary mismatch" problem in semantic search.

```python
# Your implementation in rag_service.py
template = """You are an AI language model assistant. Your task is to generate three
different versions of the given user question to retrieve relevant documents from a vector
database..."""
```

### 4. Deduplication Logic

Your `get_unique_union` using serialization/deserialization is correct and handles LangChain documents properly.

### 5. Clean Architecture

- Separation of notebook (prototyping) → backend (production)
- Proper use of Pydantic models for API contracts
- Environment variable handling
- LangSmith integration for observability

---

## Areas for Improvement

### 1. Double Retrieval Bug

**Location:** `backend/rag_service.py` lines 270-274

**Problem:** The current code runs retrieval TWICE - once in the chain and once for source extraction:

```python
# Current code (wasteful)
answer: str = final_rag_chain.invoke({"question": question})  # First retrieval
docs = retrieval_chain.invoke({"question": question})  # Second retrieval!
```

**Fix:** Retrieve once, then use the same docs for both answer generation and source extraction:

```python
def answer_question(question: str) -> Dict[str, Any]:
    if not question.strip():
        raise ValueError("Question must not be empty.")

    # Retrieve once
    docs = retrieval_chain.invoke({"question": question})

    # Format context
    context = "\n\n".join(d.page_content for d in docs)

    # Generate answer using retrieved context
    answer = (answer_prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "question": question
    })

    # Extract sources from already-retrieved docs
    sources = {}
    for d in docs:
        meta = d.metadata or {}
        source_path = meta.get("source")
        if source_path:
            sources[source_path] = {
                "source": source_path,
                "corpus": meta.get("corpus"),
                "section": meta.get("section"),
                "category_path": meta.get("category_path"),
                "top_level_category": meta.get("top_level_category"),
                "subcategory": meta.get("subcategory"),
            }

    return {"answer": answer, "sources": list(sources.values())}
```

---

### 2. Rule-Based Routing is Brittle

**Location:** `backend/rag_service.py` lines 70-170

**Problem:** Your keyword-based routing will fail on edge cases:

- "My CORS setup is broken" → routes to FAQ (because "broken") but should check dense for CORS
- "Explain the 404 response model" → routes to FAQ but it's asking about documentation

**Better Approaches:**

#### Option A: LLM-based routing (using structured output)

```python
from pydantic import BaseModel
from typing import Literal

class RouteDecision(BaseModel):
    corpus: Literal["dense_docs", "faq_docs", "both"]
    reasoning: str

routing_prompt = """Analyze this question and decide which corpus to search:
- dense_docs: tutorials, guides, how-to content, feature explanations
- faq_docs: troubleshooting, error messages, debugging, common issues
- both: when the question spans multiple concerns

Question: {question}
"""

routing_llm = llm.with_structured_output(RouteDecision)
```

#### Option B: Always retrieve from both and let reranking decide

This is often simpler and more robust — retrieve from all sources, then rerank by relevance.

---

### 3. Missing Reranking Step

**Problem:** After retrieval, you directly use documents without reranking. Adding a reranker significantly improves precision.

**Implementation:**

```python
from sentence_transformers import CrossEncoder

# Load reranker model (one-time)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_docs(question: str, docs: List[Document], top_k: int = 4) -> List[Document]:
    """Rerank documents by relevance to the question."""
    if not docs:
        return []

    # Create question-document pairs
    pairs = [(question, d.page_content) for d in docs]

    # Score each pair
    scores = reranker.predict(pairs)

    # Sort by score (descending) and return top_k
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]


# Integration into retrieval chain
retrieval_chain = (
    generate_queries
    | RunnableLambda(retrieve_multi_corpus)
    | get_unique_union
    | RunnableLambda(lambda docs: rerank_docs(question, docs))  # Add this
)
```

**Alternative: Use Cohere Rerank API**

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

compressor = CohereRerank(model="rerank-english-v3.0")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=dense_retriever
)
```

---

### 4. Chunking Strategy Could Be Smarter

**Location:** Jupyter notebook chunking cell

**Current approach:**

```python
# Fixed-size chunks - cuts through code blocks and markdown structure
RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
```

**Issues:**

- Cuts in the middle of code blocks
- Doesn't respect Markdown structure (headers, lists)
- May split important context

**Better: Structure-aware splitting**

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

# Split by markdown headers first
headers_to_split = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split)

# Then apply size-based splitting to large sections
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", "```", ". ", " "]  # Respect code blocks
)

# Two-stage splitting
def smart_split(doc):
    md_splits = md_splitter.split_text(doc.page_content)
    final_chunks = []
    for split in md_splits:
        if len(split.page_content) > 1000:
            final_chunks.extend(text_splitter.split_documents([split]))
        else:
            final_chunks.append(split)
    return final_chunks
```

**Even Better: Semantic Chunking**

```python
from langchain_experimental.text_splitter import SemanticChunker

semantic_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile"
)
```

---

### 5. Evaluation is Too Basic

**Current approach:**

```python
def score_answer_keywords(answer: str, keywords: List[str]) -> float:
    """Fraction of keywords that appear in the answer."""
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords)
```

**Issues:**

- "Depends" might appear in wrong context
- Doesn't measure answer correctness
- Can't detect hallucinations
- No context quality evaluation

**Better: Use RAGAS Framework**

```python
# pip install ragas

from ragas import evaluate
from ragas.metrics import (
    faithfulness,        # Is the answer grounded in context?
    answer_relevancy,    # Is the answer relevant to the question?
    context_precision,   # Are retrieved docs relevant?
    context_recall,      # Did we retrieve all needed info?
)
from datasets import Dataset

# Prepare evaluation data
eval_data = {
    "question": ["What is dependency injection?", ...],
    "answer": [generated_answers],
    "contexts": [retrieved_contexts],
    "ground_truth": [expected_answers],  # Optional but recommended
}

dataset = Dataset.from_dict(eval_data)

# Run evaluation
results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)

print(results)
```

**Alternative: LLM-as-Judge**

```python
judge_prompt = """You are evaluating a RAG system's response.

Question: {question}
Retrieved Context: {context}
Generated Answer: {answer}

Rate the following on a scale of 1-5:

1. **Relevance**: Does the answer address the question? (1=off-topic, 5=perfectly relevant)
2. **Faithfulness**: Is the answer supported by the context? (1=hallucinated, 5=fully grounded)
3. **Completeness**: Does the answer cover all aspects? (1=incomplete, 5=comprehensive)

Provide scores and brief justification for each."""

def evaluate_with_llm(question, context, answer):
    response = llm.invoke(judge_prompt.format(
        question=question,
        context=context,
        answer=answer
    ))
    return response
```

---

### 6. Missing Hybrid Search

**Problem:** You're only using dense (semantic) retrieval. Adding sparse (keyword) search improves recall, especially for:

- Exact term matching (API names, error codes)
- Rare terms not well-represented in embeddings

**Implementation:**

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Create BM25 (sparse) retriever from same docs
bm25_retriever = BM25Retriever.from_documents(dense_chunks)
bm25_retriever.k = 4

# Combine dense + sparse with weighted fusion
ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, bm25_retriever],
    weights=[0.6, 0.4]  # Tune these weights
)
```

**Alternative: Use a vector DB with hybrid support**

```python
# Weaviate, Pinecone, and Qdrant support hybrid search natively
from langchain_weaviate import WeaviateVectorStore

vectorstore = WeaviateVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    client=weaviate_client,
)

# Hybrid search built-in
retriever = vectorstore.as_retriever(
    search_type="hybrid",
    search_kwargs={"alpha": 0.5}  # 0=sparse, 1=dense, 0.5=balanced
)
```

---

## Advanced Techniques to Explore

| Technique | Difficulty | Impact | Description |
|-----------|------------|--------|-------------|
| **Contextual Embeddings** | Medium | High | Prepend document context before embedding ([Anthropic's technique](https://www.anthropic.com/news/contextual-retrieval)) |
| **HyDE** | Medium | Medium | Generate hypothetical document, then search for similar real docs |
| **Parent Document Retrieval** | Medium | High | Store small chunks for precision, retrieve parent docs for context |
| **Agentic RAG** | Hard | Very High | Let LLM decide when/how to retrieve, iterate if needed |
| **Graph RAG** | Hard | High | Extract entities, build knowledge graph, traverse for context |
| **Late Interaction (ColBERT)** | Hard | High | Token-level matching for better precision |
| **Query Decomposition** | Medium | High | Break complex queries into sub-queries |
| **Self-RAG** | Hard | High | Model decides when to retrieve and self-critiques |

### Contextual Embeddings Example

```python
def add_context_to_chunk(chunk: Document, doc_summary: str) -> Document:
    """Prepend document context to chunk for better embeddings."""
    contextualized_content = f"""Document: {doc_summary}

Section: {chunk.metadata.get('section', 'Unknown')}

Content:
{chunk.page_content}"""

    return Document(
        page_content=contextualized_content,
        metadata=chunk.metadata
    )
```

### HyDE (Hypothetical Document Embeddings)

```python
hyde_prompt = """Given a question, generate a hypothetical document passage that would answer it.
The passage should be detailed and factual, as if from official documentation.

Question: {question}

Hypothetical passage:"""

def hyde_retrieval(question: str, retriever) -> List[Document]:
    # Generate hypothetical answer
    hypothetical_doc = llm.invoke(hyde_prompt.format(question=question))

    # Use hypothetical doc to retrieve real docs
    return retriever.invoke(hypothetical_doc)
```

---

## Recommended Learning Path

### Phase 1: Solidify Fundamentals (Current Stage)

- [x] Basic RAG pipeline
- [x] Multi-query reformulation
- [x] Vector stores & embeddings
- [x] Metadata enrichment
- [ ] **Add reranking** ← High priority
- [ ] **Implement hybrid search** ← High priority
- [ ] **Better evaluation (RAGAS)** ← Essential for iteration

### Phase 2: Production Patterns

- [ ] Streaming responses (better UX)
- [ ] Caching (embedding cache, query cache)
- [ ] Observability & tracing (expand LangSmith usage)
- [ ] Guardrails & content filtering
- [ ] Error handling & fallbacks
- [ ] Rate limiting & cost management

### Phase 3: Advanced RAG

- [ ] Agentic retrieval (self-querying, iterative)
- [ ] Contextual compression
- [ ] Graph RAG (try with Neo4j or similar)
- [ ] Fine-tuned embeddings for your domain
- [ ] Query routing with LLM
- [ ] Multi-modal RAG (images, tables)

### Phase 4: Beyond RAG

- [ ] Tool use & function calling
- [ ] Multi-agent systems
- [ ] Fine-tuning LLMs (LoRA/QLoRA)
- [ ] Building custom embeddings
- [ ] Deployment & scaling (K8s, serverless)

---

## Code Improvement Summary

| Location | Issue | Priority | Fix |
|----------|-------|----------|-----|
| `rag_service.py:270-274` | Double retrieval | **High** | Retrieve once, reuse docs |
| `rag_service.py:70-170` | Brittle routing | Medium | Use LLM routing or always-both |
| `rag_service.py:188` | Hardcoded LLM model | Low | Make configurable via env var |
| `rag_service.py:62-63` | Fixed k values | Low | Make configurable, consider dynamic k |
| notebook chunking | Fixed-size splits | Medium | Use `MarkdownHeaderTextSplitter` |
| `app.py:17` | CORS allows all origins | Medium | Restrict in production |
| evaluation loop | Keyword matching only | **High** | Use RAGAS or LLM-as-judge |
| retrieval | No reranking | **High** | Add cross-encoder reranker |
| retrieval | Dense only | Medium | Add BM25 for hybrid search |

---

## Resources

### Reranking

- [Cohere Rerank API](https://docs.cohere.com/docs/reranking)
- [Cross-Encoders (Sentence Transformers)](https://www.sbert.net/docs/cross_encoder/pretrained_models.html)
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`

### Evaluation

- [RAGAS Documentation](https://docs.ragas.io/) — Industry standard for RAG evaluation
- [LangSmith Evaluations](https://docs.smith.langchain.com/evaluation)

### Advanced RAG

- [LangChain RAG Techniques](https://python.langchain.com/docs/tutorials/rag/)
- [Contextual Retrieval (Anthropic)](https://www.anthropic.com/news/contextual-retrieval)
- [Microsoft GraphRAG](https://github.com/microsoft/graphrag)

### Chunking

- [Chunking Strategies (Pinecone)](https://www.pinecone.io/learn/chunking-strategies/)
- [Semantic Chunking (LangChain)](https://python.langchain.com/docs/how_to/semantic-chunker/)

### Hybrid Search

- [Hybrid Search Explained](https://weaviate.io/blog/hybrid-search-explained)
- [BM25 + Dense Fusion](https://www.pinecone.io/learn/hybrid-search-intro/)

### Production RAG

- [RAG Best Practices (LlamaIndex)](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/)
- [Building Production RAG (Anthropic)](https://docs.anthropic.com/en/docs/build-with-claude/retrieval-augmented-generation)

---

## Summary

This RAG implementation demonstrates solid understanding of fundamentals. The architecture is clean, tools are appropriate (LangChain, Chroma, HuggingFace), and important aspects like metadata enrichment and query reformulation are implemented.

**Top 3 Next Actions:**

1. **Fix the double-retrieval bug** — Quick win, improves latency
2. **Add a reranking step** — Biggest quality improvement
3. **Implement proper evaluation with RAGAS** — Essential for measuring progress

Keep building and iterating!
