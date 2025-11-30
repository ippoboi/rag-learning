from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .rag_service import answer_question


app = FastAPI(title="FastAPI RAG Demo")

# Allow calls from the simple frontend (including file:// with origin 'null')
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str


class Source(BaseModel):
    source: str
    corpus: Optional[str] = None
    section: Optional[str] = None
    category_path: Optional[str] = None
    top_level_category: Optional[str] = None
    subcategory: Optional[str] = None


class AnswerResponse(BaseModel):
    answer: str
    sources: List[Source] = []


@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest) -> AnswerResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        result = answer_question(request.question)
    except Exception as exc:  # pragma: no cover - surface underlying error
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    sources = [Source(**src) for src in result.get("sources", [])]
    return AnswerResponse(answer=result["answer"], sources=sources)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


# Serve the small frontend from / using StaticFiles
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount(
    "/",
    StaticFiles(directory=FRONTEND_DIR, html=True),
    name="frontend",
)


