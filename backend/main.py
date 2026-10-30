
import os
import tempfile
import threading

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import Index
from graph import build_graph

app = FastAPI(title="InsightBot v2")

frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        frontend_url
    ],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize heavy ML components only when needed
index = None
graph = None
init_lock = threading.Lock()


def initialize():
    global index, graph

    if index is None:
        with init_lock:
            if index is None:
                print("Loading HuggingFace embedding model...")
                index = Index()
                graph = build_graph(index)
                print("InsightBot initialized successfully!")


sessions: dict = {}


class Chat(BaseModel):
    session_id: str = "default"
    question: str


@app.get("/")
def root():
    return {"message": "InsightBot API is running"}


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "initialized": index is not None,
        "chunks": index.n if index else 0
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    initialize()

    with tempfile.NamedTemporaryFile(
        suffix=".pdf", delete=False
    ) as f:
        f.write(await file.read())
        path = f.name

    try:
        n = index.ingest(path, file.filename)
    finally:
        os.remove(path)

    return {"file": file.filename, "chunks": n}


@app.post("/api/chat")
def chat(req: Chat):
    initialize()

    hist = sessions.setdefault(req.session_id, [])
    h = "\n".join(
        f"{r}: {t}" for r, t in hist[-6:]
    )

    out = graph.invoke({
        "question": req.question,
        "query": req.question,
        "history": h,
        "attempts": 0
    })

    hist += [
        ("user", req.question),
        ("assistant", out["answer"])
    ]

    return {
        "answer": out["answer"],
        "grounded": out.get("grounded", False),
        "trace": out["trace"],
        "sources": [
            {
                "page": d.metadata.get("page", 0) + 1,
                "source": d.metadata.get("source"),
                "text": d.page_content[:200]
            }
            for d in out.get("docs", [])
        ]
    }
