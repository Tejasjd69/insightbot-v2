import os, tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import Index
from graph import build_graph

app = FastAPI(title="InsightBot v2")
   
frontend_url = os.getenv(
    "FRONTEND_URL",
    "http://localhost:5173"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        frontend_url
    ],
    allow_methods=["*"],
    allow_headers=["*"]
)  
   
   
index = Index()
graph = build_graph(index)
sessions: dict = {}

class Chat(BaseModel):
    session_id: str = "default"
    question: str

@app.get("/api/health")
def health():
    return {"ok": True, "chunks": index.n}

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(await file.read())
        path = f.name
    try:
        n = index.ingest(path, file.filename)
    finally:
        os.remove(path)
    return {"file": file.filename, "chunks": n}

@app.post("/api/chat")
def chat(req: Chat):
    hist = sessions.setdefault(req.session_id, [])
    h = "\n".join(f"{r}: {t}" for r, t in hist[-6:])
    out = graph.invoke({"question": req.question, "query": req.question, "history": h, "attempts": 0})
    hist += [("user", req.question), ("assistant", out["answer"])]
    return {"answer": out["answer"], "grounded": out.get("grounded", False), "trace": out["trace"],
            "sources": [{"page": d.metadata.get("page", 0) + 1, "source": d.metadata.get("source"),
                         "text": d.page_content[:200]} for d in out.get("docs", [])]}
