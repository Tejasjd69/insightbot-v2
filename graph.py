import os, time
from typing import TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

THRESH = float(os.getenv("REL_THRESH", "0.25"))  # min retrieval relevance to answer
_llm = None
def llm():
    global _llm
    if _llm is None:  # reads GROQ_API_KEY from env; change model via GROQ_MODEL if deprecated
        _llm =ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    temperature=0
)
    return _llm

class S(TypedDict, total=False):
    question: str; history: str; query: str; docs: list; best: float
    attempts: int; answer: str; grounded: bool; trace: list

def _t(s, name, t0, info):
    return s.get("trace", []) + [{"step": name, "ms": int((time.time() - t0) * 1000), "info": info}]

def _ctx(docs):
    return "\n\n".join(f"[{i+1}] (p.{d.metadata.get('page', 0)+1}) {d.page_content}" for i, d in enumerate(docs))

def build_graph(index):
    def retrieve(s):
        t0 = time.time()
        res = index.search(s["query"])
        best = max([sc for _, sc in res], default=0.0)
        return {"docs": [d for d, _ in res], "best": best,
                "trace": _t(s, "retrieve", t0, f"best relevance {best:.2f}, {len(res)} chunks")}

    def route(s):  # agentic step: weak retrieval -> rewrite query and retry (max 2)
        return "rewrite" if s["best"] < THRESH and s.get("attempts", 0) < 2 else "answer"

    def rewrite(s):
        t0 = time.time()
        q = llm().invoke(f"Rewrite the question as a standalone search query for a PDF.\n"
                         f"Chat so far:\n{s['history']}\nQuestion: {s['question']}\nReturn only the query.").content.strip()
        return {"query": q, "attempts": s.get("attempts", 0) + 1, "trace": _t(s, "rewrite", t0, q)}

    def answer(s):
        t0 = time.time()
        if s["best"] < THRESH:  # guardrail: refuse instead of guessing
            return {"answer": "I couldn't find this in the uploaded document.", "grounded": True,
                    "trace": _t(s, "refuse", t0, "retrieval relevance below threshold")}
        a = llm().invoke(f"Answer using ONLY the context. Cite chunk numbers like [1]. "
                         f"If the context lacks the answer, say you don't know.\n\nContext:\n{_ctx(s['docs'])}\n\n"
                         f"Chat so far:\n{s['history']}\nQuestion: {s['question']}").content.strip()
        return {"answer": a, "trace": _t(s, "answer", t0, f"{len(a)} chars")}

    def check(s):  # guardrail: LLM-as-judge grounding check
        if s.get("grounded"):
            return {}
        t0 = time.time()
        v = llm().invoke(f"Context:\n{_ctx(s['docs'])}\n\nAnswer: {s['answer']}\n\n"
                         f"Is every claim in the answer supported by the context? Reply YES or NO.").content
        ok = v.strip().upper().startswith("YES")
        return {"grounded": ok, "trace": _t(s, "grounding_check", t0, "supported" if ok else "NOT fully supported")}

    g = StateGraph(S)
    for name, fn in [("retrieve", retrieve), ("rewrite", rewrite), ("answer", answer), ("check", check)]:
        g.add_node(name, fn)
    g.set_entry_point("retrieve")
    g.add_conditional_edges("retrieve", route, {"rewrite": "rewrite", "answer": "answer"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("answer", "check")
    g.add_edge("check", END)
    return g.compile()
