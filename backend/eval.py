"""Usage: python eval.py doc.pdf questions.json
questions.json: [{"q": "question text", "keyword": "text that must appear in the right chunk"}]
Prints retrieval hit@4 for several chunk sizes -> real numbers for your resume."""
import json, sys
from rag import Index

pdf, qfile = sys.argv[1], sys.argv[2]
qs = json.load(open(qfile))
for size in (400, 800, 1500, 5000):
    ix = Index(size, size // 8)
    ix.ingest(pdf, pdf)
    hits = sum(any(q["keyword"].lower() in d.page_content.lower() for d, _ in ix.search(q["q"], 4)) for q in qs)
    print(f"chunk_size={size:5d}  hit@4={hits/len(qs):.2f}")
