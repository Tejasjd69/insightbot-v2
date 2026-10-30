import uuid
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

_emb = None
def embeddings():
    global _emb
    if _emb is None:
        _emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _emb

class Index:
    """Chroma index with configurable chunking (cosine relevance scores in 0-1)."""
    def __init__(self, chunk_size=800, overlap=100):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        self.store = Chroma(collection_name=f"c{uuid.uuid4().hex[:8]}",
                            embedding_function=embeddings(),
                            collection_metadata={"hnsw:space": "cosine"})
        self.n = 0

    def ingest(self, path, name):
        docs = PyPDFLoader(path).load()
        for d in docs:
            d.metadata["source"] = name
        chunks = self.splitter.split_documents(docs)
        if chunks:
            self.store.add_documents(chunks)
            self.n += len(chunks)
        return len(chunks)

    def search(self, q, k=4):
        if self.n == 0:
            return []
        return self.store.similarity_search_with_relevance_scores(q, k=k)
