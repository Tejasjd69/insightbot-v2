import uuid
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

_emb = None

def embeddings():
    global _emb

    if _emb is None:
        print("Loading embedding model...")

        _emb = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "batch_size": 32,
                "normalize_embeddings": True
            }
        )

        print("Embedding model loaded successfully!")

    return _emb
    
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
