
import uuid

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

_emb = None


def embeddings():
    global _emb

    if _emb is None:
        print("Loading HuggingFace embedding model...")

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


class Index:
    """Chroma index with configurable chunking."""

    def __init__(self, chunk_size=800, overlap=100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )

        self.store = Chroma(
            collection_name=f"c{uuid.uuid4().hex[:8]}",
            embedding_function=embeddings(),
            collection_metadata={"hnsw:space": "cosine"}
        )

        self.n = 0

    def ingest(self, path, name):
        import time

        start = time.time()

        print(f"Processing PDF: {name}")

        docs = PyPDFLoader(path).load()

        print(f"Extracted {len(docs)} pages")

        for doc in docs:
            doc.metadata["source"] = name

        chunks = self.splitter.split_documents(docs)

        print(f"Generated {len(chunks)} chunks")

        batch_size = 64

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            self.store.add_documents(batch)

            print(
                f"Indexed {min(i + batch_size, len(chunks))}"
                f"/{len(chunks)} chunks"
            )

        self.n += len(chunks)

        print(
            f"Indexing completed in "
            f"{time.time() - start:.2f} seconds"
        )

        return len(chunks)

    def search(self, q, k=4):
        if self.n == 0:
            return []

        return self.store.similarity_search_with_relevance_scores(
            q, k=k
        )
