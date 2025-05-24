from langchain_core.embeddings import Embeddings

from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_qdrant.sparse_embeddings import SparseEmbeddings

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVector, SparseVectorParams, VectorParams

from FlagEmbedding import BGEM3FlagModel

from typing import List
import os


class BGEDenseEmbeddings(Embeddings):
    """LangChain-compatible dense embedder for BGE-M3 via FlagEmbedding."""
    def __init__(self, model: BGEM3FlagModel, device: str = "cpu"):
        self.model = model
        self.device = device

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )
        return out['dense_vecs']  # List[List[float]]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class BGESparseEmbeddings(SparseEmbeddings):
    """LangChain-compatible sparse embedder for BGE-M3 via FlagEmbedding."""
    def __init__(self, model: BGEM3FlagModel, device: str = "cpu"):
        self.model = model
        self.device = device

    def _to_sparse_vector(self, sparse_data: dict[int, float]) -> SparseVector:
        indices, values = [], []
        for token_id, weight in sparse_data.items():
            if weight > 0:
                indices.append(int(token_id))
                values.append(float(weight))
        return SparseVector(indices=indices, values=values)

    def embed_documents(self, texts: List[str]) -> List[SparseVector]:
        out = self.model.encode(
            texts,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False
        )
        return [self._to_sparse_vector(d) for d in out["lexical_weights"]]  # List[dict[int,float]]

    def embed_query(self, text: str) -> SparseVector:
        return self.embed_documents([text])[0]


def InitVectorStore(retrieval_mode="hybrid"):

    # init qdrant client
    print("Initializing Qdrant client...")
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    collection_name = "parsed_pdfs"
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense_vector": VectorParams(size=1024, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse_vector": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
            },
        )
    except:
        client.collection_exists(collection_name=collection_name)

    # init vectorstore
    print("Initializing vectorstore...")
    model_name = "BAAI/bge-m3"
    cache_dir = "embeddings"
    model = BGEM3FlagModel(model_name, cache_dir=cache_dir, use_fp16=True)
    dense_embeddings  = BGEDenseEmbeddings(model)
    sparse_embeddings = BGESparseEmbeddings(model)
    qdrant = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID if retrieval_mode == "hybrid" else RetrievalMode.DENSE,
        vector_name="dense_vector",
        sparse_vector_name="sparse_vector",
    )

    return qdrant

