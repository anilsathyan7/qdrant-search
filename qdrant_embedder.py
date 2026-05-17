from typing import Any, Dict, List

import torch
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
from FlagEmbedding import FlagAutoModel
from qdrant_client import models
from sentence_transformers import SentenceTransformer

from qdrant_config import (
    BGE_M3_MODEL_ID,
    COLBERT_MODEL_ID,
    DENSE_MODEL_ID,
    EMBEDDING_DEVICE,
    JINA_V5_DENSE_MODEL_ID,
    MINICOIL_SPARSE_MODEL_ID,
    SPARSE_MODEL_ID,
)


class DocumentEmbedder:
    """Generate dense, sparse, and ColBERT embeddings with FastEmbed models."""

    def __init__(
        self,
        dense_model_id: str = DENSE_MODEL_ID,
        sparse_model_id: str = SPARSE_MODEL_ID,
        colbert_model_id: str = COLBERT_MODEL_ID,
        device: str = EMBEDDING_DEVICE,
    ):
        cuda = device.startswith("cuda") and torch.cuda.is_available()
        self.dense_model = TextEmbedding(dense_model_id, cuda=cuda)
        self.sparse_model = SparseTextEmbedding(sparse_model_id, cuda=cuda)
        self.colbert_model = LateInteractionTextEmbedding(colbert_model_id, cuda=cuda)

    def vector_dimensions(self) -> Dict[str, Any]:
        return {
            "dense_size": self.dense_model.embedding_size,
            "colbert_size": self.colbert_model.embedding_size,
        }

    def sparse_modifier(self) -> models.Modifier | None:
        return None

    def embed_texts(self, texts: List[str], parallel: int = 0) -> Dict[str, List[Any]]:
        return {
            "dense": list(self.dense_model.passage_embed(texts, parallel=parallel)),
            "sparse": list(self.sparse_model.passage_embed(texts, parallel=parallel)),
            "colbert": list(self.colbert_model.passage_embed(texts, parallel=parallel)),
        }

    def embed_query(self, query_text: str, parallel: int = 0) -> Dict[str, Any]:
        return {
            "dense": next(self.dense_model.query_embed(query_text, parallel=parallel)),
            "sparse": next(
                self.sparse_model.query_embed(query_text, parallel=parallel)
            ).as_object(),
            "colbert": next(self.colbert_model.query_embed(query_text, parallel=parallel)),
        }


class JinaDocumentEmbedder:
    """Generate Jina dense, miniCOIL sparse, and ColBERT embeddings."""

    def __init__(
        self,
        dense_model_id: str = JINA_V5_DENSE_MODEL_ID,
        sparse_model_id: str = MINICOIL_SPARSE_MODEL_ID,
        colbert_model_id: str = COLBERT_MODEL_ID,
        device: str = EMBEDDING_DEVICE,
    ):
        cuda = device.startswith("cuda") and torch.cuda.is_available()
        self.device = device
        self.dense_model = SentenceTransformer(
            dense_model_id,
            device=device,
            trust_remote_code=True,
        )
        self.sparse_model = SparseTextEmbedding(sparse_model_id, cuda=cuda)
        self.colbert_model = LateInteractionTextEmbedding(colbert_model_id, cuda=cuda)

    def vector_dimensions(self) -> Dict[str, Any]:
        return {
            "dense_size": self.dense_model.get_embedding_dimension(),
            "colbert_size": self.colbert_model.embedding_size,
        }

    def sparse_modifier(self) -> models.Modifier | None:
        return models.Modifier.IDF

    def embed_texts(self, texts: List[str], parallel: int = 0) -> Dict[str, List[Any]]:
        dense_vectors = [vector.tolist() for vector in self.dense_model.encode(texts, prompt_name="document")]
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        return {
            "dense": dense_vectors,
            "sparse": list(self.sparse_model.passage_embed(texts, parallel=parallel)),
            "colbert": list(self.colbert_model.passage_embed(texts, parallel=parallel)),
        }

    def embed_query(self, query_text: str, parallel: int = 0) -> Dict[str, Any]:
        return {
            "dense": self.dense_model.encode([query_text], prompt_name="query")[0].tolist(),
            "sparse": next(
                self.sparse_model.query_embed(query_text, parallel=parallel)
            ).as_object(),
            "colbert": next(self.colbert_model.query_embed(query_text, parallel=parallel)),
        }


class BGEM3DocumentEmbedder:
    """Generate dense, sparse, and ColBERT embeddings with BAAI/bge-m3."""

    def __init__(
        self,
        model_id: str = BGE_M3_MODEL_ID,
        use_fp16: bool = False,
        devices: str = EMBEDDING_DEVICE,
        batch_size: int = 12,
        max_length: int = 8192,
    ):
        self.model = FlagAutoModel.from_finetuned(
            model_id,
            use_fp16=use_fp16,
            devices=devices,
        )
        self.batch_size = batch_size
        self.max_length = max_length

    def vector_dimensions(self) -> Dict[str, Any]:
        return {
            "dense_size": self.model.model.config.hidden_size,
            "colbert_size": self.model.model.colbert_linear.out_features,
        }

    def sparse_modifier(self) -> models.Modifier | None:
        return None

    def embed_texts(self, texts: List[str], parallel: int = 0) -> Dict[str, List[Any]]:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )
        return {
            "dense": list(vectors["dense_vecs"]),
            "sparse": [
                {
                    "indices": [int(token_id) for token_id in sorted(weights, key=int)],
                    "values": [
                        float(weights[token_id])
                        for token_id in sorted(weights, key=int)
                    ],
                }
                for weights in vectors["lexical_weights"]
            ],
            "colbert": list(vectors["colbert_vecs"]),
        }

    def embed_query(self, query_text: str, parallel: int = 0) -> Dict[str, Any]:
        vectors = self.embed_texts([query_text], parallel=parallel)
        return {name: values[0] for name, values in vectors.items()}
