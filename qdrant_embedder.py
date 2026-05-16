from typing import Any, Dict, List

from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
from FlagEmbedding import FlagAutoModel

from qdrant_search_config import (
    BGE_M3_MODEL_ID,
    COLBERT_MODEL_ID,
    DENSE_MODEL_ID,
    SPARSE_MODEL_ID,
)


class DocumentEmbedder:
    """Generate dense, sparse, and ColBERT embeddings with FastEmbed models."""

    def __init__(
        self,
        dense_model_id: str = DENSE_MODEL_ID,
        sparse_model_id: str = SPARSE_MODEL_ID,
        colbert_model_id: str = COLBERT_MODEL_ID,
    ):
        self.dense_model = TextEmbedding(dense_model_id)
        self.sparse_model = SparseTextEmbedding(sparse_model_id)
        self.colbert_model = LateInteractionTextEmbedding(colbert_model_id)

    def vector_dimensions(self) -> Dict[str, int]:
        return {
            "dense_size": self.dense_model.embedding_size,
            "colbert_size": self.colbert_model.embedding_size,
        }

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


class BGEM3DocumentEmbedder:
    """Generate dense, sparse, and ColBERT embeddings with BAAI/bge-m3."""

    def __init__(
        self,
        model_id: str = BGE_M3_MODEL_ID,
        use_fp16: bool = False,
        devices: str = "cpu",
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

    def vector_dimensions(self) -> Dict[str, int]:
        return {
            "dense_size": self.model.model.config.hidden_size,
            "colbert_size": self.model.model.colbert_linear.out_features,
        }

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
