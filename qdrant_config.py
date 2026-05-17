from pathlib import Path
from typing import Any, Dict, Optional

from qdrant_client import models

DENSE_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL_ID = "Qdrant/bm25"  # or SPLADE if you want heavier sparse
JINA_V5_DENSE_MODEL_ID = "jinaai/jina-embeddings-v5-text-small-retrieval"
MINICOIL_SPARSE_MODEL_ID = "Qdrant/minicoil-v1"
COLBERT_MODEL_ID = "colbert-ir/colbertv2.0"
BGE_M3_MODEL_ID = "BAAI/bge-m3"
EMBEDDING_DEVICE = "cpu"
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "datasets"
DEFAULT_FILTER = models.Filter(
    must=[
        models.FieldCondition(
            key="tags",
            match=models.MatchValue(value="python"),
        )
    ]
)


def docs_payload_indexes() -> Dict[str, models.PayloadSchemaType]:
    return {
        "page_url": models.PayloadSchemaType.KEYWORD,
        "section_url": models.PayloadSchemaType.KEYWORD,
        "tags": models.PayloadSchemaType.KEYWORD,
        "breadcrumbs": models.PayloadSchemaType.KEYWORD,
    }


DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "qdrant_docs": {
        "path": DATASET_ROOT / "qdrant_docs.json",
        "num_entries": 12,
        "payload_indexes": docs_payload_indexes(),
    },
    "qdrant_docs_10k": {
        "path": DATASET_ROOT / "qdrant_docs_10k.json",
        "num_entries": 1800,
        "payload_indexes": docs_payload_indexes(),
    },
    "qdrant_docs_100k": {
        "path": DATASET_ROOT / "qdrant_docs_100k.json",
        "num_entries": 18000,
        "payload_indexes": docs_payload_indexes(),
    },
}


def chunking_config(name: str = "all_minilm") -> Dict[str, Any]:
    configs = {
        "all_minilm": {
            "chunk_strategy": "sentence",
            "chunk_size": 180,
            "chunk_overlap": 30,
            "semantic_max_chunk_size": 800,
            "embedding_model_name": DENSE_MODEL_ID,
            "embedding_device": EMBEDDING_DEVICE,
        },
        "bge_m3": {
            "chunk_strategy": "sentence",
            "chunk_size": 512,
            "chunk_overlap": 80,
            "semantic_max_chunk_size": 2000,
            "embedding_model_name": BGE_M3_MODEL_ID,
            "embedding_device": EMBEDDING_DEVICE,
        },
        "jina_v5": {
            "chunk_strategy": "sentence",
            "chunk_size": 1024,
            "chunk_overlap": 128,
            "semantic_max_chunk_size": 2000,
            "embedding_model_name": JINA_V5_DENSE_MODEL_ID,
            "embedding_device": EMBEDDING_DEVICE,
        },
    }
    return configs[name]


def hybrid_vector_configs(
    dense_size: int,
    colbert_size: int,
    sparse_modifier: Optional[models.Modifier] = None,
    dense_distance: models.Distance = models.Distance.COSINE,
    colbert_distance: models.Distance = models.Distance.COSINE,
) -> Dict[str, Any]:
    return {
        "vector_configs": {
            "dense": models.VectorParams(
                size=dense_size,
                distance=dense_distance,
            ),
            "colbert": models.VectorParams(
                size=colbert_size,
                distance=colbert_distance,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
                hnsw_config=models.HnswConfigDiff(m=0),
            ),
        },
        "sparse_configs": {
            "sparse": models.SparseVectorParams(modifier=sparse_modifier),
        },
    }
