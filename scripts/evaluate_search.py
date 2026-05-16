import json
import sys
import time
from pathlib import Path
from statistics import median, quantiles
from typing import Any, Dict, List, Optional

from qdrant_client import models


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_search import QdrantSearchManager, setup_collection
from qdrant_embedder import BGEM3DocumentEmbedder, DocumentEmbedder
from qdrant_config import chunking_config


def embedding_setup(name: str) -> Dict[str, Any]:
    if name == "all_minilm":
        return {
            "collection_name": "docs_search",
            "embedder": DocumentEmbedder(),
            "chunk_config": chunking_config("all_minilm"),
        }

    if name == "bge_m3":
        return {
            "collection_name": "docs_search_2",
            "embedder": BGEM3DocumentEmbedder(use_fp16=False),
            "chunk_config": chunking_config("bge_m3"),
        }

    raise ValueError("name must be one of: all_minilm, bge_m3")


def load_queries(eval_file: Path) -> List[Dict[str, Any]]:
    with open(eval_file, "r", encoding="utf-8") as file:
        return json.load(file)


def matches_expected(payload: Dict[str, Any], expected_urls: List[str]) -> bool:
    section_url = payload["section_url"]
    page_url = payload["page_url"]

    for expected_url in expected_urls:
        if expected_url in (section_url, page_url):
            return True
        if "#" in expected_url and section_url.endswith("#" + expected_url.split("#", 1)[1]):
            return True

    return False


def latency_percentiles(latencies: List[float]) -> Dict[str, float]:
    if not latencies:
        return {"p50": 0.0, "p95": 0.0}
    if len(latencies) == 1:
        return {"p50": latencies[0], "p95": latencies[0]}

    return {
        "p50": median(latencies),
        "p95": quantiles(latencies, n=20, method="inclusive")[18],
    }


def evaluate_search(
    manager: QdrantSearchManager,
    eval_queries: List[Dict[str, Any]],
    candidate_limit: int = 100,
    limit: int = 10,
    fusion: models.Fusion = models.Fusion.RRF,
    hnsw_ef: Optional[int] = None,
) -> Dict[str, Any]:
    collection = manager.get_collection()
    if collection.points_count == 0:
        raise RuntimeError(
            f"Collection '{manager.collection_name}' has no points. "
            "Upload the dataset before running evaluation."
        )

    rows = []

    print(
        f"\nEvaluation: queries={len(eval_queries)}, limit={limit}, "
        f"candidate_limit={candidate_limit}, fusion={fusion.value}, hnsw_ef={hnsw_ef}"
    )

    for example in eval_queries:

        # get latency and search results
        start = time.perf_counter()
        response = manager.hybrid_search(
            query_text=example["query"],
            candidate_limit=candidate_limit,
            limit=limit,
            fusion=fusion,
            hnsw_ef=hnsw_ef,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        # get hit rank by matching ground truth 
        hit_rank = 0
        for rank, point in enumerate(response.points, start=1):
            if matches_expected(point.payload, example["expected_urls"]):
                hit_rank = rank
                break

        rows.append(
            {
                "query": example["query"],
                "query_type": example["query_type"],
                "hit_rank": hit_rank,
                "latency_ms": latency_ms,
            }
        )
        status = f"rank={hit_rank}" if hit_rank else "miss"
        print(f"  {status:>7} {latency_ms:8.1f} ms  {example['query']}")

    recall = sum(1 for row in rows if row["hit_rank"]) / len(rows)
    mrr = sum(1 / row["hit_rank"] for row in rows if row["hit_rank"]) / len(rows)
    latency = latency_percentiles([row["latency_ms"] for row in rows])

    metrics = {
        f"recall@{limit}": recall,
        f"mrr@{limit}": mrr,
        "latency_p50_ms": latency["p50"],
        "latency_p95_ms": latency["p95"],
        "rows": rows,
    }

    print(
        f"\nSummary: recall@{limit}={recall:.3f}, mrr@{limit}={mrr:.3f}, "
        f"p50={metrics['latency_p50_ms']:.1f} ms, "
        f"p95={metrics['latency_p95_ms']:.1f} ms"
    )

    return metrics


if __name__ == "__main__":
    embedding_name = "bge_m3"
    config = embedding_setup(embedding_name)
    dataset_name = "qdrant_docs_10k"
    eval_file = PROJECT_ROOT / "datasets" / f"{dataset_name}_eval.json"
    eval_queries = load_queries(eval_file)

    manager, _ = setup_collection(
        collection_name=config["collection_name"],
        dataset_name=dataset_name,
        embedder=config["embedder"],
        chunk_config=config["chunk_config"],
        on_disk=False,
        recreate=False,
    )
    evaluate_search(
        manager=manager,
        eval_queries=eval_queries,
        candidate_limit=100,
        limit=10,
        fusion=models.Fusion.RRF,
    )
