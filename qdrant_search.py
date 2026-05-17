import json
from typing import List, Dict, Any, Optional

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser.text.semantic_double_merging_splitter import (
    SemanticDoubleMergingSplitterNodeParser,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient, models
from transformers import AutoTokenizer

from qdrant_embedder import DocumentEmbedder, BGEM3DocumentEmbedder, JinaDocumentEmbedder
from qdrant_config import (
    DATASET_CONFIGS,
    DEFAULT_FILTER,
    DENSE_MODEL_ID,
    EMBEDDING_DEVICE,
    chunking_config,
    hybrid_vector_configs,
)


class TextChunker:
    """Reusable text chunker for fixed, sentence, and semantic chunking."""

    def __init__(
        self,
        model_name: str = DENSE_MODEL_ID,
        chunk_size: int = 180,  # stays below 256-token model limit
        chunk_overlap: int = 30,  # preserves cross-chunk context
        semantic_max_chunk_size: int = 800,  # ~200 tokens by char heuristic
        embedding_device: str = EMBEDDING_DEVICE,
    ):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.semantic_max_chunk_size = semantic_max_chunk_size
        self.embedding_device = embedding_device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sentence_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=self.tokenizer.tokenize,
        )
        self.semantic_splitter = SemanticDoubleMergingSplitterNodeParser(
            initial_threshold=0.4,
            appending_threshold=0.5,
            merging_threshold=0.5,
            max_chunk_size=semantic_max_chunk_size,
            embed_model=HuggingFaceEmbedding(
                model_name=model_name,
                device=self.embedding_device,
            ),
        )

    def fixed_size_chunks(self, text: str) -> List[str]:
        """Fixed-size chunking: splits at exact token boundaries."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return [
            self.tokenizer.decode(
                tokens[i:i + self.chunk_size],
                skip_special_tokens=True,
            )
            for i in range(0, len(tokens), self.chunk_size)
        ]

    def sentence_chunks(self, text: str) -> List[str]:
        """Sentence-aware chunking: respects sentence boundaries."""
        return self.sentence_splitter.split_text(text)

    def semantic_chunks(self, text: str) -> List[str]:
        """Semantic chunking: uses embedding similarity to find natural breaks."""
        nodes = self.semantic_splitter.get_nodes_from_documents([Document(text=text)])
        return [node.get_content() for node in nodes]

    def chunk_text(self, text: str, strategy: str = "sentence") -> List[str]:
        if strategy == "fixed":
            return self.fixed_size_chunks(text)

        if strategy == "sentence":
            return self.sentence_chunks(text)

        if strategy == "semantic":
            return self.semantic_chunks(text)

        raise ValueError("strategy must be one of: fixed, sentence, semantic")


class QdrantSearchManager:
    def __init__(
        self,
        collection_name: str,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        chunker: Optional[TextChunker] = None,
        embedder: Optional[Any] = None,
        chunk_strategy: str = "sentence",
        chunk_size: int = 180,  # stays below 256-token model limit
        chunk_overlap: int = 30,  # preserves cross-chunk context
        semantic_max_chunk_size: int = 800,  # ~200 tokens by char heuristic
        embedding_device: str = EMBEDDING_DEVICE,
        embedding_model_name: str = DENSE_MODEL_ID,
    ):
        self.client = QdrantClient(url=url, api_key=api_key)
        self._verify_connection(url)

        self.collection_name = collection_name
        self.chunk_strategy = chunk_strategy
        self.embedder = embedder or DocumentEmbedder()

        collection_configs = hybrid_vector_configs(
            **self.embedder.vector_dimensions(),
            sparse_modifier=self.embedder.sparse_modifier(),
        )
        self.vector_configs = collection_configs["vector_configs"]
        self.sparse_configs = collection_configs["sparse_configs"]
        self.chunker = chunker or TextChunker(
            model_name=embedding_model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            semantic_max_chunk_size=semantic_max_chunk_size,
            embedding_device=embedding_device,
        )

    def _verify_connection(self, url: str) -> None:
        """Fail fast when Qdrant is unreachable or credentials are invalid."""
        try:
            self.client.get_collections()
        except Exception as exc:
            raise ConnectionError(
                f"Could not connect to Qdrant at '{url}'. "
                "Start Qdrant or pass a valid url/api_key."
            ) from exc

    def _create_payload_indexes(self, payload_indexes: Dict[str, Any]) -> None:
        """Create payload indexes that are missing from the collection."""
        collection = self.client.get_collection(collection_name=self.collection_name)
        existing_indexes = getattr(collection, "payload_schema", {}) or {}

        for field_name, schema in payload_indexes.items():
            if field_name in existing_indexes:
                continue

            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=schema
            )
            print(f"✓ Payload index ready for '{field_name}'.")

    def create_collection(
        self,
        payload_indexes: Optional[Dict[str, Any]] = None,
        on_disk: bool = True,
        recreate: bool = True,
    ):
        """Recreate a collection with production optimizations and optional payload indexes."""
        if self.client.collection_exists(self.collection_name):
            if not recreate:
                print(f"Keeping existing collection '{self.collection_name}'.")
                if payload_indexes:
                    self._create_payload_indexes(payload_indexes)
                return

            self.client.delete_collection(collection_name=self.collection_name)
            print(f"Deleted existing collection '{self.collection_name}'.")

        # Apply on_disk to all vector configs for production scalability
        for config in self.vector_configs.values():
            if hasattr(config, 'hnsw_config') and config.hnsw_config is None:
                config.hnsw_config = models.HnswConfigDiff(on_disk=on_disk)
            elif not hasattr(config, 'hnsw_config'):
                # For models.VectorParams
                config.hnsw_config = models.HnswConfigDiff(on_disk=on_disk)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=self.vector_configs,
            sparse_vectors_config=self.sparse_configs,
            # Enable quantization for better performance/memory trade-off
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True
                )
            ) if on_disk else None
        )

        if payload_indexes:
            self._create_payload_indexes(payload_indexes)
        print(f"✓ Production-ready collection '{self.collection_name}' initialized.")

    def get_collection(self):
        """Fetch collection metadata and vector configuration."""
        return self.client.get_collection(collection_name=self.collection_name)

    def chunk_documents(
        self,
        documents: List[Dict[str, Any]],
        text_key: str = "chunk_text",
        strategy: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Chunk document payloads while preserving metadata."""
        selected_strategy = strategy or self.chunk_strategy
        chunked_documents = []

        for doc_index, doc in enumerate(documents):
            text = doc.get(text_key, "")
            chunks = self.chunker.chunk_text(text, strategy=selected_strategy)

            for chunk_index, chunk in enumerate(chunks):
                payload = dict(doc)
                payload[text_key] = chunk
                payload["source_doc_id"] = doc.get("id", doc_index)
                payload["chunk_index"] = chunk_index
                payload["chunking_strategy"] = selected_strategy
                chunked_documents.append(payload)

        return chunked_documents

    def embed_documents(
        self,
        documents: List[Dict[str, Any]],
        text_key: str = "chunk_text",
        parallel: int = 0,
    ) -> Dict[str, List[Any]]:
        """Embed document payloads while preserving order across vector types."""
        texts = [doc.get(text_key, "") for doc in documents]
        return self.embedder.embed_texts(texts, parallel=parallel)

    def embed_and_upsert_documents(
        self,
        documents: List[Dict[str, Any]],
        text_key: str = "chunk_text",
        parallel: int = 0,
        batch_size: int = 128,
        upload_batch_size: int = 64,
        start_id: int = 0,
    ) -> int:
        """Embed each document, create points, and upload in small batches."""
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if upload_batch_size <= 0:
            raise ValueError("upload_batch_size must be a positive integer")

        uploaded_count = 0
        total_documents = len(documents)

        for start in range(0, total_documents, batch_size):
            batch_documents = documents[start:start + batch_size]
            print(f"Processing chunks {start + 1}-{start + len(batch_documents)}/{total_documents}...")
            texts = [document.get(text_key, "") for document in batch_documents]
            embeddings = self.embedder.embed_texts(texts, parallel=parallel)
            points = []

            for batch_index, document in enumerate(batch_documents):
                index = start + batch_index
                dense_vector = embeddings["dense"][batch_index]
                sparse_vector = embeddings["sparse"][batch_index]
                colbert_vector = embeddings["colbert"][batch_index]

                if hasattr(sparse_vector, "as_object"):
                    sparse_vector = sparse_vector.as_object()

                points.append(
                    models.PointStruct(
                        id=start_id + index,
                        payload=document,
                        vector={
                            "dense": dense_vector,
                            "sparse": sparse_vector,
                            "colbert": colbert_vector,
                        },
                    )
                )

            self.upsert_data(points, batch_size=upload_batch_size)
            uploaded_count += len(points)

        return uploaded_count

    def upsert_data(self, points: List[models.PointStruct], batch_size: int = 100):
        """Standard batch upload for any point structure."""
        self.client.upload_points(
            collection_name=self.collection_name,
            points=points,
            batch_size=batch_size
        )
        print(f"✓ Uploaded {len(points)} points.")

    def hybrid_search(
        self,
        query_text: str,
        query_filter: Optional[models.Filter] = DEFAULT_FILTER,
        candidate_limit: int = 100,
        limit: int = 10,
        fusion: models.Fusion = models.Fusion.RRF,
        parallel: int = 0,
        hnsw_ef: Optional[int] = None,
    ):
        """Run dense+sparse fusion, then ColBERT reranking for a text query."""
        query_embeddings = self.embedder.embed_query(query_text, parallel=parallel)
        dense_query = query_embeddings["dense"]
        sparse_query = query_embeddings["sparse"]
        colbert_query = query_embeddings["colbert"]
        search_params = models.SearchParams(hnsw_ef=hnsw_ef) if hnsw_ef else None

        root_query = models.Prefetch(
            prefetch=[
                models.Prefetch(
                    query=dense_query,
                    using="dense",
                    filter=query_filter,
                    params=search_params,
                    limit=candidate_limit,
                ),
                models.Prefetch(
                    query=sparse_query,
                    using="sparse",
                    filter=query_filter,
                    params=search_params,
                    limit=candidate_limit,
                ),
            ],
            query=models.FusionQuery(fusion=fusion),
            limit=candidate_limit,
        )

        return self.client.query_points(
            collection_name=self.collection_name,
            prefetch=root_query,
            query=colbert_query,
            using="colbert",
            query_filter=query_filter,
            search_params=search_params,
            limit=limit,
            with_payload=True
        )


def setup_collection(
    collection_name: str,
    dataset_name: str,
    url: str = "http://localhost:6333",
    api_key: Optional[str] = None,
    embedder: Optional[Any] = None,
    chunk_config: Optional[Dict[str, Any]] = None,
    on_disk: bool = False,
    recreate: bool = False,
) -> tuple[QdrantSearchManager, Dict[str, Any]]:
    """Create or reuse a collection for a configured dataset."""
    # Validate the dataset name before creating any external resources.
    if dataset_name not in DATASET_CONFIGS:
        known_datasets = ", ".join(DATASET_CONFIGS)
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose one of: {known_datasets}")

    chunk_config = chunk_config or chunking_config("all_minilm")

    # Create the manager with embedding-derived default vector configs.
    manager = QdrantSearchManager(
        collection_name=collection_name,
        url=url,
        api_key=api_key,
        embedder=embedder,
        chunk_strategy=chunk_config["chunk_strategy"],
        chunk_size=chunk_config["chunk_size"],
        chunk_overlap=chunk_config["chunk_overlap"],
        semantic_max_chunk_size=chunk_config["semantic_max_chunk_size"],
        embedding_device=chunk_config["embedding_device"],
        embedding_model_name=chunk_config["embedding_model_name"],
    )

    # Resolve dataset-specific file and payload index settings.
    dataset_config = DATASET_CONFIGS[dataset_name]
    print(
        f"Using dataset file: {dataset_config['path']} "
        f"({dataset_config['num_entries']} entries)"
    )

    # Create or reuse the Qdrant collection with the dataset's payload indexes.
    manager.create_collection(
        payload_indexes=dataset_config["payload_indexes"],
        on_disk=on_disk,
        recreate=recreate,
    )

    # Print a compact readiness summary from Qdrant metadata.
    collection = manager.get_collection()
    status = getattr(collection.status, "value", collection.status)
    print(
        f"Collection '{manager.collection_name}' ready "
        f"(status={status}, points={collection.points_count}, "
        f"segments={collection.segments_count})."
    )

    return manager, dataset_config


def upload_dataset(
    manager: QdrantSearchManager,
    dataset_config: Dict[str, Any],
    text_key: str = "chunk_text",
    batch_size: int = 128,
    upload_batch_size: int = 32,
    parallel: int = 0,
) -> int:
    """Load, chunk, embed, and upload a configured dataset."""
    with open(dataset_config["path"], "r", encoding="utf-8") as dataset_file:
        documents = json.load(dataset_file)

    chunked_documents = manager.chunk_documents(documents, text_key=text_key)
    print(
        f"Chunked {len(documents)} documents into {len(chunked_documents)} chunks "
        f"using '{manager.chunk_strategy}' chunking."
    )

    uploaded_count = manager.embed_and_upsert_documents(
        chunked_documents,
        text_key=text_key,
        parallel=parallel,
        batch_size=batch_size,
        upload_batch_size=upload_batch_size,
    )
    print(f"✓ Uploaded {uploaded_count} embedded chunks to Qdrant.")
    return uploaded_count


def print_results(response: Any) -> None:
    print(f"✓ Found {len(response.points)} results")
    for rank, point in enumerate(response.points, start=1):
        payload = point.payload
        print(f"\n{rank}. {payload['page_title']}")
        print(f"   Section: {payload['section_title']}")
        print(f"   Score: {point.score:.4f}")
        print(f"   Section URL: {payload['section_url']}")
        print(f"   Page URL: {payload['page_url']}")
        print(f"   Match: {' '.join(payload['chunk_text'].split())}")


if __name__ == "__main__":
    manager, dataset_config = setup_collection(
        collection_name="docs_search_jina_v5",
        dataset_name="qdrant_docs_10k",
        on_disk=False,
        recreate=False,
        embedder=JinaDocumentEmbedder(),
        chunk_config=chunking_config("jina_v5"),
    )

    upload_dataset(
        manager=manager,
        dataset_config=dataset_config,
        batch_size=4,
        upload_batch_size=4,
    )

    query_text = "How do I tune Qdrant indexing for better recall?"
    print(f"\nQuery: {query_text}")

    response = manager.hybrid_search(
        query_text=query_text,
        candidate_limit=100,
        limit=10,
    )
    print_results(response)
