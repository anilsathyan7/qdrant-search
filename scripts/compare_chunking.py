import sys
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_search import TextChunker
from qdrant_config import chunking_config


def chunk_text(data_file: Path, chunker: TextChunker) -> None:
    with open(data_file, "r") as f:
        test_doc = json.load(f)

    text = test_doc[0]["chunk_text"]
    print("=" * 80)
    print(f"BEFORE CHUNKING ({len(text)} characters)")
    print("=" * 80)
    print(text)

    print("\n" + "=" * 80)
    print("AFTER CHUNKING")
    print("=" * 80)
    for strategy in ["fixed", "sentence", "semantic"]:
        chunks = chunker.chunk_text(text, strategy=strategy)

        print(f"\n[{strategy.upper()}] {len(chunks)} chunks")
        print("-" * 80)
        for idx, chunk in enumerate(chunks):
            token_count = len(chunker.tokenizer.tokenize(chunk))
            print(f"\nChunk {idx + 1} | tokens: {token_count}")
            print(chunk)


if __name__ == "__main__":
    embedding_name = "bge_m3"
    config = chunking_config(embedding_name)
    chunker = TextChunker(
        model_name=config["embedding_model_name"],
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        semantic_max_chunk_size=config["semantic_max_chunk_size"],
        embedding_device=config["embedding_device"],
    )
    dataset = PROJECT_ROOT / "datasets" / "qdrant_docs.json"
    chunk_text(dataset, chunker)
