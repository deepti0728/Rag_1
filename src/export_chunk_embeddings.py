import csv
import json
from pathlib import Path

from embeddings import get_embeddings


def export_embeddings(chunks_csv_path="data/chunks.csv", output_csv_path="data/chunks_embeddings.csv"):
    chunks_path = Path(chunks_csv_path)
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks CSV not found: {chunks_path}")

    with chunks_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        raise ValueError("No rows found in chunks CSV.")

    texts = [row.get("chunk_text", "") for row in rows]
    embedder = get_embeddings()
    vectors = embedder.embed_documents(texts)

    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = ["chunk_id", "source", "page", "chunk_length", "chunk_text", "embedding"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row, vector in zip(rows, vectors):
            writer.writerow(
                {
                    "chunk_id": row.get("chunk_id", ""),
                    "source": row.get("source", ""),
                    "page": row.get("page", ""),
                    "chunk_length": row.get("chunk_length", ""),
                    "chunk_text": row.get("chunk_text", ""),
                    "embedding": json.dumps(vector, ensure_ascii=True),
                }
            )

    print(f"Exported {len(rows)} embeddings to {output_path.resolve()}")


if __name__ == "__main__":
    export_embeddings()
