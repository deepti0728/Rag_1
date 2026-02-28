import csv
from pathlib import Path

from chunker import chunk_documents
from loader import load_pdfs


def export_chunks_to_csv(data_dir="data", output_file="chunks.csv"):
    documents = load_pdfs(data_dir)
    chunks = chunk_documents(documents)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["chunk_id", "source", "page", "chunk_length", "chunk_text"])

        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", "")
            text = chunk.page_content
            writer.writerow([i, source, page, len(text), text])

    print(f"Exported {len(chunks)} chunks to {output_path.resolve()}")


if __name__ == "__main__":
    export_chunks_to_csv(data_dir="data", output_file="data/chunks.csv")
