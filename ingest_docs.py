from pathlib import Path

from app.ingestion import ingest_documents


def main() -> None:
    result = ingest_documents(Path("docs"))
    print("Ingestion complete:")
    print(result)


if __name__ == "__main__":
    main()

