from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [chunk.strip() for chunk in splitter.split_text(cleaned) if chunk.strip()]

