from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import get_settings


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    settings = get_settings()
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    vectors = model.encode(texts, normalize_embeddings=True)
    return vectors.tolist()

