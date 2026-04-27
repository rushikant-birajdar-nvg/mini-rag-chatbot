from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import get_settings


class VectorStore:
    def __init__(self) -> None:
        settings = get_settings()
        self.collection = settings.qdrant_collection
        self.client = QdrantClient(url=settings.qdrant_url)

    def ensure_collection(self, vector_size: int) -> None:
        collections = self.client.get_collections().collections
        if any(c.name == self.collection for c in collections):
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    def upsert(self, texts: list[str], vectors: list[list[float]], metadatas: list[dict]) -> None:
        if not vectors:
            return
        self.ensure_collection(len(vectors[0]))
        points = [
            models.PointStruct(id=str(uuid4()), vector=vector, payload={"text": text, **metadata})
            for text, vector, metadata in zip(texts, vectors, metadatas, strict=True)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

