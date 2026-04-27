from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import get_settings
from app.models import RetrievedChunk


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

    def search(self, query_vector: list[float], department: str, level: int, limit: int) -> list[RetrievedChunk]:
        # RBAC at query layer:
        # (department == user.department OR department == hr) AND access_level <= user.level
        rbac_filter = models.Filter(
            must=[
                models.FieldCondition(key="access_level", range=models.Range(lte=level)),
                models.Filter(
                    should=[
                        models.FieldCondition(
                            key="department", match=models.MatchValue(value=department)
                        ),
                        models.FieldCondition(key="department", match=models.MatchValue(value="hr")),
                    ]
                ),
            ]
        )
        if hasattr(self.client, "search"):
            results = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                query_filter=rbac_filter,
                limit=limit,
            )
        else:
            response = self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                query_filter=rbac_filter,
                limit=limit,
            )
            results = response.points
        chunks = [
            RetrievedChunk(
                text=str(item.payload.get("text", "")),
                metadata=dict(item.payload),
                score=float(item.score),
            )
            for item in results
        ]

        threshold = get_settings().retrieval_score_threshold
        chunks = [c for c in chunks if c.score >= threshold]
        return chunks

