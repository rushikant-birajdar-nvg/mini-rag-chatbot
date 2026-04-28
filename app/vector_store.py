"""Qdrant data access layer for ingestion upserts and RBAC retrieval."""

from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import get_settings
from app.models import RetrievedChunk
from app.sparse_embeddings import make_sparse_vector


class VectorStore:
    """Wrap Qdrant operations used by the chat application."""

    def __init__(self) -> None:
        settings = get_settings()
        self.collection = settings.qdrant_collection
        self.client = QdrantClient(url=settings.qdrant_url)

    def ensure_collection(self, vector_size: int) -> None:
        """Create the collection if missing, with dense or hybrid schema."""
        settings = get_settings()
        collections = self.client.get_collections().collections
        if any(c.name == self.collection for c in collections):
            self._validate_existing_collection_schema()
            return
        if settings.retrieval_mode == "hybrid" and hasattr(models, "SparseVectorParams"):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    "dense": models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
                },
                sparse_vectors_config={"sparse": models.SparseVectorParams()},
            )
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    def _validate_existing_collection_schema(self) -> None:
        """Validate existing collection schema against current retrieval mode."""
        settings = get_settings()
        info = self.client.get_collection(self.collection)
        params = getattr(getattr(info, "config", None), "params", None)
        vectors_config = getattr(params, "vectors", None)
        sparse_vectors_config = getattr(params, "sparse_vectors", None)

        has_named_vectors = isinstance(vectors_config, dict)
        has_dense_named = has_named_vectors and "dense" in vectors_config
        has_sparse_named = isinstance(sparse_vectors_config, dict) and "sparse" in sparse_vectors_config

        if settings.retrieval_mode == "hybrid":
            if has_dense_named and has_sparse_named:
                return
            raise RuntimeError(
                "Qdrant collection schema mismatch for RETRIEVAL_MODE=hybrid. "
                f"Collection '{self.collection}' does not have named vectors 'dense' and 'sparse'. "
                "Delete and recreate the collection, then run ingestion again. "
                f"Example: curl -X DELETE \"{get_settings().qdrant_url}/collections/{self.collection}\""
            )

        if settings.retrieval_mode == "dense" and has_named_vectors:
            raise RuntimeError(
                "Qdrant collection schema mismatch for RETRIEVAL_MODE=dense. "
                f"Collection '{self.collection}' appears to use named vectors. "
                "Delete and recreate the collection, then run ingestion again. "
                f"Example: curl -X DELETE \"{get_settings().qdrant_url}/collections/{self.collection}\""
            )

    def upsert(
        self,
        texts: list[str],
        vectors: list[list[float]],
        metadatas: list[dict],
        sparse_vectors: list[dict[str, list[int] | list[float]]] | None = None,
    ) -> None:
        """Upsert chunk payloads with dense vectors and optional sparse vectors."""
        if not vectors:
            return
        settings = get_settings()
        self.ensure_collection(len(vectors[0]))
        points = []
        for i, (text, vector, metadata) in enumerate(zip(texts, vectors, metadatas, strict=True)):
            point_vector: dict | list[float]
            sparse = sparse_vectors[i] if sparse_vectors and i < len(sparse_vectors) else None
            if (
                settings.retrieval_mode == "hybrid"
                and hasattr(models, "SparseVector")
            ):
                point_vector = {"dense": vector}
                if (
                    sparse
                    and isinstance(sparse.get("indices"), list)
                    and isinstance(sparse.get("values"), list)
                    and sparse["indices"]
                ):
                    point_vector["sparse"] = models.SparseVector(
                        indices=sparse["indices"], values=sparse["values"]
                    )
            else:
                point_vector = vector
            points.append(
                models.PointStruct(id=str(uuid4()), vector=point_vector, payload={"text": text, **metadata})
            )
        self.client.upsert(collection_name=self.collection, points=points)

    def search(
        self, query_vector: list[float], query_text: str, department: str, level: int, limit: int
    ) -> list[RetrievedChunk]:
        """Run RBAC-filtered retrieval and return thresholded chunk results."""
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
        results = self._hybrid_search_or_fallback(query_vector, query_text, rbac_filter, limit)
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

    def _hybrid_search_or_fallback(
        self, query_vector: list[float], query_text: str, query_filter: models.Filter, limit: int
    ):
        """Try native hybrid search first and fall back to dense when needed."""
        settings = get_settings()
        if settings.retrieval_mode != "hybrid":
            return self._dense_search(query_vector, query_filter, limit)

        if not (
            hasattr(self.client, "query_points")
            and hasattr(models, "Prefetch")
            and hasattr(models, "FusionQuery")
            and hasattr(models, "SparseVector")
        ):
            return self._dense_search(query_vector, query_filter, limit)

        sparse_query = make_sparse_vector(query_text)
        if not sparse_query["indices"]:
            return self._dense_search(query_vector, query_filter, limit)

        try:
            response = self.client.query_points(
                collection_name=self.collection,
                prefetch=[
                    models.Prefetch(
                        query=query_vector,
                        using="dense",
                        limit=max(limit * 4, limit),
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_query["indices"], values=sparse_query["values"]
                        ),
                        using="sparse",
                        limit=max(limit * 4, limit),
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=query_filter,
                limit=limit,
            )
            return response.points
        except Exception:
            return self._dense_search(query_vector, query_filter, limit)

    def _dense_search(self, query_vector: list[float], query_filter: models.Filter, limit: int):
        """Run standard dense vector search across supported Qdrant client APIs."""
        if hasattr(self.client, "search"):
            return self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
            )
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
        )
        return response.points

