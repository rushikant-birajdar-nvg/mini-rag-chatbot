from fastapi.testclient import TestClient

import app.main as main_module


def test_ingest_endpoint_success(monkeypatch) -> None:
    monkeypatch.setattr(
        main_module,
        "ingest_documents",
        lambda _docs: {"documents_ingested": 1, "chunks_upserted": 2},
    )
    client = TestClient(main_module.app)
    response = client.post("/ingest")
    assert response.status_code == 200
    assert response.json()["documents_ingested"] == 1


def test_ingest_endpoint_internal_error(monkeypatch) -> None:
    def _raise(_docs):
        raise RuntimeError("boom")

    monkeypatch.setattr(main_module, "ingest_documents", _raise)
    client = TestClient(main_module.app)
    response = client.post("/ingest")
    assert response.status_code == 500
    assert response.json() == {"message": "Ingestion failed due to an internal error."}
