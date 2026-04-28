# README_SUBMISSION

## 1) How to run ingestion script

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Start Qdrant first (required for ingestion/retrieval)
docker run -p 6333:6333 qdrant/qdrant

# Then run ingestion
python3 ingest_docs.py
```

Alternative: start the API and call `POST /ingest`.
Make sure `QDRANT_URL` in `.env` matches the running Qdrant endpoint (default `http://localhost:6333`).

### OCR preflight (for scanned PDFs)

```bash
tesseract --version
```

If missing:

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y tesseract-ocr
```

Ingestion response includes `ocr_available`, `scanned_files_skipped`, and `warnings`.

## 2) How to run server

```bash
uvicorn app.main:app --reload
```

Provider selection is env-driven via `LLM_PROVIDER` (`ollama`, `groq`, `openai`, `gemini`, `anthropic`).

Retrieval mode is env-driven via `RETRIEVAL_MODE` (`dense` or `hybrid`).
If you switch from `dense` to `hybrid`, run ingestion again so sparse vectors are indexed in Qdrant.
If native hybrid query support is unavailable at runtime, the system safely falls back to dense retrieval.
Hybrid tuning knobs are defined in `app/config.py`: `retrieval_limit`, `hybrid_retrieval_limit`, `retrieval_score_threshold`, and `hybrid_retrieval_score_threshold` (and can still be overridden via env if needed). After changing retrieval mode/schema-related settings, recreate the collection and re-ingest (`curl -X DELETE "http://localhost:6333/collections/policy_chunks" && python3 ingest_docs.py`).

## 3) How to test (WebSocket exchange)

1. Set `JWT_SECRET` in `.env`.
2. Generate tokens: `python3 mint_tokens.py`.
3. Start Qdrant: `docker run -p 6333:6333 qdrant/qdrant`.
4. Run ingestion once: `python3 ingest_docs.py`.
5. Start server: `uvicorn app.main:app --reload`.
6. Open `test_client.html` and connect to `ws://localhost:8000/ws/chat`.
7. Send auth payload first:
   - `{"type":"auth","token":"<jwt>"}`
8. After `auth_success`, send:
   - `{"type":"message","text":"How many leave days do I get?"}`
9. Observe streamed responses:
   - `{"type":"stream","text":"..."}`
   - `{"type":"stream","text":"..."}`
   - `{"type":"done"}`

## 4) Chunking strategy

- Chunk size: `180` characters
- Overlap: `25` characters
- Splitter: recursive character splitting
- Rationale: these PDFs contain short policy clauses and entitlement statements, so smaller chunks improved precision for exact policy lookups while overlap preserved nearby context.

## 5) RBAC approach

RBAC is enforced directly in the vector DB query filter (Qdrant), not by filtering Python lists post-retrieval.

Filter logic:

- `access_level <= user.level`
- `department == user.department OR department == "hr"`

This keeps security checks close to retrieval and avoids accidental leakage in application logic.

## 6) Async/parallel approach

`get_employee_context(user_id)` runs three mock calls in parallel using `asyncio.gather(...)`:

- profile (`sleep(1)`)
- manager info (`sleep(1)`)
- team info (`sleep(1)`)

Total execution is ~1 second (parallel), not ~3 seconds (serial).

LLM/provider failures are surfaced as user-facing websocket error/fallback messages instead of crashing the chat flow.
Blocking work in async handlers is avoided by offloading sync-heavy steps (ingestion, embedding generation, vector search) via `asyncio.to_thread(...)`.

## 7) One thing I would improve with more time

With more time, I would move ACCESS_LEVEL_BY_FILE from local code to a DB-backed policy store and add full document lifecycle management (new/updated files, metadata-only changes, stale chunk cleanup) using version/hash tracking, so ingestion becomes incremental, auditable, and production-safe.

## 8) One tradeoff I made and why

I chose OpenAI as the default LLM provider for consistent response quality and low operational overhead, at the cost of external API dependency and usage costs.
I also evaluated Ollama for local inference, but due to inconsistent outputs and integration challenges in the RAG pipeline, I prioritized a more reliable hosted solution.

## Assumptions

- `ACCESS_LEVEL_BY_FILE` is intentionally kept as a local in-code mapping for this iteration (not DB-managed yet).
- Policy documents are ingested in batch mode, and access mapping changes are expected to be infrequent.
- Answers are grounded in retrieved policy text; the system does not integrate with live HR systems for personalized balances.
- OCR behavior depends on host setup (`tesseract` installed and available on PATH).

## Additional tradeoffs

- Keeping access-level mapping in code keeps behavior deterministic and simple for assignment scope, but is less flexible than runtime DB-managed policy updates.
- Hybrid retrieval improves recall/precision balance, but increases ingestion/query complexity and tuning overhead.
- Smaller chunk settings improve policy lookup precision, but can reduce long-span context within a single retrieved chunk.

## LLM and embedding choice

- Default LLM provider: `openai`
- Embeddings: `sentence-transformers` (`BAAI/bge-small-en-v1.5`)
- Provider layer is pluggable, so evaluators can switch by env vars without code changes.

## Stretch implemented

- Hybrid retrieval mode with Qdrant native fusion (`RETRIEVAL_MODE=hybrid`)
- Safe fallback to dense retrieval if hybrid query path is unavailable/fails

## Requirement-to-code map

- Ingestion pipeline: `ingest_docs.py`, `app/ingestion.py`, `app/pdf_loader.py`, `app/chunking.py`, `app/embeddings.py`, `app/vector_store.py`
- WebSocket protocol: `app/main.py`, `test_client.html`
- Auth/JWT expiry: `app/auth.py`, `app/main.py`, `mint_tokens.py`
- RBAC retrieval filters: `app/vector_store.py`
- Parallel tool call: `app/tools.py`
- LLM abstraction + tool usage: `app/llm.py`, `app/chat_service.py`
- Hybrid sparse support: `app/sparse_embeddings.py`, `app/vector_store.py`, `app/ingestion.py`

