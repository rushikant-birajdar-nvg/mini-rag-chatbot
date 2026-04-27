"""Chat orchestration for retrieval, prompt building, and LLM streaming."""

import asyncio
import logging
from collections.abc import AsyncIterator

import httpx

from app.config import get_settings
from app.embeddings import embed_texts
from app.llm import FallbackLLM, get_llm_client
from app.models import AuthenticatedUser, RetrievedChunk
from app.tools import get_employee_context
from app.vector_store import VectorStore

logger = logging.getLogger(__name__)


class ChatService:
    """Coordinates RAG retrieval and response streaming for chat requests."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.vector_store = VectorStore()
        self.llm = get_llm_client()

    async def stream_response(self, user: AuthenticatedUser, text: str) -> AsyncIterator[str]:
        """Stream a policy-grounded answer for one user question."""
        query_vector = (await asyncio.to_thread(embed_texts, [text]))[0]
        chunks = await asyncio.to_thread(
            self.vector_store.search,
            query_vector,
            text,
            user.department,
            user.level,
            self.settings.retrieval_limit,
        )
        if not chunks:
            yield "I could not find any policy content you are allowed to access yet. Run ingestion first."
            return

        context_lines = []
        for idx, chunk in enumerate(chunks, start=1):
            source = chunk.metadata.get("source_file", "unknown")
            page = chunk.metadata.get("page", "?")
            context_lines.append(f"[{idx}] {source} page {page}: {chunk.text}")

        prompt = (
            "You are a company policy assistant. "
            "Answer questions strictly using the retrieved policy context below. "
            "Do not use outside knowledge.\n\n"
            
            "Rules:\n"
            "- Answer strictly from the retrieved policy context below.\n"
            "- Questions about leave, expenses, or any entitlement — even phrased as 'how many X do I have' "
            "— are policy questions. Answer them from context.\n"
            "- If the answer is genuinely not in the context, say: 'I don't have enough information to answer that.'\n"
            "- Never show tool call syntax or internal steps in your response. Answer directly.\n"
            "- Never mention that certain documents exist but are restricted.\n"
            "- Cite the source document naturally (e.g. 'According to the leave policy...').\n\n"

            "Tool usage:\n"
            "- Call `get_employee_context(user_id)` ONLY when the user explicitly asks about their "
            "manager name, grade, or team.\n\n"
            
            f"User context: department={user.department}, level={user.level}, user_id={user.user_id}\n\n"
            
            "Retrieved policy context:\n"
            + "\n".join(context_lines)
            + f"\n\nQuestion: {text}"
        )
        tool_executor = self._execute_tool #if enable_tool_calls else None

        try:
            async for token in self.llm.stream_answer(
                prompt=prompt,
                user_id=user.user_id,
                tool_executor=tool_executor,
            ):
                yield token
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                logger.warning("LLM rate limit hit for provider call: %s", exc)
                yield "The selected LLM provider is rate-limited right now (HTTP 429). Please retry in a minute."
                return
            if exc.response.status_code in (401, 403):
                logger.warning("LLM authentication/permission error from provider: %s", exc)
                yield (
                    "The selected LLM provider rejected the request (HTTP "
                    f"{exc.response.status_code}). Please verify the API key, model access, and provider settings."
                )
                return
            logger.exception("HTTP error from LLM provider; using fallback response")
            fallback = FallbackLLM()
            async for token in fallback.stream_answer(prompt, user.user_id, tool_executor):
                yield token
        except Exception:
            logger.exception("Primary LLM call failed; using fallback response")
            fallback = FallbackLLM()
            async for token in fallback.stream_answer(prompt, user.user_id, tool_executor):
                yield token

    async def _execute_tool(self, tool_name: str, args: dict) -> dict:
        """Execute supported tools requested by the LLM."""
        if tool_name != "get_employee_context":
            return {"error": f"Unsupported tool: {tool_name}"}
        user_id = str(args.get("user_id", "")).strip()
        if not user_id:
            return {"error": "Missing required argument: user_id"}
        return await get_employee_context(user_id)

