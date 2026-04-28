"""Chat orchestration for retrieval, prompt building, and LLM streaming."""

import asyncio
import logging
import re
from collections.abc import AsyncIterator

import httpx

from app.config import get_settings
from app.embeddings import embed_texts
from app.llm import FallbackLLM, get_llm_client
from app.models import AuthenticatedUser, RetrievedChunk
from app.tools import get_employee_context
from app.vector_store import VectorStore

logger = logging.getLogger(__name__)
QUESTION_START_RE = (
    r"(?:what|how|who|when|where|why|which|can|could|do|does|did|is|are|should|will|would)"
)


class ChatService:
    """Coordinates RAG retrieval and response streaming for chat requests."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.vector_store = VectorStore()
        self.llm = get_llm_client()

    async def stream_response(self, user: AuthenticatedUser, text: str) -> AsyncIterator[str]:
        """Stream a policy-grounded answer for one user question."""
        questions = self._split_questions(text)
        chunks = await self._retrieve_chunks(user, questions)
        if not chunks:
            yield "I could not find any policy content you are allowed to access yet. Run ingestion first."
            return

        context_lines = []
        for idx, chunk in enumerate(chunks, start=1):
            source = chunk.metadata.get("source_file", "unknown")
            page = chunk.metadata.get("page", "?")
            context_lines.append(f"[{idx}] {source} page {page}: {chunk.text}")

        user_question_block = (
            f"Question: {text}"
            if len(questions) == 1
            else "Questions:\n"
            + "\n".join(f"{idx}. {q}" for idx, q in enumerate(questions, start=1))
            + "\n\nAnswer each question separately in the same order."
        )

        prompt = (
            "You are a company policy assistant. "
            "Answer questions strictly using the retrieved policy context below. "
            "Do not use outside knowledge.\n\n"
            
            "Rules:\n"
            "- Carefully read ALL retrieved context before answering.\n"
            "- Combine information from multiple context chunks if needed.\n"
            "- If the question contains multiple parts, answer ALL parts.\n"
            "- Questions about leave, expenses, or any entitlement — even phrased as 'how many X do I have' "
            "— are policy questions. Answer them from context.\n"
            "- Give a clear, direct answer first, then add brief explanation if helpful.\n"
            "- If the answer is not found anywhere in the context, say exactly: "
            "'I don't have enough information to answer that.'\n"
            "- Never show tool call syntax or internal steps in your response.\n"
            "- Never mention missing documents or access restrictions.\n"
            "- Every answer MUST include a citation from the source document.\n"
            "- Cite the source document naturally (e.g. 'According to the leave policy...').\n\n"
            "- If multiple sources are used, cite each clearly.\n"
            "- Do NOT answer without citing a source.\n"
            "- Do NOT mention the source file name or page number explicitly.\n"
            "- If the user asks about a policy that is not in the context, say: 'I don't have enough information to answer that.'\n"

            "Answer format:\n"
            "- Write a clear answer.\n"
            "- Start with the citation.\n"
            "- Then explain.\n\n"

            "Tool usage:\n"
            "- Call `get_employee_context(user_id)` ONLY when the user explicitly asks about their "
            "manager name, grade, or team.\n\n"
            
            f"User context: department={user.department}, level={user.level}, user_id={user.user_id}\n\n"
            
            "Retrieved policy context:\n"
            + "\n".join(context_lines)
            + "\n\n"
            + user_question_block
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

    @staticmethod
    def _split_questions(text: str) -> list[str]:
        """Split user text into one or more normalized question-like parts."""
        cleaned = (text or "").strip()
        if not cleaned:
            return [""]

        # Strong delimiters first.
        parts = [p.strip(" .") for p in re.split(r"[?\n;]+", cleaned) if p.strip()]
        if len(parts) > 1:
            return parts

        # Handle prompts like:
        # "How many leaves do I get and who is my manager"
        # "Tell me leave policy, also what is travel reimbursement"
        parts = [
            p.strip(" .")
            for p in re.split(
                rf"\s+(?:and|also|then|plus)\s+(?={QUESTION_START_RE}\b)",
                cleaned,
                flags=re.IGNORECASE,
            )
            if p.strip()
        ]
        if len(parts) > 1:
            return parts

        # Fallback: split on comma before a likely question starter.
        parts = [
            p.strip(" .")
            for p in re.split(
                rf",\s*(?={QUESTION_START_RE}\b)",
                cleaned,
                flags=re.IGNORECASE,
            )
            if p.strip()
        ]
        return parts if parts else [cleaned]

    async def _retrieve_chunks(
        self, user: AuthenticatedUser, questions: list[str]
    ) -> list[RetrievedChunk]:
        """Retrieve chunks per question and merge unique high-signal results."""
        combined: list[RetrievedChunk] = []
        seen_keys: set[tuple[str, str, str]] = set()

        for question in questions:
            query_vector = (await asyncio.to_thread(embed_texts, [question]))[0]
            chunks = await asyncio.to_thread(
                self.vector_store.search,
                query_vector,
                question,
                user.department,
                user.level,
                self.settings.retrieval_limit,
            )
            for chunk in chunks:
                key = (
                    chunk.text,
                    str(chunk.metadata.get("source_file", "")),
                    str(chunk.metadata.get("page", "")),
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                combined.append(chunk)

        combined.sort(key=lambda c: c.score, reverse=True)
        return combined[: self.settings.retrieval_limit]

