"""Provider-agnostic LLM clients with streaming and optional tool-calling."""

import json
from collections.abc import AsyncIterator
from typing import Any, Awaitable, Callable

import httpx
import ollama
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from app.config import get_settings


ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]

def _employee_tool_openai_schema() -> list[dict[str, Any]]:
    """Return OpenAI-compatible schema for the employee context tool."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_employee_context",
                "description": "Get employee profile, manager, and team info for personalization.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string", "description": "Employee user id"},
                    },
                    "required": ["user_id"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def _employee_tool_gemini_schema() -> list[dict[str, Any]]:
    """Return Gemini tool declaration for the employee context tool."""
    return [
        {
            "function_declarations": [
                {
                    "name": "get_employee_context",
                    "description": "Get employee profile, manager, and team info for personalization.",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "user_id": {"type": "STRING", "description": "Employee user id"}
                        },
                        "required": ["user_id"],
                    },
                }
            ]
        }
    ]


class LLMClient:
    """Base interface for streaming chat responses from an LLM provider."""

    async def stream_answer(
        self,
        prompt: str,
        user_id: str,
        tool_executor: ToolExecutor | None = None,
    ) -> AsyncIterator[str]:
        raise NotImplementedError


class OllamaLLM(LLMClient):
    """Ollama-backed streaming client with optional tool-call support."""

    def __init__(self) -> None:
        settings = get_settings()
        self.client = ollama.AsyncClient(host=settings.ollama_base_url)
        self.model = settings.ollama_model

    async def stream_answer(
        self,
        prompt: str,
        user_id: str,
        tool_executor: ToolExecutor | None = None,
    ) -> AsyncIterator[str]:
        """Stream an answer from Ollama, resolving tool calls when requested."""
        if tool_executor is None:
            stream = await self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            async for part in stream:
                content = part.get("message", {}).get("content", "")
                if content:
                    yield content
            return

        first_response = await self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            tools=_employee_tool_openai_schema(),
            stream=False,
        )
        message = first_response.get("message", {})
        tool_calls = message.get("tool_calls", []) or []
        if not tool_calls:
            stream = await self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            async for part in stream:
                content = part.get("message", {}).get("content", "")
                if content:
                    yield content
            return

        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        messages.append(
            {
                "role": "assistant",
                "content": message.get("content", "") or "",
                "tool_calls": tool_calls,
            }
        )
        for call in tool_calls:
            function = call.get("function", {})
            name = function.get("name", "")
            args = function.get("arguments", {}) or {}
            if isinstance(args, str):
                args = json.loads(args or "{}")
            args.setdefault("user_id", user_id)
            result = await tool_executor(name, args)
            messages.append({"role": "tool", "content": json.dumps(result)})

        stream = await self.client.chat(
            model=self.model,
            messages=messages,
            stream=True,
        )
        async for part in stream:
            content = part.get("message", {}).get("content", "")
            if content:
                yield content


# OpenAI | Groq compatible LLM
class OpenAICompatibleLLM(LLMClient):
    """OpenAI-compatible client used for OpenAI and Groq APIs."""

    def __init__(self, *, api_key: str, model: str, base_url: str | None = None) -> None:
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = get_settings().llm_temperature

    async def stream_answer(
        self,
        prompt: str,
        user_id: str,
        tool_executor: ToolExecutor | None = None,
    ) -> AsyncIterator[str]:
        """Stream an answer and execute tool calls through compatible APIs."""
        if tool_executor is None:
            stream = await self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                stream=True,
                messages=[{"role": "user", "content": prompt}],
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
            return

        tools = _employee_tool_openai_schema()

        first_response = await self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            stream=False,
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
            tool_choice="auto",
        )
        message = first_response.choices[0].message
        tool_calls = message.tool_calls or []
        if not tool_calls:
            stream = await self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                stream=True,
                messages=[{"role": "user", "content": prompt}],
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
            return

        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        assistant_tool_calls: list[dict[str, Any]] = []
        for call in tool_calls:
            assistant_tool_calls.append(
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
            )
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": assistant_tool_calls,
            }
        )

        for call in tool_calls:
            args = json.loads(call.function.arguments or "{}")
            args.setdefault("user_id", user_id)
            result = await tool_executor(call.function.name, args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result),
                }
            )

        stream = await self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            stream=True,
            messages=messages,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta


class GeminiLLM(LLMClient):
    """Gemini client using HTTP endpoints for tool use and SSE streaming."""

    def __init__(self, *, api_key: str, model: str, base_url: str) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = get_settings().llm_temperature

    async def stream_answer(
        self,
        prompt: str,
        user_id: str,
        tool_executor: ToolExecutor | None = None,
    ) -> AsyncIterator[str]:
        """Stream Gemini output and insert tool responses into follow-up calls."""
        stream_url = (
            f"{self.base_url}/models/{self.model}:streamGenerateContent"
            f"?alt=sse&key={self.api_key}"
        )
        generate_url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.temperature},
        }

        if tool_executor is not None:
            async with httpx.AsyncClient(timeout=60.0) as client:
                first_response = await client.post(
                    generate_url,
                    json={**payload, "tools": _employee_tool_gemini_schema()},
                )
                first_response.raise_for_status()
                first_json = first_response.json()

            first_candidates = first_json.get("candidates") or []
            first_parts = (
                first_candidates[0].get("content", {}).get("parts", []) if first_candidates else []
            )
            function_call = next((p.get("functionCall") for p in first_parts if p.get("functionCall")), None)
            if function_call:
                args = function_call.get("args", {}) or {}
                args.setdefault("user_id", user_id)
                tool_result = await tool_executor(function_call.get("name", ""), args)
                payload = {
                    "contents": [
                        {"role": "user", "parts": [{"text": prompt}]},
                        {"role": "model", "parts": [{"functionCall": function_call}]},
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "functionResponse": {
                                        "name": function_call.get("name", ""),
                                        "response": {"result": tool_result},
                                    }
                                }
                            ],
                        },
                    ],
                    "generationConfig": {"temperature": self.temperature},
                    "tools": _employee_tool_gemini_schema(),
                }

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", stream_url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw:
                        continue
                    chunk = json.loads(raw)
                    candidates = chunk.get("candidates") or []
                    if not candidates:
                        continue
                    parts = candidates[0].get("content", {}).get("parts") or []
                    for part in parts:
                        text = part.get("text")
                        if text:
                            yield text


class AnthropicLLM(LLMClient):
    """Anthropic streaming client with support for tool execution."""

    def __init__(self, *, api_key: str, model: str) -> None:
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.temperature = get_settings().llm_temperature

    async def stream_answer(
        self,
        prompt: str,
        user_id: str,
        tool_executor: ToolExecutor | None = None,
    ) -> AsyncIterator[str]:
        """Stream Anthropic responses, including tool-use."""
        if tool_executor is None:
            async with self.client.messages.stream(
                model=self.model,
                max_tokens=1024,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for text in stream.text_stream:
                    if text:
                        yield text
            return

        tools = [
            {
                "name": "get_employee_context",
                "description": "Get employee profile, manager, and team info for personalization.",
                "input_schema": {
                    "type": "object",
                    "properties": {"user_id": {"type": "string", "description": "Employee user id"}},
                    "required": ["user_id"],
                },
            }
        ]
        first = await self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=self.temperature,
            tools=tools,
            messages=[{"role": "user", "content": prompt}],
        )
        tool_uses = [block for block in first.content if getattr(block, "type", None) == "tool_use"]
        if not tool_uses:
            async with self.client.messages.stream(
            model=self.model,
            max_tokens=1024,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for text in stream.text_stream:
                    if text:
                        yield text
            return

        assistant_content: list[dict[str, Any]] = []
        for block in first.content:
            if getattr(block, "type", None) == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif getattr(block, "type", None) == "tool_use":
                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )

        tool_result_blocks: list[dict[str, Any]] = []
        for block in tool_uses:
            args = dict(block.input or {})
            args.setdefault("user_id", user_id)
            result = await tool_executor(block.name, args)
            tool_result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                }
            )

        async with self.client.messages.stream(
            model=self.model,
            max_tokens=1024,
            temperature=self.temperature,
            tools=tools,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant_content},
                {"role": "user", "content": tool_result_blocks},
            ],
        ) as stream:
            async for text in stream.text_stream:
                if text:
                    yield text


class FallbackLLM(LLMClient):
    """Fallback responder used when no configured provider is available."""

    async def stream_answer(
        self,
        prompt: str,
        user_id: str,
        tool_executor: ToolExecutor | None = None,
    ) -> AsyncIterator[str]:
        """Stream a configuration hint in small chunks."""
        settings = get_settings()
        content = (
            "I do not have a configured LLM provider right now. "
            "So Configure LLM_PROVIDER plus provider env vars in .env "
            "(ollama/openai/groq/gemini/anthropic) to enable model output.\n\n"
        )
        for i in range(0, len(content), settings.stream_chunk_chars):
            yield content[i : i + settings.stream_chunk_chars]


def get_llm_client() -> LLMClient:
    """Return the active LLM client based on environment configuration."""
    settings = get_settings()
    provider = settings.llm_provider

    if provider == "openai":
        if settings.openai_api_key:
            return OpenAICompatibleLLM(
                api_key=settings.openai_api_key,
                model=settings.openai_model,
            )
        return FallbackLLM()

    if provider == "groq":
        if settings.groq_api_key:
            return OpenAICompatibleLLM(
                api_key=settings.groq_api_key,
                model=settings.groq_model,
                base_url=settings.groq_base_url,
            )
        return FallbackLLM()

    if provider == "ollama":
        if settings.ollama_model:
            return OllamaLLM()
        return FallbackLLM()

    if provider == "gemini":
        if settings.gemini_api_key:
            return GeminiLLM(
                api_key=settings.gemini_api_key,
                model=settings.gemini_model,
                base_url=settings.gemini_base_url,
            )
        return FallbackLLM()

    if provider == "anthropic":
        if settings.anthropic_api_key:
            return AnthropicLLM(
                api_key=settings.anthropic_api_key,
                model=settings.anthropic_model,
            )
        return FallbackLLM()

    return FallbackLLM()

