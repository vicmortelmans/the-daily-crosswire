from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from app.config import settings


class LLMUnavailable(RuntimeError):
    pass


def _using_ollama() -> bool:
    """Explicit opt-in: both OLLAMA_API_BASE and OLLAMA_MODEL must be set.
    When on, we bypass the OpenAI-compat layer and use Ollama's native
    /api/chat endpoint (better tool-calling behavior)."""
    return bool(settings.ollama_api_base and settings.ollama_model)


def chat_llm_configured() -> bool:
    if _using_ollama():
        return True
    return bool(settings.openai_api_key and settings.openai_api_key != "sk-replace-me")


@lru_cache(maxsize=1)
def get_chat_llm() -> BaseChatModel:
    if not chat_llm_configured():
        raise LLMUnavailable(
            "No LLM configured. Either set OLLAMA_API_BASE + OLLAMA_MODEL or "
            "OPENAI_API_KEY in .env."
        )
    if _using_ollama():
        # Native Ollama client. Tool calling goes through /api/chat with a
        # real `tools=[...]` payload that Gemma 4 / Qwen / etc. can act on
        # with their native function-calling behavior.
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_api_base.rstrip("/"),
            temperature=0.3,
        )
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base,
        temperature=0.3,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        reasoning_effort="none",  # required for gpt-5.6-luna
    )


def get_structured_llm(schema, temperature: float | None = None):
    """Return an LLM bound to emit output matching `schema`. ChatOllama uses
    its native tool-calling; ChatOpenAI uses function-calling against the
    OpenAI-compat endpoint. Our _invoke_with_parse_retry catches the odd
    schema miss in either case.

    Pass `temperature=0` on decision nodes (select, critique) for
    near-deterministic output. Writing nodes (draft, editorial) can stay at
    the factory default (0.3) for slight natural variation.
    """
    llm = get_chat_llm()
    if temperature is not None:
        llm = llm.bind(temperature=temperature)
    if _using_ollama():
        return llm.with_structured_output(schema)
    return llm.with_structured_output(schema, method="function_calling")
