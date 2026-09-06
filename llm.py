"""
Shared LLM instances.

Kept as qwen3.5-instruct (non-thinking build) after the earlier debugging
session found that the thinking variant (qwen3.5:9b) sometimes writes its
entire output to a separate `thinking` field and leaves `response.content`
empty -- a known, currently-open Ollama bug that gets worse the more the
prompt is stuffed with context (exactly what synthesize/writer does with
~12 search results). The instruct build avoids this class of bug by
writing directly to `content`.

_response_text is kept as a defensive fallback even though it should be
dead code on the instruct model -- cheap insurance, and worth knowing
it's there on purpose rather than leftover cruft.
"""
from langchain_ollama import ChatOllama

from config import settings

supervisor_llm = ChatOllama(
    model=settings.supervisor_model,
    base_url=settings.ollama_base_url,
    num_ctx=settings.num_ctx,
    num_predict=settings.num_predict,
    temperature=0,
)

worker_llm = ChatOllama(
    model=settings.worker_model,
    base_url=settings.ollama_base_url,
    num_ctx=settings.num_ctx,
    num_predict=settings.num_predict,
    temperature=0.3,
)


def response_text(response) -> str:
    """
    Extract text from a ChatOllama response, falling back to the
    thinking/reasoning field if `content` came back empty.
    """
    content = (response.content or "").strip()
    if content:
        return content

    kwargs = getattr(response, "additional_kwargs", {}) or {}
    fallback = kwargs.get("thinking") or kwargs.get("reasoning_content") or ""
    fallback = fallback.strip()
    if fallback:
        print("[warn] response.content was empty; used thinking/reasoning fallback")
        return fallback

    print("[warn] response.content and thinking/reasoning fallback were both empty")
    return ""
