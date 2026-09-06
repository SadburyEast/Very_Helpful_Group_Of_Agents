"""
Recall: checks long-term memory for similar past sessions.

Runs once, deterministically, before the supervisor takes over -- this
isn't something worth an LLM routing decision. No LLM call: it's an
embedding lookup, so it's cheap and doesn't touch the GPU/generation
budget at all.
"""
import memory
from state import ResearchState


def recall_node(state: ResearchState) -> dict:
    recalled = memory.recall(state["task"])
    return {"recalled_context": recalled}
