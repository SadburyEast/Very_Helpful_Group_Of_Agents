"""
Persist: writes the completed session to long-term memory.

Runs after `finalize`, before END. No LLM call -- just embeds the task
and stores it alongside the final answer for future `recall` calls.
"""
import memory
from state import ResearchState


def persist_memory_node(state: ResearchState) -> dict:
    memory.persist(state["task"], state.get("final_answer", ""))
    return {}
