"""Finalize: no LLM call. Appends a source list to the last draft and terminates the graph."""
from state import ResearchState


def finalize_node(state: ResearchState) -> dict:
    sources = state.get("search_results", [])
    source_lines = "\n".join(
        f"[{i+1}] {s['title']} - {s['url']}" for i, s in enumerate(sources) if s.get("url")
    )

    final = state.get("draft_answer", "").strip()
    if source_lines:
        final += f"\n\nSources:\n{source_lines}"

    return {"final_answer": final}
