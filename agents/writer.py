"""
Writer: the only agent with access to raw search_results.

Deliberate context isolation -- the critic never sees raw sources, only
this agent's draft. That forces the critic to judge the answer the way
a reader would, and keeps its context window small and fast.
"""
from llm import response_text, worker_llm
from state import ResearchState

_SYSTEM_PROMPT = """You write a clear, well-cited answer to a research \
question using ONLY the provided search results (and, if present, a prior \
session summary explicitly labeled as such). Cite web sources inline as \
[n] matching their order in the source list. If you draw on the prior \
session, say so explicitly (e.g. "per an earlier session...") rather than \
presenting it as a fresh finding -- it may be stale. If the results don't \
fully answer the question, say what's missing rather than inventing content."""


def writer_node(state: ResearchState) -> dict:
    sources = state.get("search_results", [])
    source_block = "\n".join(
        f"[{i+1}] {s['title']} ({s['url']})\n{s['content'][:800]}"
        for i, s in enumerate(sources)
    ) or "(no sources found)"

    prompt = f"Question: {state['task']}\n\nSources:\n{source_block}"

    recalled = state.get("recalled_context") or []
    if recalled:
        top = recalled[0]
        prompt += (
            f"\n\nPrior session summary (similarity {top['similarity']}, "
            f"from {top['timestamp']} -- may be stale, treat as background "
            f"not a citable web source):\n{top['final_answer'][:800]}"
        )

    response = worker_llm.invoke(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )

    draft = response_text(response)
    if not draft:
        # Last-resort guard from the original project -- draft_answer
        # should never silently be "".
        draft = "[writer produced no output -- see [warn] logs above]"

    return {"draft_answer": draft}
