"""
Critic: judges the draft. Architecturally cannot see search_results or
call tools -- only `task` and `draft_answer` are in its prompt. This is
enforced by what we pass into the call, not just by instruction, so it
can't quietly re-litigate sources instead of judging the answer itself.
"""
from pydantic import BaseModel, Field

from llm import supervisor_llm
from state import ResearchState


class Critique(BaseModel):
    is_sufficient: bool = Field(description="True if the draft adequately answers the task.")
    critique: str = Field(description="Specific, actionable feedback. Empty string if sufficient.")


_SYSTEM_PROMPT = """You judge whether a draft answer sufficiently addresses \
a research question. Be specific about gaps -- vague feedback like "needs \
more detail" is not useful. If the draft is genuinely adequate, say so; \
don't nitpick for the sake of another loop."""


def critic_node(state: ResearchState) -> dict:
    critic = supervisor_llm.with_structured_output(Critique)

    prompt = f"Question: {state['task']}\n\nDraft answer:\n{state['draft_answer']}"

    result: Critique = critic.invoke(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )

    return {"is_sufficient": result.is_sufficient, "critique": result.critique}
