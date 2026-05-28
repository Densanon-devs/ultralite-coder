"""Prompt templates for the /goal persistent-objective loop.

Two templates do all the work — no Python-side loop state machine.

`CONTINUATION_PROMPT` is fed back into the running session after each
final answer. It asks the model to evaluate whether the *original*
goal has been fully achieved. If yes, the model emits the literal
sentinel `GOAL_COMPLETE`. If no, the model picks the next concrete
sub-task and continues.

`BUDGET_LIMIT_PROMPT` is used only when the token budget is exhausted
mid-loop. It asks the model to summarize what remains undone so the
human operator can resume manually next session.
"""

DONE_SENTINEL = "GOAL_COMPLETE"

CONTINUATION_PROMPT = (
    "Evaluation step. The persistent goal is:\n"
    "<goal>\n{goal}\n</goal>\n\n"
    "Iteration {iteration} just produced the answer above. Token budget remaining: ~{remaining_tokens} tokens.\n\n"
    "Decide one of two things:\n"
    "1. The goal is FULLY achieved — every requirement met, all files written, all tests passing if the goal asked for them. "
    f"In that case, reply with the single literal word `{DONE_SENTINEL}` on its own line and nothing else.\n"
    "2. The goal is NOT yet fully achieved. In that case, briefly state what's still outstanding (one sentence), "
    "then pick the next concrete sub-task and execute it using your tools. Do not repeat work already done above.\n\n"
    "Be honest. If the previous iteration silently dropped a requirement, surface it now. "
    "Do not claim completion just to exit the loop."
)

ACCEPTANCE_RETRY_PROMPT = (
    "You replied `" + DONE_SENTINEL + "`, but an objective acceptance check "
    "says the goal is NOT actually complete yet:\n"
    "<check_failed>\n{feedback}\n</check_failed>\n\n"
    "The persistent goal is:\n"
    "<goal>\n{goal}\n</goal>\n\n"
    "Do NOT declare completion again until the issue above is resolved. "
    "Address it now using your tools, then re-evaluate. "
    "(acceptance retry {retry}/{max_retries})"
)

BUDGET_LIMIT_PROMPT = (
    "Token budget exhausted. The persistent goal was:\n"
    "<goal>\n{goal}\n</goal>\n\n"
    "Summarize in 3-6 bullets:\n"
    "- What was completed across all iterations.\n"
    "- What is still outstanding.\n"
    "- The single most useful next step a human or future agent could take to resume.\n\n"
    "Do not call any more tools. Reply with the summary as plain text."
)
