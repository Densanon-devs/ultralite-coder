"""
Session-Complexity Monitor — Feature 2 of Phase 14+

Research motivation: Stroop-effect attention-collapse studies (ScienceDaily
2026-06-10) show that tool-call discipline degrades as agentic chains lengthen.
The failure mode is distractor accumulation — not just raw prompt length —
so simply counting chars (as context_char_budget already does) is insufficient.
This monitor adds a PROACTIVE second signal:

    1. Tool-call depth — how many tool calls have executed this session.
       A large depth means the model has processed many observations and its
       working "mental model" of the task state may be stale.

    2. Tool-output volume — how many chars of tool output have been
       accumulated in the transcript. Large tool results (grep matches,
       long file reads) are the primary distractor class.

When either metric exceeds its configured threshold, the monitor fires
Agent._maybe_compact_transcript() PROACTIVELY — before the reactive
context_char_budget ceiling is hit. This gives the model a leaner context
earlier in the chain, when compaction is cheapest (fewer old turns to elide).

Design:
- Pure in-process, zero servers, stdlib only.
- No coupling to the Agent constructor — wire it in the calling site
  (ulcagent.py / main.py) after constructing the Agent.
- Thresholds default to conservative values that never fire on short tasks
  (depth < 12 or volume < 32000 chars is typical for the 14-task bench).
- Setting a threshold to 0 disables that axis entirely (opt-out per axis).
- Idempotent: calling maybe_compact() repeatedly is safe — it delegates
  to the same _maybe_compact_transcript that is already idempotent.

Integration pattern (ulcagent.py / main.py):

    monitor = SessionComplexityMonitor(
        tool_call_depth_threshold=15,    # fire when 15+ tool calls in session
        tool_output_volume_threshold=50000,  # or 50k chars of tool output
    )
    # In the agent loop (or after each iteration):
    monitor.maybe_compact(agent)

Or use Agent.complexity_monitor to wire it directly into the run loop
(see Agent.__init__ for the optional parameter).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.agent import Agent

# Default thresholds — conservative so short sessions are unaffected.
_DEFAULT_DEPTH_THRESHOLD: int = 15   # tool calls in session
_DEFAULT_VOLUME_THRESHOLD: int = 32_000  # chars of accumulated tool output


class SessionComplexityMonitor:
    """Proactive compaction trigger based on session-complexity heuristics.

    Attributes
    ----------
    tool_call_depth_threshold : int
        Fire compaction when the cumulative tool-call count reaches this value.
        Set to 0 to disable the depth axis.
    tool_output_volume_threshold : int
        Fire compaction when the cumulative tool-output char count reaches
        this value.  Set to 0 to disable the volume axis.
    """

    def __init__(
        self,
        tool_call_depth_threshold: int = _DEFAULT_DEPTH_THRESHOLD,
        tool_output_volume_threshold: int = _DEFAULT_VOLUME_THRESHOLD,
    ) -> None:
        self.tool_call_depth_threshold = int(tool_call_depth_threshold)
        self.tool_output_volume_threshold = int(tool_output_volume_threshold)

    # ── pure metric computation ───────────────────────────────────

    def compute_metrics(self, agent: "Agent") -> tuple[int, int]:
        """Return (tool_call_depth, tool_output_volume) from the agent's
        current transcript.

        - tool_call_depth: count of turns with role == "tool".
        - tool_output_volume: sum of char lengths of those turns' content.

        These are cumulative session metrics: they grow monotonically as
        the agent processes more tool calls. The method is read-only — no
        side effects on the agent.
        """
        depth = 0
        volume = 0
        for turn in agent._transcript:
            if turn.get("role") == "tool":
                depth += 1
                volume += len(turn.get("content") or "")
        return depth, volume

    def should_compact(self, tool_call_depth: int, tool_output_volume: int) -> bool:
        """Return True if either complexity metric has crossed its threshold.

        Both axes are independent. Either crossing its threshold fires
        compaction. A threshold of 0 means the axis is disabled.
        """
        depth_triggered = (
            self.tool_call_depth_threshold > 0
            and tool_call_depth >= self.tool_call_depth_threshold
        )
        volume_triggered = (
            self.tool_output_volume_threshold > 0
            and tool_output_volume >= self.tool_output_volume_threshold
        )
        return depth_triggered or volume_triggered

    # ── side-effecting integration ────────────────────────────────

    def maybe_compact(self, agent: "Agent") -> bool:
        """Compute metrics from *agent* and proactively compact if triggered.

        Delegates to Agent._maybe_compact_transcript() — the same method the
        reactive compaction uses — so the compaction logic is not duplicated.
        Returns True if compaction was requested (threshold crossed), False
        otherwise.

        Calling this is idempotent: the underlying _maybe_compact_transcript
        is a no-op when the transcript is already within budget or too short
        to elide anything useful.
        """
        depth, volume = self.compute_metrics(agent)
        if not self.should_compact(depth, volume):
            return False
        agent._maybe_compact_transcript()
        return True

    # ── representation ────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"SessionComplexityMonitor("
            f"depth_threshold={self.tool_call_depth_threshold}, "
            f"volume_threshold={self.tool_output_volume_threshold})"
        )
