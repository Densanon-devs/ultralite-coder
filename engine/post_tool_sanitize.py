"""PostToolUse-style output sanitizer (Task 5, 2026-05-19).

Sits at the seam between tool execution and the format-for-model serialization.
The motivation (per project_ultralight_coder_phase14_next.md 2026-04-28):
Claude Code's PostToolUse hooks can REPLACE tool output before the model
sees it. The 14B's "JSON quote-escape mid-run recovery" ceiling
(per feedback_14b_tool_call_ceilings.md and project_ulcagent_capacity_2026-04-26.md
ceiling #1) is partly driven by the model re-emitting problematic characters
it sees in tool returns. Cleaning those characters at the boundary lowers the
JSON-escape burden on the model.

Scope of the v1 sanitizer (intentionally narrow — measure before broadening):

1. Strip lone UTF-16 surrogates (U+D800–U+DFFF) — these appear in mis-decoded
   binary `read_file` returns and crash json.dumps on Python builds that
   serialize via UTF-16 internals.
2. Escape embedded NUL (\x00) — file reads of binary content can include these,
   and they corrupt many downstream renderers.
3. Soft-truncate very long single-line outputs that would balloon the JSON
   payload and consume too many tokens (configurable max_chars, default 30_000
   matching the web_tools cap shipped 2026-05-10).
4. No surface-format changes (do NOT rewrite quotes, do NOT alter line
   structure). Per feedback_dont_change_tool_return_formats.md: changing the
   shape of tool returns regresses the model.

Tools can opt out individually by name (e.g., `auto_verify` shouldn't be
truncated). Sanitizers are registry-extensible so future ceilings can register
additional cleaners without touching this file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

# Lone surrogates: high (D800-DBFF) and low (DC00-DFFF) that aren't paired.
# Python str can hold these but json.dumps fails depending on the codec.
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")

# Control chars except \t (\x09), \n (\x0a), \r (\x0d). Keep printable + common
# whitespace. \x00 is included so we can escape it explicitly below.
_BAD_CONTROL_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f]"
)


SanitizerFn = Callable[[str], str]


@dataclass
class SanitizerConfig:
    """Per-registry configuration for output sanitization."""
    enabled: bool = True
    max_chars: int = 30_000
    truncate_marker: str = "\n[truncated by post-tool-sanitize at {n} chars]"
    # Tool names whose content field should be passed through untouched.
    opt_out_tools: frozenset = frozenset(
        {"auto_verify"}  # auto_verify already produces compact stable text
    )
    extra_sanitizers: tuple = ()  # tuple[SanitizerFn, ...]


def _strip_lone_surrogates(s: str) -> str:
    """Replace lone surrogates with U+FFFD. Pure-Python, no codec round-trip."""
    if not _SURROGATE_RE.search(s):
        return s
    return _SURROGATE_RE.sub("�", s)


def _escape_bad_control_chars(s: str) -> str:
    """Escape NUL + most control chars as their \\xHH form (visible to model).

    Keep tab / newline / CR / vertical tab boundary cases alone — they're
    meaningful whitespace for code-shaped content.
    """
    if not _BAD_CONTROL_RE.search(s):
        return s
    return _BAD_CONTROL_RE.sub(lambda m: f"\\x{ord(m.group(0)):02x}", s)


def _maybe_truncate(s: str, max_chars: int, marker: str) -> str:
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    return s[:max_chars] + marker.format(n=max_chars)


_BUILTIN_SANITIZERS: tuple[SanitizerFn, ...] = (
    _strip_lone_surrogates,
    _escape_bad_control_chars,
)


def sanitize_string(s: str, cfg: Optional[SanitizerConfig] = None) -> str:
    """Apply all sanitizers in order. Idempotent. Pure."""
    cfg = cfg or SanitizerConfig()
    if not cfg.enabled or not s:
        return s
    out = s
    for fn in _BUILTIN_SANITIZERS:
        out = fn(out)
    for fn in cfg.extra_sanitizers:
        out = fn(out)
    out = _maybe_truncate(out, cfg.max_chars, cfg.truncate_marker)
    return out


def sanitize_content(content: Any, cfg: Optional[SanitizerConfig] = None) -> Any:
    """Recursively sanitize a tool result's `content` field.

    - str → sanitize_string
    - list/tuple → element-wise (preserves list type)
    - dict → values sanitized; keys left alone (model rarely re-emits them)
    - other → returned as-is

    Does NOT alter the structure (no flattening, no key renaming) — the model
    expects identical shapes per feedback_dont_change_tool_return_formats.md.
    """
    cfg = cfg or SanitizerConfig()
    if not cfg.enabled:
        return content
    if isinstance(content, str):
        return sanitize_string(content, cfg)
    if isinstance(content, list):
        return [sanitize_content(x, cfg) for x in content]
    if isinstance(content, tuple):
        return tuple(sanitize_content(x, cfg) for x in content)
    if isinstance(content, dict):
        return {k: sanitize_content(v, cfg) for k, v in content.items()}
    return content


def sanitize_tool_result(
    tool_name: str,
    content: Any,
    cfg: Optional[SanitizerConfig] = None,
) -> Any:
    """Entry point for ToolRegistry.execute's post-execution hook.

    Returns the sanitized content. Caller wraps in ToolResult.
    """
    cfg = cfg or SanitizerConfig()
    if not cfg.enabled or tool_name in cfg.opt_out_tools:
        return content
    return sanitize_content(content, cfg)
