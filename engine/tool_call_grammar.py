"""GBNF grammar for constrained tool-call generation (Task A1, 2026-05-19).

Per the harness SOTA research (ACL 2025 + Don't Fine-Tune Decode arXiv 2310.07075):
grammar-constrained generation drives Mistral-Instruct from 0% to 52% tool
accuracy and eliminates syntax errors at the source. The 14B's open ceiling #1
(JSON quote-escape mid-run recovery) is exactly this failure mode — invalid
`\\@`, unescaped `"`, Python-style `'...'` — all unrepresentable under a
properly-scoped GBNF.

DESIGN NOTE: this grammar is OPT-IN, not default. Two reasons:

1. Over-constraining the model BREAKS the final-answer signal. The model
   indicates "done" by emitting prose with NO tool_call blocks. A grammar
   that REQUIRES a tool_call invalidates that contract; the agent loop
   never terminates cleanly. We support BOTH shapes (prose final OR tool
   call) but the grammar to express "either prose or one tool call" is
   non-trivial in GBNF and adds risk of over-/under-restriction.

2. Per feedback_dont_change_tool_return_formats.md and the GPT-5.5 prompt
   lift regression, the 14B is sensitive to surface-format changes.
   Enabling a grammar by default could regress baseline pass rate; the
   right deployment is a flagged opt-in for the operator to A/B against
   the current 41/42 baseline using the calibration framework shipped
   2026-05-19 AM.

USAGE:

    from engine.tool_call_grammar import load_tool_call_grammar
    agent = Agent(
        ...,
        grammar=load_tool_call_grammar(),
        grammar_use="tool_call_only",  # constrain entire output to one call
    )

Best applied selectively (e.g. ONLY when re-sampling after parse-failure,
not on first try). The Agent has no built-in opt-in flag yet — operator
wires grammar=... explicitly into model.generate calls.
"""

from __future__ import annotations

from typing import Optional

# ── The grammar ─────────────────────────────────────────────────────
#
# Modelled on llama.cpp's `grammars/json.gbnf` (ggml-org/llama.cpp tree
# /grammars/json.gbnf). We embed the JSON grammar inline and wrap it in
# the Hermes <tool_call>...</tool_call> envelope. The grammar accepts
# exactly ONE tool call.
#
# Strings: JSON-compliant — only the seven valid escape characters
# (\" \\ \/ \b \f \n \r \t) plus \uXXXX unicode. No `\@`, no `\d`,
# no unescaped `"`. This is the load-bearing constraint that targets
# ceiling #1.
TOOL_CALL_ONLY_GBNF = r'''
root ::= ws "<tool_call>" ws object ws "</tool_call>" ws

object ::= "{" ws pair ( ws "," ws pair )* ws "}"
       | "{" ws "}"
pair   ::= string ws ":" ws value

value ::= string | number | object | array | "true" | "false" | "null"

array  ::= "[" ws value ( ws "," ws value )* ws "]"
       | "[" ws "]"

string ::= "\"" char* "\""
char   ::= [^"\\] | "\\" escape
escape ::= ["\\/bfnrt] | "u" hex hex hex hex
hex    ::= [0-9a-fA-F]

number ::= "-"? int frac? exp?
int    ::= "0" | [1-9] [0-9]*
frac   ::= "." [0-9]+
exp    ::= [eE] [-+]? [0-9]+

ws ::= [ \t\n]*
'''


# Cache the parsed grammar object so we don't re-parse on every call.
_GRAMMAR_CACHE: dict = {}


def get_tool_call_only_gbnf() -> str:
    """Return the raw GBNF source string. Side-effect-free; safe to call
    from tests without llama-cpp-python installed."""
    return TOOL_CALL_ONLY_GBNF


def load_tool_call_grammar(variant: str = "tool_call_only"):
    """Lazy-load a LlamaGrammar object. Returns None if llama-cpp-python
    is unavailable so callers can branch on availability without try/except.

    variant: which grammar to use. Currently only "tool_call_only" — the
    grammar that forces the model to emit exactly one well-formed
    <tool_call>{...}</tool_call> block. Future variants might encode
    "either tool_call or prose" or per-tool argument schemas.
    """
    if variant in _GRAMMAR_CACHE:
        return _GRAMMAR_CACHE[variant]

    try:
        from llama_cpp import LlamaGrammar
    except ImportError:
        return None

    if variant == "tool_call_only":
        gbnf = TOOL_CALL_ONLY_GBNF
    else:
        raise ValueError(
            f"Unknown grammar variant {variant!r}. "
            "Currently supported: 'tool_call_only'."
        )

    grammar = LlamaGrammar.from_string(gbnf)
    _GRAMMAR_CACHE[variant] = grammar
    return grammar


# ── Static validator (no llama_cpp needed) ──────────────────────────
#
# Used by unit tests to verify that strings the grammar SHOULD accept
# and SHOULD reject behave as expected. This mirrors the grammar
# closely enough to catch obvious bugs without round-tripping through
# llama-cpp-python's parser.


def _validates_as_tool_call_envelope(text: str) -> bool:
    """Best-effort check that `text` looks like the grammar would accept.
    Conservative — returns False for ambiguous cases."""
    import json
    import re

    # Must be: optional ws, <tool_call>, ws, JSON object, ws, </tool_call>, optional ws
    m = re.fullmatch(
        r"\s*<tool_call>\s*(\{.*\})\s*</tool_call>\s*",
        text,
        flags=re.DOTALL,
    )
    if not m:
        return False
    body = m.group(1)
    try:
        # The grammar accepts only strict JSON (no Python single quotes,
        # no invalid escapes). Use the strict decoder.
        json.loads(body)
        return True
    except json.JSONDecodeError:
        return False
