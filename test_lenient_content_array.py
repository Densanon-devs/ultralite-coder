"""Tier-6 lenient content-array recovery: reconstruct a write_file call whose
content array has UNESCAPED inner double-quotes (the residual failure the
array-of-lines form does NOT solve — arrays fix newlines, not inner quotes).

Fast, no GPU. Run: python -m pytest test_lenient_content_array.py -q
"""
from engine import agent_tools as T


def test_unescaped_inner_quotes_reconstructs():
    """A content line like `print(f"{x}")` with UNESCAPED inner quotes breaks
    strict JSON; tier-6 must reconstruct the lines verbatim."""
    body = (
        '{"name": "write_file", "arguments": {"path": "cli.py", "content": '
        '["import argparse", "def show(t):", '
        '"    print(f"{t.id}: {t.title}")", "    return t"]}}'
    )
    obj = T._decode_with_repair(body)
    assert obj is not None, "tier-6 should have recovered the call"
    assert obj["name"] == "write_file"
    assert obj["arguments"]["path"] == "cli.py"
    assert obj["arguments"]["content"] == [
        "import argparse",
        "def show(t):",
        '    print(f"{t.id}: {t.title}")',   # inner quotes preserved verbatim
        "    return t",
    ], obj["arguments"]["content"]


def test_escaped_inner_quotes_still_work():
    """When the model DOES escape (\\\"), the reconstruction must yield the
    unescaped source line, not double-backslashes. Second element is unescaped
    so strict decode fails and tier-6 fires."""
    body = (
        '{"name": "write_file", "arguments": {"path": "a.py", "content": '
        '["x = {\\"k\\": \\"v\\"}", "print(f"{x}")"]}}'
    )
    obj = T._decode_with_repair(body)
    assert obj is not None
    content = obj["arguments"]["content"]
    assert content[0] == 'x = {"k": "v"}', content
    assert content[1] == 'print(f"{x}")', content


def test_clean_body_still_parses_via_tier1():
    """A well-escaped body must still parse (via strict/lenient Tier-1) —
    tier-6 must not interfere with the happy path."""
    body = (
        '{"name": "write_file", "arguments": {"path": "ok.py", "content": '
        '["def add(a, b):", "    return a + b"]}}'
    )
    obj = T._decode_with_repair(body)
    assert obj is not None
    assert obj["arguments"]["content"] == ["def add(a, b):", "    return a + b"]


def test_non_write_file_not_reconstructed():
    """Tier-6 is gated to write_file: a non-write_file body must return None
    from the reconstructor (not be mis-claimed)."""
    body = (
        '{"name": "edit_file", "arguments": {"path": "a.py", '
        '"old_string": "print("x")", "new_string": "print("y")"}}'
    )
    assert T._try_lenient_content_array(body) is None


def test_truncated_array_not_claimed_by_tier6():
    """A genuinely truncated array (no closing `]`) must NOT be claimed by
    tier-6 (it self-gates on a closing bracket) so tier-4 can handle it."""
    body = (
        '{"name": "write_file", "arguments": {"path": "x.py", "content": '
        '["line one", "line two", "line thr'
    )
    assert T._try_lenient_content_array(body) is None


def test_full_parse_path_end_to_end():
    """Through the public parser: a bare-JSON write_file with unescaped inner
    quotes should yield one valid ToolCall, no parse_error."""
    text = (
        'Sure, creating the file.\n'
        '{"name": "write_file", "arguments": {"path": "cli.py", "content": '
        '["def show(t):", "    print(f"{t.title}")"]}}'
    )
    calls, errors = T.parse_tool_calls_with_errors(text)
    assert len(calls) == 1, (calls, errors)
    assert calls[0].name == "write_file"
    assert calls[0].arguments["content"][1] == '    print(f"{t.title}")'


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    for fn in [test_unescaped_inner_quotes_reconstructs, test_escaped_inner_quotes_still_work,
               test_clean_body_still_parses_via_tier1, test_non_write_file_not_reconstructed,
               test_truncated_array_not_claimed_by_tier6, test_full_parse_path_end_to_end]:
        fn()
        print(f"PASS {fn.__name__}")
    print("all green")
