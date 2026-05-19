"""Tests for /export HTML mode (Task 3, 2026-05-19).

Covers:
- _parse_export_args: flag parsing edge cases
- _render_message_html: code fences + inline backticks + severity chips + escape
- _build_session_html: end-to-end document well-formedness
- _export_session: integration with the session log + file write
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import ulcagent


class TestParseExportArgs(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(ulcagent._parse_export_args(""), ("", None))

    def test_filename_only(self):
        self.assertEqual(ulcagent._parse_export_args("foo.md"), ("foo.md", None))

    def test_html_flag(self):
        self.assertEqual(ulcagent._parse_export_args("--html"), ("", "html"))

    def test_md_flag(self):
        self.assertEqual(
            ulcagent._parse_export_args("--md myfile"), ("myfile", "md")
        )

    def test_format_kv(self):
        self.assertEqual(
            ulcagent._parse_export_args("--format html"), ("", "html")
        )

    def test_format_kv_then_name(self):
        self.assertEqual(
            ulcagent._parse_export_args("--format html report"),
            ("report", "html"),
        )

    def test_name_then_format(self):
        # tokens parsed in order — filename can come first
        self.assertEqual(
            ulcagent._parse_export_args("review --html"), ("review", "html")
        )

    def test_html_suffix_infers_format(self):
        self.assertEqual(
            ulcagent._parse_export_args("audit.html"), ("audit.html", "html")
        )


class TestRenderMessageHtml(unittest.TestCase):
    def test_plain_text_escaped(self):
        out = ulcagent._render_message_html("a & b < c > d")
        self.assertIn("a &amp; b &lt; c &gt; d", out)

    def test_fenced_code_block(self):
        out = ulcagent._render_message_html("```python\ndef foo():\n    pass\n```")
        self.assertIn("<pre><code>", out)
        self.assertIn("def foo():", out)
        self.assertIn("</code></pre>", out)
        # lang hint is dropped
        self.assertNotIn("python\ndef foo", out)

    def test_inline_backticks(self):
        out = ulcagent._render_message_html("call `foo()` next")
        self.assertIn("<code>foo()</code>", out)

    def test_severity_chip_critical(self):
        out = ulcagent._render_message_html("- critical: SQL injection")
        self.assertIn('class="sev sev-critical"', out)
        self.assertIn(">critical<", out)

    def test_severity_chip_all_four(self):
        text = (
            "critical: hot\n"
            "high: medium-hot\n"
            "medium: warm\n"
            "low: chill\n"
        )
        out = ulcagent._render_message_html(text)
        for level in ("critical", "high", "medium", "low"):
            self.assertIn(f"sev-{level}", out, f"missing chip for {level}")

    def test_severity_only_at_line_start(self):
        # Should NOT chip "low" in the middle of a sentence
        out = ulcagent._render_message_html("this scored low across the board")
        self.assertNotIn('sev-low', out)

    def test_html_injection_escaped(self):
        out = ulcagent._render_message_html("<script>alert(1)</script>")
        self.assertNotIn("<script>", out)
        self.assertIn("&lt;script&gt;", out)


class TestBuildSessionHtml(unittest.TestCase):
    def _entries(self):
        return [
            {"role": "user", "content": "fix the auth bug"},
            {
                "role": "assistant",
                "content": "Done.\n\nhigh: race condition\n\n```\nlock.acquire()\n```",
                "stats": "iter=4 wall=18.2s",
            },
        ]

    def test_doctype_and_meta(self):
        out = ulcagent._build_session_html(Path("/tmp/ws"), self._entries())
        self.assertTrue(out.startswith("<!doctype html>"))
        self.assertIn('<meta charset="utf-8">', out)

    def test_user_and_assistant_classes(self):
        out = ulcagent._build_session_html(Path("/tmp/ws"), self._entries())
        self.assertIn('class="user-turn"', out)
        self.assertIn('class="assistant-turn"', out)
        self.assertIn('class="stats"', out)

    def test_workspace_path_appears(self):
        # str(Path(...)) uses native separators on Windows, so compare against
        # what the platform actually produces.
        ws = Path("/some/special/workspace")
        out = ulcagent._build_session_html(ws, self._entries())
        self.assertIn(str(ws), out)

    def test_dark_mode_css_present(self):
        out = ulcagent._build_session_html(Path("/tmp/ws"), self._entries())
        # confirms the CSS is embedded rather than referenced — air-gap safe
        self.assertIn("prefers-color-scheme: dark", out)
        self.assertNotIn("<link rel=\"stylesheet\"", out)

    def test_entry_count_in_header(self):
        out = ulcagent._build_session_html(Path("/tmp/ws"), self._entries())
        self.assertIn("2 entries", out)


class TestExportSessionIntegration(unittest.TestCase):
    def setUp(self):
        self._saved = ulcagent._session_log[:]
        ulcagent._session_log[:] = [
            {"role": "user", "content": "review the diff"},
            {
                "role": "assistant",
                "content": "low: trailing whitespace on line 12",
                "stats": "iter=2",
            },
        ]

    def tearDown(self):
        ulcagent._session_log[:] = self._saved

    def test_default_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            ulcagent._export_session(ws, "")
            outs = list(ws.glob("session_*.md"))
            self.assertEqual(len(outs), 1, f"expected 1 .md file, got {outs}")
            text = outs[0].read_text(encoding="utf-8")
            self.assertIn("review the diff", text)
            self.assertIn("low: trailing whitespace", text)

    def test_html_via_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            ulcagent._export_session(ws, "--html")
            outs = list(ws.glob("session_*.html"))
            self.assertEqual(len(outs), 1)
            text = outs[0].read_text(encoding="utf-8")
            self.assertTrue(text.startswith("<!doctype html>"))
            self.assertIn("sev-low", text)

    def test_html_via_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            ulcagent._export_session(ws, "myreport.html")
            p = ws / "myreport.html"
            self.assertTrue(p.exists())
            self.assertIn("<!doctype html>", p.read_text(encoding="utf-8"))

    def test_html_custom_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            ulcagent._export_session(ws, "--format html review2026")
            # extension auto-appended
            p = ws / "review2026.html"
            self.assertTrue(p.exists())


if __name__ == "__main__":
    unittest.main()
