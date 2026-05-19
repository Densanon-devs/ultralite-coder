"""Tests for the C2 hashline edit tool (Task C2, 2026-05-19)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from engine.agent_builtins import build_default_registry
from engine.agent_tools import ToolCall


class TestHashlineEditBasics(unittest.TestCase):
    def _make_registry(self, tmpdir):
        return build_default_registry(tmpdir, extended_tools=True)

    def test_tool_registered_with_extended(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._make_registry(tmp)
            self.assertIsNotNone(reg.get("edit_file_hashline"))

    def test_tool_NOT_registered_without_extended(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = build_default_registry(tmp, extended_tools=False)
            self.assertIsNone(reg.get("edit_file_hashline"))

    def test_happy_path_single_line_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.py"
            p.write_text("x = 1\n", encoding="utf-8")
            reg = self._make_registry(tmp)
            r = reg.execute(ToolCall(
                name="edit_file_hashline",
                arguments={
                    "path": "x.py",
                    "line_number": 1,
                    "old_line": "x = 1",
                    "new_line": "x = 42",
                },
                raw="",
            ))
            self.assertTrue(r.success, r.error)
            self.assertEqual(p.read_text(encoding="utf-8"), "x = 42\n")

    def test_replace_middle_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.py"
            p.write_text("a\nb\nc\n", encoding="utf-8")
            reg = self._make_registry(tmp)
            r = reg.execute(ToolCall(
                name="edit_file_hashline",
                arguments={
                    "path": "x.py",
                    "line_number": 2,
                    "old_line": "b",
                    "new_line": "B!",
                },
                raw="",
            ))
            self.assertTrue(r.success, r.error)
            self.assertEqual(p.read_text(encoding="utf-8"), "a\nB!\nc\n")

    def test_replace_last_line_with_trailing_newline(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.py"
            p.write_text("a\nb\nc\n", encoding="utf-8")
            reg = self._make_registry(tmp)
            r = reg.execute(ToolCall(
                name="edit_file_hashline",
                arguments={
                    "path": "x.py",
                    "line_number": 3,
                    "old_line": "c",
                    "new_line": "C",
                },
                raw="",
            ))
            self.assertTrue(r.success, r.error)
            self.assertEqual(p.read_text(encoding="utf-8"), "a\nb\nC\n")

    def test_multi_line_new_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.py"
            p.write_text("import os\nx = 1\n", encoding="utf-8")
            reg = self._make_registry(tmp)
            r = reg.execute(ToolCall(
                name="edit_file_hashline",
                arguments={
                    "path": "x.py",
                    "line_number": 1,
                    "old_line": "import os",
                    "new_line": "import os\nimport sys",
                },
                raw="",
            ))
            self.assertTrue(r.success, r.error)
            self.assertEqual(
                p.read_text(encoding="utf-8"),
                "import os\nimport sys\nx = 1\n",
            )


class TestHashlineEditSafety(unittest.TestCase):
    def _make_registry(self, tmpdir):
        return build_default_registry(tmpdir, extended_tools=True)

    def test_mismatch_rejects_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.py"
            p.write_text("x = 1\ny = 2\n", encoding="utf-8")
            reg = self._make_registry(tmp)
            r = reg.execute(ToolCall(
                name="edit_file_hashline",
                arguments={
                    "path": "x.py",
                    "line_number": 1,
                    "old_line": "x = 999",  # WRONG
                    "new_line": "x = 42",
                },
                raw="",
            ))
            self.assertFalse(r.success)
            self.assertIn("old_line mismatch", r.error)
            # File untouched
            self.assertEqual(p.read_text(encoding="utf-8"), "x = 1\ny = 2\n")

    def test_line_past_eof_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.py"
            p.write_text("a\nb\n", encoding="utf-8")
            reg = self._make_registry(tmp)
            r = reg.execute(ToolCall(
                name="edit_file_hashline",
                arguments={
                    "path": "x.py",
                    "line_number": 99,
                    "old_line": "anything",
                    "new_line": "x",
                },
                raw="",
            ))
            self.assertFalse(r.success)
            self.assertIn("past EOF", r.error)

    def test_line_number_zero_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.py"
            p.write_text("a\n", encoding="utf-8")
            reg = self._make_registry(tmp)
            r = reg.execute(ToolCall(
                name="edit_file_hashline",
                arguments={
                    "path": "x.py",
                    "line_number": 0,
                    "old_line": "a",
                    "new_line": "b",
                },
                raw="",
            ))
            self.assertFalse(r.success)
            # The schema validator catches this first (minimum: 1).
            self.assertTrue(
                "minimum" in (r.error or "").lower()
                or "positive" in (r.error or "").lower()
            )

    def test_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = self._make_registry(tmp)
            r = reg.execute(ToolCall(
                name="edit_file_hashline",
                arguments={
                    "path": "no_such_file.py",
                    "line_number": 1,
                    "old_line": "x",
                    "new_line": "y",
                },
                raw="",
            ))
            self.assertFalse(r.success)
            self.assertIn("not found", r.error.lower())


class TestHashlineEditQuoteHeavy(unittest.TestCase):
    """The actual reason this tool exists — quote-heavy lines the model
    can't reliably re-emit via JSON-escape edit_file."""

    def test_fstring_with_nested_quotes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.py"
            original = '''label = f"id:{i} {"done" if d else "todo"}"\n'''
            p.write_text(original, encoding="utf-8")
            reg = build_default_registry(tmp, extended_tools=True)
            r = reg.execute(ToolCall(
                name="edit_file_hashline",
                arguments={
                    "path": "x.py",
                    "line_number": 1,
                    "old_line": original.rstrip("\n"),
                    "new_line": 'label = f"id:{i}"',
                },
                raw="",
            ))
            self.assertTrue(r.success, r.error)
            self.assertEqual(p.read_text(encoding="utf-8"), 'label = f"id:{i}"\n')

    def test_regex_with_backslashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.py"
            original = 'pattern = re.compile(r"\\s+\\d+")\n'
            p.write_text(original, encoding="utf-8")
            reg = build_default_registry(tmp, extended_tools=True)
            r = reg.execute(ToolCall(
                name="edit_file_hashline",
                arguments={
                    "path": "x.py",
                    "line_number": 1,
                    "old_line": original.rstrip("\n"),
                    "new_line": 'pattern = re.compile(r"\\w+")',
                },
                raw="",
            ))
            self.assertTrue(r.success, r.error)
            self.assertIn('r"\\w+"', p.read_text(encoding="utf-8"))


class TestEOLPreservation(unittest.TestCase):
    """Preserve \\r\\n style when the source file uses it."""

    def test_crlf_file_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.txt"
            p.write_bytes(b"a\r\nb\r\nc\r\n")
            reg = build_default_registry(tmp, extended_tools=True)
            r = reg.execute(ToolCall(
                name="edit_file_hashline",
                arguments={
                    "path": "x.txt",
                    "line_number": 2,
                    "old_line": "b",
                    "new_line": "B",
                },
                raw="",
            ))
            self.assertTrue(r.success, r.error)
            # Must still have \r\n separators
            self.assertEqual(p.read_bytes(), b"a\r\nB\r\nc\r\n")

    def test_lf_file_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.txt"
            p.write_bytes(b"a\nb\nc\n")
            reg = build_default_registry(tmp, extended_tools=True)
            r = reg.execute(ToolCall(
                name="edit_file_hashline",
                arguments={
                    "path": "x.txt",
                    "line_number": 2,
                    "old_line": "b",
                    "new_line": "B",
                },
                raw="",
            ))
            self.assertTrue(r.success, r.error)
            self.assertEqual(p.read_bytes(), b"a\nB\nc\n")


if __name__ == "__main__":
    unittest.main()
