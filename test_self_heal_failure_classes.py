"""Tests for the extended self_heal failure classifier (Task 8, 2026-05-19).

Covers the 5 new classes added on top of the existing 8:
- IMPORT_ERROR
- ASSERTION_FAILURE
- TYPE_ERROR
- ATTRIBUTE_ERROR
- KEY_INDEX_ERROR
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Optional

from engine.self_heal import (
    ASSERTION_FAILURE,
    ATTRIBUTE_ERROR,
    CWD_FAILURE,
    IMPORT_ERROR,
    KEY_INDEX_ERROR,
    MISSING_IMPORT,
    NAME_NOT_DEFINED,
    PARSE_ERROR,
    STALE_ANCHOR,
    SYNTAX_ERROR,
    TRACEBACK,
    TYPE_ERROR,
    TOOL_REJECTED,
    _PER_CLASS_HINT,
    classify_failure,
    diagnose_message,
)


@dataclass
class _R:
    """Minimal ToolResult stand-in."""
    name: str = "x"
    success: bool = False
    error: Optional[str] = None
    output: Optional[str] = None


class TestImportError(unittest.TestCase):
    def test_module_not_found(self):
        r = _R(error="Traceback ... ModuleNotFoundError: No module named 'requests'")
        self.assertEqual(classify_failure(r), IMPORT_ERROR)

    def test_import_error(self):
        r = _R(error="ImportError: cannot import name 'foo' from 'bar'")
        self.assertEqual(classify_failure(r), IMPORT_ERROR)

    def test_no_module_named_phrase_alone(self):
        r = _R(error="ERROR  No module named 'baz' on stdout")
        self.assertEqual(classify_failure(r), IMPORT_ERROR)

    def test_import_wins_over_name(self):
        # The message contains "not" and a name, but the class must be IMPORT
        r = _R(error="ModuleNotFoundError: No module named 'undefined_module'")
        self.assertEqual(classify_failure(r), IMPORT_ERROR)


class TestAssertionFailure(unittest.TestCase):
    def test_assertion_error(self):
        r = _R(error="Traceback ... AssertionError: expected 5 got 3")
        self.assertEqual(classify_failure(r), ASSERTION_FAILURE)

    def test_assert_keyword_in_error(self):
        r = _R(error="E       assert 3 == 5")
        self.assertEqual(classify_failure(r), ASSERTION_FAILURE)


class TestTypeError(unittest.TestCase):
    def test_type_error(self):
        r = _R(error="Traceback ... TypeError: 'NoneType' object is not subscriptable")
        self.assertEqual(classify_failure(r), TYPE_ERROR)

    def test_missing_positional_arg(self):
        r = _R(error="TypeError: foo() missing 1 required positional argument: 'x'")
        self.assertEqual(classify_failure(r), TYPE_ERROR)


class TestAttributeError(unittest.TestCase):
    def test_attribute_error(self):
        r = _R(error="AttributeError: 'NoneType' object has no attribute 'split'")
        self.assertEqual(classify_failure(r), ATTRIBUTE_ERROR)

    def test_has_no_attribute_phrase_alone(self):
        r = _R(error="has no attribute 'foo' in scope")
        self.assertEqual(classify_failure(r), ATTRIBUTE_ERROR)

    def test_attribute_wins_over_type(self):
        # error includes both phrases; AttributeError wins
        r = _R(error="AttributeError: ... TypeError in framework helper")
        self.assertEqual(classify_failure(r), ATTRIBUTE_ERROR)


class TestKeyIndexError(unittest.TestCase):
    def test_key_error(self):
        r = _R(error="KeyError: 'username'")
        self.assertEqual(classify_failure(r), KEY_INDEX_ERROR)

    def test_index_error(self):
        r = _R(error="IndexError: list index out of range")
        self.assertEqual(classify_failure(r), KEY_INDEX_ERROR)


class TestExistingClassesStillWork(unittest.TestCase):
    """Regression — the 8 pre-existing classes must still classify."""

    def test_syntax_error_still(self):
        self.assertEqual(
            classify_failure(_R(error="SyntaxError: invalid syntax")),
            SYNTAX_ERROR,
        )

    def test_stale_anchor_still(self):
        self.assertEqual(
            classify_failure(_R(error="old_string not found in file")),
            STALE_ANCHOR,
        )

    def test_name_not_defined_still(self):
        self.assertEqual(
            classify_failure(_R(error="NameError: name 'x' is not defined")),
            NAME_NOT_DEFINED,
        )

    def test_missing_import_via_output_still(self):
        self.assertEqual(
            classify_failure(_R(success=True, output="auto_verify: foo.py references undefined names: bar")),
            MISSING_IMPORT,
        )

    def test_cwd_failure_still(self):
        self.assertEqual(
            classify_failure(_R(error="No such file or directory: 'x'")),
            CWD_FAILURE,
        )

    def test_parse_error_still(self):
        self.assertEqual(
            classify_failure(_R(name="parse_error", error="bad")),
            PARSE_ERROR,
        )

    def test_generic_traceback_still(self):
        self.assertEqual(
            classify_failure(_R(error='Traceback ... File "x.py", line 3, in foo')),
            TRACEBACK,
        )

    def test_tool_rejected_generic_still(self):
        self.assertEqual(
            classify_failure(_R(error="something happened")),
            TOOL_REJECTED,
        )


class TestSuccessReturnsNone(unittest.TestCase):
    def test_success_no_undefined(self):
        self.assertIsNone(classify_failure(_R(success=True, error=None)))


class TestPerClassHints(unittest.TestCase):
    """Every classifier output must have a tailored hint, or fall back to
    TOOL_REJECTED. Confirm the new classes don't fall through."""

    def test_new_classes_have_hints(self):
        for cls in (IMPORT_ERROR, ASSERTION_FAILURE, TYPE_ERROR, ATTRIBUTE_ERROR, KEY_INDEX_ERROR):
            self.assertIn(cls, _PER_CLASS_HINT, f"missing hint for {cls}")

    def test_diagnose_message_renders_new_classes(self):
        for cls in (IMPORT_ERROR, ASSERTION_FAILURE, TYPE_ERROR, ATTRIBUTE_ERROR, KEY_INDEX_ERROR):
            msg = diagnose_message(cls, attempts=2)
            self.assertIn("PAUSE", msg)
            self.assertIn(cls, msg)
            # The class-specific hint body should be present
            self.assertIn(_PER_CLASS_HINT[cls][:30], msg)


if __name__ == "__main__":
    unittest.main()
