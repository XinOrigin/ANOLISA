"""Regression tests for native unittest and TAP report evidence verification."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from native_reports import verify_case


class NativeReportVerificationTests(unittest.TestCase):
    def test_unittest_rejects_standalone_ok_or_multiple_summaries(self) -> None:
        case = {"name": "unit", "tests": 2, "report": "unittest"}
        verify_case(
            case,
            "OK\nFAILED to connect, retrying\nOK (cached)\n..\n"
            "----------------------------------------------------------------------\n"
            "Ran 2 tests in 0.1s\n\nOK\n",
        )
        for output in (
            "OK\n----------------------------------------------------------------------\nRan 2 tests in 0.1s\n\nOK (skipped=1)\n",
            "OK\n----------------------------------------------------------------------\nRan 2 tests in 0.1s\n\nOK (expected failures=1)\n",
            "OK\n----------------------------------------------------------------------\nRan 2 tests in 0.1s\n\nFAILED (failures=1)\n",
            "Ran 2 tests in 0.1s\n\nOK (skipped=1)\nOK\n",
            "Ran 2 tests in 0.1s\nOK\nOK (skipped=1)\n",
            "Ran 2 tests in 0.1s\nOK\nFAILED (failures=1)\n",
            "Ran 2 tests in 0.1s\n\nOK\nRan 2 tests in 0.1s\n\nFAILED (failures=1)\n",
            "Ran 2 tests in 0.1s\n\nOK\nRan 2 tests in 0.1s\n\nOK\n",
        ):
            with self.subTest(output=output), self.assertRaises(ValueError):
                verify_case(case, output)

    def test_tap_rejects_duplicate_or_overwritten_summary_counts(self) -> None:
        case = {"name": "tap-suite", "tests": 2, "report": "tap"}
        passing = "# tests 2\n# pass 2\n# fail 0\n# cancelled 0\n# skipped 0\n# todo 0\n"
        verify_case(case, passing)
        for output in (
            "# tests 2\n# pass 2\n# fail 1\n# cancelled 0\n# skipped 0\n# todo 0\n# fail 0\n",
            "# tests 2\n# pass 2\n# fail 0\n# cancelled 0\n# skipped 1\n# todo 0\n# skipped 0\n",
            "# tests 2\n# pass 1\n# fail 1\n# cancelled 0\n# skipped 0\n# todo 0\n" + passing,
            passing + passing,
        ):
            with self.subTest(output=output), self.assertRaises(ValueError):
                verify_case(case, output)


if __name__ == "__main__":
    unittest.main()
