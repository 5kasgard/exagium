import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import envcheck as E


class T(unittest.TestCase):
    C = {"FAIL_TO_PASS": ["t.py::a", "t.py::b[x - y]"], "PASS_TO_PASS": ["t.py::c"]}

    def test_statuses_reads_summary_lines(self):
        log = "t.py::a PASSED [10%]\n=== short test summary info ===\nPASSED t.py::a\nFAILED t.py::b[x - y] - AssertionError: no\nERROR t.py::c"
        self.assertEqual(E.statuses(log, ["t.py::a", "t.py::b[x - y]", "t.py::c"]),
                         {"t.py::a": "PASSED", "t.py::b[x - y]": "FAILED", "t.py::c": "ERROR"})

    def test_usable(self):
        neg = {"t.py::a": "FAILED", "t.py::c": "PASSED"}
        pos = {"t.py::a": "PASSED", "t.py::b[x - y]": "PASSED", "t.py::c": "PASSED"}
        self.assertEqual(E.judge(self.C, neg, pos), (True, "ok"))

    def test_negative_control_must_fail(self):
        allpass = {"t.py::a": "PASSED", "t.py::b[x - y]": "PASSED", "t.py::c": "PASSED"}
        self.assertFalse(E.judge(self.C, allpass, allpass)[0])

    def test_positive_control_needs_every_test(self):
        neg = {"t.py::a": "FAILED"}
        pos = {"t.py::a": "PASSED", "t.py::b[x - y]": "PASSED"}   # c missing
        self.assertFalse(E.judge(self.C, neg, pos)[0])

    def test_no_f2p(self):
        self.assertFalse(E.judge({"FAIL_TO_PASS": [], "PASS_TO_PASS": []}, {}, {})[0])


if __name__ == "__main__":
    unittest.main()
