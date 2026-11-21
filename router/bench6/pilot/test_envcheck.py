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

    def test_resolved_needs_every_test_passed(self):
        allp = {"t.py::a": "PASSED", "t.py::b[x - y]": "PASSED", "t.py::c": "PASSED"}
        self.assertEqual(E.resolved(self.C, allp), (True, "ok"))
        self.assertFalse(E.resolved(self.C, dict(allp, **{"t.py::c": "FAILED"}))[0])   # a PASS_TO_PASS broken
        self.assertFalse(E.resolved(self.C, {"t.py::a": "PASSED", "t.py::c": "PASSED"})[0])   # a test not reported
        self.assertFalse(E.resolved({"FAIL_TO_PASS": [], "PASS_TO_PASS": ["t.py::c"]}, {"t.py::c": "PASSED"})[0])

    def test_test_files_from_patch(self):
        tp = "diff --git a/t/x.py b/t/x.py\n--- a/t/x.py\n+++ b/t/x.py\n@@\n+x\ndiff --git a/t/n.py b/t/n.py\n--- /dev/null\n+++ b/t/n.py\n"
        self.assertEqual(E.test_files(tp), ["t/n.py", "t/x.py"])

    def test_grade_attempts_known_answer(self):
        import json, tempfile
        with tempfile.TemporaryDirectory() as d:
            for i, r in enumerate((True, True, None)):
                json.dump({"id": f"t{i}", "diff": "gold", "resolved": r, "reason": "", "seconds": 1},
                          open(os.path.join(d, f"attempt-t{i}.json"), "w"))
            self.assertEqual(E.grade_attempts(d, "resolved"), 1)   # a grader that did not finish is a miss
            os.remove(os.path.join(d, "attempt-t2.json"))
            self.assertEqual(E.grade_attempts(d, "resolved"), 0)
            self.assertEqual(E.grade_attempts(d, "unresolved"), 1)
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(E.grade_attempts(d, "resolved"), 1)   # nothing graded is not a pass


if __name__ == "__main__":
    unittest.main()
