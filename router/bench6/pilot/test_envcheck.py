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
        tp = ("diff --git a/t/x.py b/t/x.py\n--- a/t/x.py\n+++ b/t/x.py\n@@\n+x\n"
              "diff --git a/t/n.py b/t/n.py\n--- /dev/null\n+++ b/t/n.py\n"
              "diff --git a/t/gone.py b/t/gone.py\ndeleted file mode 100644\n--- a/t/gone.py\n+++ /dev/null\n"
              "diff --git a/t/old.py b/t/new.py\nsimilarity index 100%\nrename from t/old.py\nrename to t/new.py\n"
              "diff --git \"a/t/t\\303\\251st.py\" \"b/t/t\\303\\251st.py\"\n--- \"a/t/t\\303\\251st.py\"\n+++ \"b/t/t\\303\\251st.py\"\n")
        self.assertEqual(E.test_files(tp), ["t/gone.py", "t/n.py", "t/new.py", "t/old.py", "t/testést.py".replace("testést", "tést"), "t/x.py"])

    def test_summary_ignores_lines_outside_pytest_block(self):
        log = ("PASSED t.py::a\n=== short test summary info ===\nFAILED t.py::a - x\n=== 1 failed in 0.1s ===\n"
               "PASSED t.py::a\n")   # the last line is printed by the code under test at exit
        self.assertEqual(E.statuses(E.summary(log), ["t.py::a"]), {"t.py::a": "FAILED"})
        self.assertEqual(E.summary("no summary here"), "")

    def test_grade_attempts_known_answer(self):
        import json, tempfile

        def put(d, i, mode, r):
            json.dump({"id": i, "diff": mode, "resolved": r, "reason": "", "seconds": 1},
                      open(os.path.join(d, f"attempt-{i}-{mode}.json"), "w"))
        with tempfile.TemporaryDirectory() as d:
            for i in E.USABLE:
                for m, want in E.MODES.items():
                    put(d, i, m, want)
            self.assertEqual(E.grade_attempts(d), 0)
            put(d, E.USABLE[0], "cheat", True)          # a cheat read resolved: a miss
            self.assertEqual(E.grade_attempts(d), 1)
            put(d, E.USABLE[0], "cheat", None)          # the grader did not reach the tests: a miss
            self.assertEqual(E.grade_attempts(d), 1)
            put(d, E.USABLE[0], "cheat", False)
            os.remove(os.path.join(d, f"attempt-{E.USABLE[1]}-gold.json"))   # a run missing: not a pass
            self.assertEqual(E.grade_attempts(d), 1)
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(E.grade_attempts(d), 1)   # nothing graded is not a pass


class Seal(unittest.TestCase):
    """Q82: sealed containers; the network is cut after the warm-up and before the attempt's diff is applied."""

    def test_seal_flags(self):
        for f in ("--cap-drop", "ALL", "no-new-privileges", "--pids-limit", "--memory", "--cpus"):
            self.assertIn(f, E.SEAL)

    def test_reset_restores_more_and_skips_hooks(self):
        for b in ("pyproject.toml", "sitecustomize.py", "usercustomize.py", '.endswith(".pth")'):
            self.assertIn(b, E.HELPER)
        reset = E.HELPER.split('step == "reset"', 1)[1].split("elif step", 1)[0]
        self.assertEqual(reset.count("git("), reset.count('git("-c", "core.hooksPath=/dev/null"'))

    def _run(self, fail=None):
        calls = []
        def fake(cmd, timeout):
            calls.append(cmd)
            if fail and fail in cmd:
                return 1, "", 0.0
            return 0, "=====FORGE-WARM=====\n" if "/w/warm.sh" in cmd else "", 0.0
        orig = E.sh
        E.sh = fake
        try:
            rc = E.sealed_run("img", "/tmp/w", "forge-x", lambda: 100)[0]
        finally:
            E.sh = orig
        return rc, calls

    def test_disconnect_before_diff_is_applied(self):
        rc, calls = self._run()
        kinds = [c[1] if c[1] != "exec" else c[-1] for c in calls]
        self.assertEqual(kinds, ["run", "/w/warm.sh", "network", "/w/inner.sh", "rm"])
        self.assertNotIn("none", calls[0])                               # the warm-up has network
        self.assertTrue(set(E.SEAL) <= set(calls[0]))                    # every other seal flag from the start
        self.assertIn("disconnect", calls[2])
        self.assertIn("attempt.diff", E.INNER_ATTEMPT)                   # inner.sh is what applies the diff
        self.assertNotIn("attempt.diff", E.INNER_WARM)                   # the warm-up never sees it
        self.assertLess(E.INNER_ATTEMPT.index("FORGE-ONLINE"), E.INNER_ATTEMPT.index("attempt.diff"))

    def test_failed_disconnect_never_grades(self):
        rc, calls = self._run(fail="disconnect")
        self.assertEqual(rc, "sealfail")
        self.assertNotIn("/w/inner.sh", [c[-1] for c in calls])
        self.assertEqual(calls[-1][:3], ["docker", "rm", "-f"])        # the container is always removed


if __name__ == "__main__":
    unittest.main()
