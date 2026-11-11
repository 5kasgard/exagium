"""Grade one agent attempt by hidden tests (SWE-bench style). Standard library only.

usage: grade.py WORKDIR FIX_TEST_FILE HIDDEN_TEST [PASS_TO_PASS_JSON]

Copies the fix commit's record/test_record.py over the agent's copy (so the agent can't
edit the grading tests), runs every test in it with unittest, and prints one JSON object:
  hidden: "pass" | "fail" | "error" | "missing"
  pass_to_pass_broken: tests that passed on the original code but fail now
  results: {test_id: outcome}
Exit 0 always; the verdict is in the JSON.
"""
import json, os, shutil, subprocess, sys

RUNNER = r'''
import json, sys, unittest
sys.path.insert(0, ".")
suite = unittest.defaultTestLoader.loadTestsFromName("test_record")
out = {}
class R(unittest.TestResult):
    def addSuccess(self, t): out[t.id().split(".")[-1]] = "pass"
    def addFailure(self, t, e): out[t.id().split(".")[-1]] = "fail"
    def addError(self, t, e): out[getattr(t, "id", lambda: str(t))().split(".")[-1]] = "error"
    def addSkip(self, t, r): out[t.id().split(".")[-1]] = "skip"
suite.run(R())
print("@@RESULTS@@" + json.dumps(out))
'''


def run_tests(workdir, fix_test_file, timeout=900):
    rec = os.path.join(workdir, "record")
    shutil.copyfile(fix_test_file, os.path.join(rec, "test_record.py"))
    try:
        p = subprocess.run([sys.executable, "-c", RUNNER], cwd=rec, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"_harness": "timeout"}
    for line in p.stdout.splitlines():
        if line.startswith("@@RESULTS@@"):
            return json.loads(line[len("@@RESULTS@@"):])
    return {"_harness": "no results", "_stderr": p.stderr[-800:]}


def main():
    workdir, fix_test_file, hidden = sys.argv[1:4]
    p2p = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else None
    res = run_tests(workdir, fix_test_file)
    broken = [t for t in (p2p or []) if res.get(t) != "pass"]
    print(json.dumps({"hidden": res.get(hidden, "missing"), "pass_to_pass_broken": broken,
                      "n_tests": len([k for k in res if not k.startswith("_")]), "results": res}))


if __name__ == "__main__":
    main()
