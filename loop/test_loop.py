"""Tests for loop.py. Toy candidates only: no test-set scores are computed here."""
import contextlib, io, json, os, shutil, subprocess, sys, tempfile, unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loop  # noqa: E402

CONST = """#include <stdio.h>
#include <stdlib.h>
int main(int c, char **v){ long n = atol(v[3]); FILE *f = fopen(v[1], "rb"); if(!f) return 2; int k = fgetc(f);
  for(long i=0;i<n;i++) putchar(k); putchar('\\n'); return 0; }
"""
TRAIN = "import sys\nopen(sys.argv[2], 'wb').write(b'%s')\n"


def make(root, name, weights=b"7", train=None, infer=CONST):
    d = os.path.join(root, name)
    os.makedirs(d)
    with open(os.path.join(d, "train.py"), "w") as f:
        f.write(train if train is not None else TRAIN % weights.decode())
    with open(os.path.join(d, "infer.c"), "w") as f:
        f.write(infer)
    with open(os.path.join(d, "weights.bin"), "wb") as f:
        f.write(weights)
    return d


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = loop.main(list(argv))
        except SystemExit as e:
            code = e.code
    return code, out.getvalue() + err.getvalue()


class Loop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_candidate_must_hold_exactly_its_files(self):
        d = make(self.tmp, "a")
        open(os.path.join(d, "test.csv"), "w").close()
        code, msg = run("freeze", d)
        self.assertEqual(code, 1)
        self.assertIn("extra ['test.csv']", msg)

    def test_symlinked_file_is_refused(self):
        d = make(self.tmp, "a")
        os.remove(os.path.join(d, "weights.bin"))
        os.symlink(loop.TASK["test"], os.path.join(d, "weights.bin"))
        self.assertEqual(run("freeze", d)[0], 1)

    def test_freeze_covers_task_harness_counter_and_files(self):
        d = make(self.tmp, "a")
        code, msg = run("freeze", d)
        self.assertEqual(code, 0)
        names = [x["name"] for x in json.loads(msg)]
        for want in ("loop/task/optdigits.tra", "loop/task/optdigits.tes", "loop/loop.py", "counter/count.py"):
            self.assertIn(want, names)
        self.assertEqual(sum(n.endswith(("train.py", "infer.c", "weights.bin")) for n in names), 3)

    def test_reproduce_identical_and_different(self):
        same = make(self.tmp, "same", b"7")
        other = make(self.tmp, "other", b"7", train=TRAIN % "8")
        self.assertEqual(run("reproduce", same)[0], 0)
        code, msg = run("reproduce", other)
        self.assertEqual(code, 1)
        self.assertIn("DIFFERENT", msg)

    def test_training_sees_only_the_training_file(self):
        probe = ("import os, sys\nok = sorted(os.listdir('.')) == ['train.csv', 'train.py']"
                 " and open('train.csv').read() == open(%r).read()\nopen(sys.argv[2], 'wb').write(b'1' if ok else b'0')\n"
                 % loop.TASK["train"])
        d = make(self.tmp, "p", b"1", train=probe)
        self.assertEqual(run("reproduce", d)[0], 0)

    @unittest.skipUnless(shutil.which("unshare") and subprocess.run(["unshare", "-rn", "true"]).returncode == 0,
                         "no user namespaces here")
    def test_no_net_training_has_no_network(self):
        probe = ("import socket, sys\ntry:\n socket.create_connection(('1.1.1.1', 443), timeout=3); r = b'net'\n"
                 "except OSError:\n r = b'none'\nopen(sys.argv[2], 'wb').write(r)\n")
        d = make(self.tmp, "n", b"none", train=probe)
        self.assertEqual(run("reproduce", d, "--no-net")[0], 0)

    def test_test_split_needs_a_plan(self):
        d = make(self.tmp, "a")
        code, msg = run("evaluate", d, "--test", "test")
        self.assertEqual(code, 2)
        self.assertIn("needs --plan", msg)

    def test_test_split_refused_when_plan_does_not_freeze_these_files(self):
        d = make(self.tmp, "a")
        code, msg = run("evaluate", d, "--test", "test", "--plan", "1")  # entry 1 is the record's plan
        self.assertEqual(code, 1)
        self.assertIn("does not freeze", msg)

    def test_malformed_output_fails(self):
        bad = CONST.replace("putchar('\\n');", "")
        d = make(self.tmp, "bad", infer=bad)
        code, msg = run("evaluate", d, "--rounds", "2", "--passes", "1", "--records", os.path.join(self.tmp, "r"))
        self.assertEqual(code, 1)
        self.assertIn("exactly", msg)

    def test_evaluate_constant_classifier_on_dev(self):
        d = make(self.tmp, "seven", b"7")
        out = os.path.join(self.tmp, "s.json")
        code, msg = run("evaluate", d, "--rounds", "2", "--passes", "2", "--records", os.path.join(self.tmp, "r"),
                        "--out", out)
        self.assertEqual(code, 0, msg)
        s = json.load(open(out))
        c = s["candidates"][0]
        dev = loop.split("dev")
        self.assertEqual(c["correct"], sum(r[64] == 7 for r in dev))
        self.assertGreater(c["simulated_Ir_per_image"], 0)
        self.assertEqual(len(c["counter_records"]), 2)
        self.assertEqual(code, 0)
        self.assertEqual(run("evaluate", d, "--rounds", "2", "--passes", "1", "--records",
                             os.path.join(self.tmp, "r"), "--out", out)[0], 3)  # never overwrite a scoreboard

    def test_timing_run_must_agree_with_counted_predictions(self):
        # answers depend on how many images it is asked for: the timing run (N*P) disagrees with the counted run
        flaky = CONST.replace("int k = fgetc(f);", "int k = fgetc(f); if (n > 2000) k = '3';")
        d = make(self.tmp, "flaky", infer=flaky)
        code, msg = run("evaluate", d, "--rounds", "2", "--passes", "3", "--records", os.path.join(self.tmp, "r"))
        self.assertEqual(code, 1)
        self.assertIn("timing run's predictions differ", msg)


if __name__ == "__main__":
    unittest.main()
