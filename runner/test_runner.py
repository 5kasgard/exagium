"""Tests for the contract's untrusted-byte paths: python3 -m unittest runner/test_runner.py (stdlib + numpy)."""
import json, os, subprocess, sys, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import records  # noqa: E402

SAT_CHECK = os.path.join(REPO, "tasks", "sat", "check.py")
LABEL_CHECK = os.path.join(HERE, "checkers", "label.py")


def run_check(checker, ins, keys, out_bytes):
    with tempfile.TemporaryDirectory() as d:
        p = lambda n: os.path.join(d, n)
        records.write(p("in"), ins); records.write(p("key"), keys)
        open(p("out"), "wb").write(out_bytes)
        r = subprocess.run([sys.executable, "-I", checker, p("in"), p("key"), p("out")], capture_output=True, text=True)
    return r.returncode, [json.loads(x) for x in r.stdout.splitlines()]


def sat_instance(n, clauses):
    lits = [l & 0xFF for c in clauses for l in c]
    return bytes([n, len(clauses) % 256, len(clauses) // 256] + lits)


class Records(unittest.TestCase):
    def test_roundtrip(self):
        recs = [b"", b"a", bytes(range(256)) * 3]
        self.assertEqual(records.unpack(records.pack(recs)), recs)

    def test_rejects(self):
        good = records.pack([b"ab", b"c"])
        for bad, kw in ((good[:-1], {}), (good + b"\x01", {}), (good, {"max_len": 1}), (good, {"count": 3}),
                        (good, {"count": 1})):
            with self.assertRaises(ValueError):
                records.unpack(bad, **kw)


class LabelChecker(unittest.TestCase):
    def test_scores(self):
        rc, s = run_check(LABEL_CHECK, [b"x", b"y"], [b"\x01", b"\x02"], records.pack([b"\x01", b"\x03"]))
        self.assertEqual((rc, [x["score"] for x in s]), (0, [1, 0]))

    def test_malformed(self):
        for out in (records.pack([b"\x01"]), records.pack([b"\x01", b"\x02\x02"]), b"\x05\x00"):
            rc, _ = run_check(LABEL_CHECK, [b"x", b"y"], [b"\x01", b"\x02"], out)
            self.assertEqual(rc, 1, out)


class SatChecker(unittest.TestCase):
    inst = sat_instance(3, [[1, 2, -3], [-1, 2, 3]])

    def test_solved_and_not(self):
        rc, s = run_check(SAT_CHECK, [self.inst] * 3, [b""] * 3,
                          records.pack([bytes([0, 1, 0]), bytes([1, 0, 0]), bytes([1, 0, 1])]))
        self.assertEqual(rc, 0)
        self.assertEqual([x["score"] for x in s], [1, 0, 1])
        self.assertEqual(s[1]["satisfied"], 0.5)

    def test_malformed_item_scores_zero(self):
        rc, s = run_check(SAT_CHECK, [self.inst] * 2, [b""] * 2, records.pack([bytes([0, 1]), bytes([0, 2, 0])]))
        self.assertEqual(rc, 0)
        self.assertEqual([x["malformed"] for x in s], [True, True])
        self.assertEqual([x["score"] for x in s], [0, 0])

    def test_wrong_count(self):
        rc, _ = run_check(SAT_CHECK, [self.inst] * 2, [b""] * 2, records.pack([bytes([0, 1, 0])]))
        self.assertEqual(rc, 1)


class NoTaskCodeInRunner(unittest.TestCase):
    def test_runner_names_no_task(self):
        src = open(os.path.join(HERE, "run.py")).read()
        names = [n for n in os.listdir(os.path.join(REPO, "tasks")) if os.path.isdir(os.path.join(REPO, "tasks", n))]
        self.assertTrue(names)
        for n in names:
            self.assertNotIn(f'"{n}"', src, f"run.py names task {n}")


if __name__ == "__main__":
    unittest.main()
