"""Tests for record.py's ledger rules (offline; anchors are tested in CI)."""
import io, json, os, shutil, tempfile, unittest, contextlib, hashlib
import record as R

PLAN = {"by": "proposer-agent", "claim": "mlp beats linear", "test": "count.py ...", "decision_rule": "Ir lower by > 2 SE"}


class Ledger(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.d, "inbox"))

    def tearDown(self):
        shutil.rmtree(self.d)

    def run_cmd(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = R.main(["--ledger", self.d, *args])
        return rc, out.getvalue(), err.getvalue()

    def propose(self, name, kind, body):
        with open(os.path.join(self.d, "inbox", name), "w") as f:
            json.dump({"kind": kind, "body": body}, f)

    def entry(self, n):
        return os.path.join(self.d, "entries", f"{n:06d}.json")

    def plan_ref(self, n):
        return {"seq": n, "sha256": hashlib.sha256(open(self.entry(n), "rb").read()).hexdigest()}

    def verdict(self, n, by="verifier-agent", outcome="pass"):
        return {"by": by, "plan": self.plan_ref(n), "outcome": outcome, "evidence": []}

    def test_plan_then_verdict_chains_and_verifies(self):
        self.propose("a.json", "plan", PLAN)
        self.assertEqual(self.run_cmd("ingest")[0], 0)
        self.propose("b.json", "verdict", self.verdict(1))
        self.assertEqual(self.run_cmd("ingest")[0], 0)
        rc, out, _ = self.run_cmd("verify")
        self.assertEqual(rc, 0)
        rep = json.loads(out)
        self.assertEqual((rep["entries"], rep["verdicts"], rep["open_plans"]), (2, 1, []))
        e2 = json.load(open(self.entry(2)))
        self.assertEqual(e2["prev"], self.plan_ref(1)["sha256"])
        self.assertEqual(os.listdir(os.path.join(self.d, "inbox")), [])

    def test_open_and_overdue_plans_are_reported(self):
        self.propose("a.json", "plan", dict(PLAN, verdict_due="2000-01-01"))
        self.propose("b.json", "plan", PLAN)
        self.run_cmd("ingest")
        rep = json.loads(self.run_cmd("verify")[1])
        self.assertEqual((rep["open_plans"], rep["overdue_plans"]), ([1, 2], [1]))

    def _two_entries(self):
        self.propose("a.json", "plan", PLAN)
        self.propose("b.json", "plan", dict(PLAN, claim="second"))
        self.run_cmd("ingest")

    def test_edited_entry_breaks_the_chain(self):
        self._two_entries()
        raw = open(self.entry(1), "rb").read().replace(b"mlp beats linear", b"mlp beats linea!")
        open(self.entry(1), "wb").write(raw)
        rc, _, err = self.run_cmd("verify")
        self.assertEqual(rc, 1)
        self.assertIn("prev doesn't match", err)

    def test_dropped_entry_is_caught(self):
        self._two_entries()
        os.remove(self.entry(1))
        self.assertEqual(self.run_cmd("verify")[0], 1)

    def test_dropped_entry_with_rewritten_chain_is_only_caught_by_anchors(self):
        # Removing entry 1 and rebuilding entry 2 as a fresh first entry gives a valid chain:
        # the chain alone can't show a missing record. That's what anchors and the monitor are for.
        self._two_entries()
        e2 = json.load(open(self.entry(2)))
        os.remove(self.entry(1)); os.remove(self.entry(2))
        e2.update(seq=1, prev=None)
        open(self.entry(1), "wb").write(R.canonical(e2))
        self.assertEqual(self.run_cmd("verify")[0], 0)

    def test_non_canonical_bytes_are_refused(self):
        self._two_entries()
        e = json.load(open(self.entry(2)))
        open(self.entry(2), "w").write(json.dumps(e, indent=1))
        rc, _, err = self.run_cmd("verify")
        self.assertEqual(rc, 1)
        self.assertIn("canonical", err)

    def test_duplicate_keys_are_refused(self):
        self._two_entries()
        raw = open(self.entry(2), "rb").read()
        open(self.entry(2), "wb").write(raw.replace(b'{"body":{', b'{"body":{"by":"x",', 1))
        rc, _, err = self.run_cmd("verify")
        self.assertEqual(rc, 1)
        self.assertIn("duplicate key", err)

    def test_floats_are_refused(self):
        self.propose("a.json", "plan", dict(PLAN, notes=1.5))
        rc, _, err = self.run_cmd("ingest")
        self.assertEqual(rc, 1)
        self.assertIn("float", err)
        self.assertFalse(os.path.exists(self.entry(1)))

    def test_one_verdict_per_plan(self):
        self.propose("a.json", "plan", PLAN)
        self.run_cmd("ingest")
        self.propose("b.json", "verdict", self.verdict(1, outcome="fail"))
        self.run_cmd("ingest")
        self.propose("c.json", "verdict", self.verdict(1, outcome="pass"))
        rc, _, err = self.run_cmd("ingest")
        self.assertEqual(rc, 1)
        self.assertIn("already has a verdict", err)
        self.assertFalse(os.path.exists(self.entry(3)))
        self.assertTrue(os.path.exists(os.path.join(self.d, "inbox", "c.json")))

    def test_verdict_must_come_from_someone_else(self):
        self.propose("a.json", "plan", PLAN)
        self.run_cmd("ingest")
        self.propose("b.json", "verdict", self.verdict(1, by="Proposer-Agent "))
        rc, _, err = self.run_cmd("ingest")
        self.assertEqual(rc, 1)
        self.assertIn("someone else", err)

    def test_verdict_must_name_the_plan_by_hash(self):
        self.propose("a.json", "plan", PLAN)
        self.run_cmd("ingest")
        v = self.verdict(1)
        v["plan"]["sha256"] = "0" * 64
        self.propose("b.json", "verdict", v)
        self.assertEqual(self.run_cmd("ingest")[0], 1)

    def test_canonical_form_matches_jcs_rules(self):
        # Sorted keys, no spaces, UTF-8 unescaped, control chars escaped: the JCS output for these values.
        obj = {"b": "é\n\u0001", "a": [1, True, None], "A": -9007199254740991}
        self.assertEqual(R.canonical(obj), b'{"A":-9007199254740991,"a":[1,true,null],"b":"\xc3\xa9\\n\\u0001"}')
        with self.assertRaises(R.Bad):
            R._check_value({"x": 2**53}, "t")
        with self.assertRaises(R.Bad):
            R._check_value({"é": 1}, "t")

    def test_canonical_form_is_rfc8785(self):
        try:
            import rfc8785
        except ImportError:
            self.skipTest("rfc8785 not installed")
        import random
        rnd = random.Random(7)
        chars = 'aZ0 "\\\n\t\b\f\r\x01\x1f\x7fé€\U0001f600 /'
        def s():
            return "".join(rnd.choice(chars) for _ in range(rnd.randint(0, 6)))
        def v(d=0):
            t = rnd.randint(0, 5 if d < 3 else 2)
            return [lambda: rnd.randint(-R.MAX_INT, R.MAX_INT), s, lambda: rnd.choice([True, False, None]),
                    lambda: [v(d + 1) for _ in range(rnd.randint(0, 3))],
                    lambda: {"k" + s().encode("ascii", "ignore").decode(): v(d + 1) for _ in range(rnd.randint(0, 3))}, s][t]()
        for _ in range(3000):
            o = {"x": v(), "y": v()}
            self.assertEqual(R.canonical(o), rfc8785.dumps(o))

    def test_unknown_files_and_fields_are_refused(self):
        self._two_entries()
        open(os.path.join(self.d, "entries", "notes.txt"), "w").write("x")
        self.assertEqual(self.run_cmd("verify")[0], 1)
        os.remove(os.path.join(self.d, "entries", "notes.txt"))
        self.propose("c.json", "plan", dict(PLAN, secret="x"))
        self.assertEqual(self.run_cmd("ingest")[0], 1)


if __name__ == "__main__":
    unittest.main()
