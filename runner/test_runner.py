"""Tests for the contract's untrusted-byte paths: python3 -m unittest runner/test_runner.py (stdlib + numpy)."""
import json, os, re, subprocess, sys, tempfile, unittest

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
            # a task name inside a string or a path ('sat', "text", /adult/), not Python's own text=True
            self.assertIsNone(re.search(r"""['"/]""" + re.escape(n) + r"""['"/.]""", src), f"run.py names task {n}")




class ContractEquivalenceCheck(unittest.TestCase):
    """GLM F064: the contract names the equivalence check and must describe it below."""
    def test_equivalence_check_is_described(self):
        text = open(os.path.join(HERE, "CONTRACT.md")).read()
        self.assertNotRegex(text, r"(?i)equivalence check below")
        self.assertRegex(text, r"(?m)^## Equivalence check$")
        section = text.split("## Equivalence check", 1)[1]
        self.assertIn("local/*.pred", section)
        self.assertIn("out-of-band", section)


class ContractFloor(unittest.TestCase):
    """GLM F065: the floor is reported, not enforced; run.py consumes no floor gate."""
    def test_floor_is_reported_not_enforced(self):
        text = open(os.path.join(HERE, "CONTRACT.md")).read()
        self.assertIn("reported, not enforced", text)
        self.assertNotRegex(text, r"must beat to count")


class CounterExit(unittest.TestCase):
    """GLM F051: count.py exits 1 when reps differ, stdout differs between reps or counting could be switched off;
    the runner took that as a count. Only exit 0 counts now."""
    def test_exit_1_is_not_a_count(self):
        import types
        from unittest import mock
        import run as R
        line = b'{"reps": [{"counts": {"Ir": 5}, "stdout_sha256": "x"}], "digest_sha256": "y"}\n'
        fake = types.SimpleNamespace(returncode=1, stdout=line, stderr=b"reps differ")
        c = {"name": "c", "exe": __file__, "weights_path": __file__}
        with tempfile.TemporaryDirectory() as d, mock.patch.object(R.subprocess, "run", return_value=fake), \
                mock.patch.object(R.J, "jail", return_value=fake):
            inp = os.path.join(d, "in.rec"); open(inp, "wb").close()
            with self.assertRaisesRegex(R.Failed, "counter failed \\(exit 1\\)"):
                R.count(c, inp, 0, d)


class ReadmeListsAllCandidates(unittest.TestCase):
    """GLM F066: the Parts section of runner/README.md must list every candidate family in runner/candidates/."""
    def test_readme_lists_all_candidates(self):
        cand_dir = os.path.join(HERE, "candidates")
        families = sorted([d for d in os.listdir(cand_dir)
                           if os.path.isdir(os.path.join(cand_dir, d))])
        self.assertEqual(len(families), 9, f"expected 9 families, found {families}")
        readme = open(os.path.join(HERE, "README.md")).read()
        # Extract the candidates line from the Parts section
        m = re.search(r"`candidates/`:([^.\n]+)", readme)
        self.assertIsNotNone(m, "candidates line not found in README")
        listed = [x.strip(" `,") for x in m.group(1).split() if x.strip(" `,")]
        for f in families:
            self.assertIn(f, listed, f"{f} missing from README candidates list")


class Controls(unittest.TestCase):
    """GLM F052/F053: known-bad candidates kept in runner/controls/, each caught by an end-to-end run on 2026-11-21
    (runner/results/2026-11-21-glm-guards/). They need valgrind and the jail, so CI checks only that they exist."""
    def test_controls_present(self):
        for c in ("ctl-pos", "ctl-n0", "ctl-cnt"):
            self.assertTrue(os.path.exists(os.path.join(HERE, "controls", c, "infer.c")), c)

    def test_duplicate_names_refused(self):
        import run as R
        a = type("A", (), {"task": "optdigits", "data": "/nonexistent", "cand": ["majority", "majority"]})
        with mock_task(R):
            with self.assertRaisesRegex(R.Failed, "unique"):
                R.prepare(a)


class SatWalksatMalformedLiteral(unittest.TestCase):
    """GLM F071: a literal naming a variable >= n is malformed input, not an uninitialized read."""
    def test_literal_out_of_range_is_rejected(self):
        src = os.path.join(HERE, "candidates", "sat-walksat", "infer.c")
        with tempfile.TemporaryDirectory() as d:
            exe = os.path.join(d, "infer")
            subprocess.run([os.environ.get("CC", "cc"), "-O2", "-o", exe, src], check=True)
            weights = os.path.join(d, "weights.bin")
            inp = os.path.join(d, "in.rec")
            open(weights, "wb").write(bytes(8))  # noise 0, max flips 0
            open(inp, "wb").write(records.pack([bytes([1, 1, 0, 2, 1, 1])]))  # n=1, literal 2 > n
            r = subprocess.run([exe, weights, inp, "1"], capture_output=True)
        self.assertNotEqual(r.returncode, 0)


class KnnWiderGrid(unittest.TestCase):
    """K16: the kNN grid goes past 15 and k selection uses the tuning split only."""
    def test_grid_and_tuning_selection(self):
        import importlib.util
        import struct
        import numpy as np

        path = os.path.join(HERE, "candidates", "label-knn", "train.py")
        spec = importlib.util.spec_from_file_location("label_knn_train", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertGreater(max(mod.KS), 15)

        card = {"classes": 2, "input": {"bytes": 1, "types": ["categorical"]}}
        ins = [bytes([i]) for i in range(10)]
        keys = [b"\x00"] * 8 + [b"\x01"] * 2       # tuning split is the last 2 items
        chosen = max(mod.KS)
        calls = []

        def fake_predict(cost, ref_X, ref_y, Q, k, C):
            calls.append((len(ref_X), len(Q), k))
            # only `chosen` is right, and only when scored on the 2-item tuning split
            return np.ones(len(Q), dtype=np.int64) if k == chosen and len(Q) == 2 else np.zeros(len(Q), dtype=np.int64)

        old_predict = mod.predict
        with tempfile.TemporaryDirectory() as d:
            p = lambda n: os.path.join(d, n)
            open(p("card.json"), "w").write(json.dumps(card))
            open(p("train.in"), "wb").write(records.pack(ins))
            open(p("train.key"), "wb").write(records.pack(keys))
            mod.predict = fake_predict
            old_argv = sys.argv
            try:
                sys.argv = ["train.py", p("card.json"), p("train.in"), p("train.key"), p("weights.bin")]
                mod.main()
            finally:
                mod.predict = old_predict
                sys.argv = old_argv
            out = open(p("weights.bin"), "rb").read()

        magic, d, n, C, k = struct.unpack_from("<5i", out)
        self.assertEqual(k, chosen)
        self.assertTrue(calls)
        self.assertTrue(all(r == 8 and q == 2 for r, q, _ in calls))


class KnnInferKGreaterThanN(unittest.TestCase):
    """GLM F072: infer accepts k > n and clamps to the training-set size, like train.py's predict."""
    def test_k_greater_than_n_is_clamped(self):
        import struct
        src = os.path.join(HERE, "candidates", "label-knn", "infer.c")

        def weights(k):
            d, n, C = 1, 2, 2
            return (struct.pack("<5i", 0x314E4E4B, d, n, C, k)
                    + bytes(2 * d * 65536) + bytes([0, 1]) + bytes([0, 1]))

        with tempfile.TemporaryDirectory() as d:
            exe = os.path.join(d, "infer")
            subprocess.run([os.environ.get("CC", "cc"), "-O2", "-o", exe, src], check=True)
            inp = os.path.join(d, "in.rec")
            open(inp, "wb").write(records.pack([b"\x00"]))
            outs = []
            for k in (2, 5):
                w = os.path.join(d, "w%d.bin" % k)
                open(w, "wb").write(weights(k))
                r = subprocess.run([exe, w, inp, "1"], capture_output=True)
                self.assertEqual(r.returncode, 0, r.stderr)
                outs.append(r.stdout)
            self.assertEqual(outs[0], outs[1])


def mock_task(R):
    from unittest import mock
    return mock.patch.object(R, "task", return_value={"name": "t"})



class PlanBeforeScoring(unittest.TestCase):
    """K52: a test-split plan digest mismatch refuses before check() is called."""
    def test_digest_mismatch_refuses_before_check(self):
        import run as R
        from unittest import mock
        args = type("A", (), {
            "data": "/data", "task": "t", "cand": ["c"], "out": "/tmp/out.json",
            "split": "test", "rounds": 1, "seed": 1, "plan": 1, "records": None,
        })
        fake_task = {"name": "t"}
        fake_cand = {"name": "c", "dir": "/nonexistent"}
        with mock.patch.object(R.J, "check", return_value={}), \
             mock.patch.object(R, "task", return_value=fake_task), \
             mock.patch.object(R, "cand", return_value=fake_cand), \
             mock.patch.object(R, "require_plan", side_effect=R.Failed("plan 1 does not freeze these as they are now: c/infer.c")), \
             mock.patch.object(R, "check") as mock_check:
            with self.assertRaisesRegex(R.Failed, "does not freeze"):
                R.cmd_evaluate(args)
            mock_check.assert_not_called()


class MarginalGuard(unittest.TestCase):
    """Audit 2026-11-21 (R4 F2): atk-n0burn burned 240M instructions only when N = 0 and was accepted at -250,724 Ir/item."""
    def test_marginal_problem(self):
        import run
        self.assertIsNone(run.marginal_problem(444_175 + 240_136_328, 240_136_328))
        self.assertIn("no more than", run.marginal_problem(444_175, 240_136_328))   # the audit's numbers
        self.assertIn("no more than", run.marginal_problem(10, 10))
        self.assertIn("not counts", run.marginal_problem(None, 10))

if __name__ == "__main__":
    unittest.main()
