import json, os, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent_attempt as A  # noqa: E402
import envcheck as E  # noqa: E402


class Agent(unittest.TestCase):
    def test_warm_hides_the_tests(self):
        """The test_patch reaches the agent's container only for the warm-up, and is reversed and deleted before the agent."""
        w = A.WARM
        self.assertLess(w.index("apply /tmp/fw/test_patch.diff"), w.index("apply -R /tmp/fw/test_patch.diff"))
        self.assertLess(w.index("apply -R"), w.index("rm -rf /tmp/fw"))
        self.assertNotIn("/w/", E.HELPER.replace('"/w/', '"/tmp/fw/'))

    def test_task_text_says_only_where(self):
        self.assertEqual(A.TASK_TAIL.strip(), "The repository is in /workspace.")

    def test_unseal_matches_seal(self):
        seal = open(os.path.join(A.BENCH4, "seal.sh")).read()
        for rule in A.UNSEAL.format(port=8088).split("; "):
            spec = rule.split(" -D ", 1)[1].replace("8088", '"$PORT"')
            self.assertIn("-I " + spec.split(" ", 1)[0] + " 1 " + spec.split(" ", 1)[1], seal.replace("'B4DROP '", '"B4DROP "')
                          .replace('"B4DROP "', "'B4DROP '"))

    def test_statements_frozen_for_every_usable_task(self):
        st = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "statements.json")))["statements"]
        self.assertEqual(sorted(st), sorted(E.USABLE))
        import hashlib
        for v in st.values():
            self.assertEqual(hashlib.sha256(v["text"].encode()).hexdigest(), v["sha256"])


class Grade(unittest.TestCase):
    def test_cap_comes_from_the_meter_log(self):
        import contextlib, io
        import grade_pilot as G
        with tempfile.TemporaryDirectory() as d:
            for i, (n, cap) in enumerate([(80, True), (33, False)]):
                iid = f"t{i}"
                json.dump({"id": iid, "ending": "exited", "wall_s": 100, "wall_cap_s": 2100, "resolved": False, "harness_end": None},
                          open(os.path.join(d, f"agent-{iid}-model.json"), "w"))
                with open(os.path.join(d, f"meter-{iid}.jsonl"), "w") as f:
                    f.write("".join(json.dumps({"status": 200}) + "\n" for _ in range(n)))
                    if cap:
                        f.write(json.dumps({"status": "refused_cap"}) + "\n")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                G.grade(d)
            self.assertIn("t0                                         call_cap", out.getvalue())
            self.assertIn("kept 1 of 2 started", out.getvalue())


if __name__ == "__main__":
    unittest.main()
