import json, os, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent as A  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
