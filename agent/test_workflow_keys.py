import json, os, re, unittest
HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "models.json")))
WF = os.path.join(HERE, "..", ".github", "workflows", "opencode.yml")


class WorkflowMatchesConfig(unittest.TestCase):
    """Every provider in models.json has exactly one step, which holds exactly its secret under its env name, and no step
    holds more than one secret (the per-step key rule)."""

    def test_one_step_per_provider_one_secret_each(self):
        import yaml
        steps = yaml.safe_load(open(WF))["jobs"]["answer"]["steps"]
        keyed = {}
        for s in steps:
            secrets = [(k, v) for k, v in (s.get("env") or {}).items() if "secrets." in str(v)]
            self.assertLessEqual(len(secrets), 1, s.get("name"))
            m = re.search(r"provider == '([a-z0-9-]+)'", str(s.get("if", "")))
            if secrets:
                self.assertIsNotNone(m, s.get("name"))
                keyed.setdefault(m.group(1), []).append(secrets[0])
        for prov, c in CFG["providers"].items():
            want = [(c["env"], "${{ secrets.%s }}" % (c.get("secret") or c["paid_secret"]))]
            self.assertEqual(keyed.get(prov), want, prov)
        self.assertEqual(set(keyed), set(CFG["providers"]))
        post = yaml.safe_load(open(WF))["jobs"]["post"]
        self.assertNotIn("secrets.", str(post).replace("github.token", ""))   # the post job holds no provider key


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
