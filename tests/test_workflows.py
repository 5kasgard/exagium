"""Every workflow file parses with no duplicate keys (2026-11-20: tier.yml defined 'actions' twice; yaml.safe_load took
the last value silently, GitHub refused the file, and the scheduled tier stopped running)."""
import glob, os, unittest
import yaml


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    keys = set()
    for k, _ in node.value:
        key = loader.construct_object(k, deep=deep)
        if key in keys:
            raise yaml.constructor.ConstructorError(None, None, f"duplicate key {key!r}", k.start_mark)
        keys.add(key)
    return loader.construct_mapping(node, deep)


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


class Workflows(unittest.TestCase):
    def test_no_grep_count_or_echo(self):
        """'grep -c' prints 0 and exits 1 when nothing matches, so '|| echo 0' yields two lines (killed the idle job, 2026-11-20)."""
        import re
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".github", "workflows")
        for f in sorted(glob.glob(os.path.join(root, "*.yml"))):
            for n, line in enumerate(open(f).read().splitlines(), 1):
                self.assertFalse(re.search(r"grep -c[^|]*\|\|\s*echo", line), f"{os.path.basename(f)}:{n}: {line.strip()}")

    def test_no_duplicate_keys(self):
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".github", "workflows")
        files = sorted(glob.glob(os.path.join(root, "*.yml")))
        self.assertTrue(files)
        for f in files:
            with self.subTest(f=os.path.basename(f)):
                yaml.load(open(f), Loader=UniqueKeyLoader)



class MergeAllowlist(unittest.TestCase):
    """Audit 2026-11-21 (R1 F1, A F1, B F1): the tier auto-merged model-written code into files that jobs holding keys run
    (router/probe.py #19, worker/digest.py #15, router/decisions_route.py #14, router/bench6/illusion/illusion.py #42).
    Auto-merge now takes only experiment modules. This fails if a privileged job (any secret, the token, or write
    permission) names, runs or imports a file the tier may merge."""
    ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

    def _tier(self):
        import sys
        sys.path.insert(0, os.path.join(self.ROOT, "worker"))
        import tier
        return tier

    def _privileged_jobs(self):
        import re
        for f in sorted(glob.glob(os.path.join(self.ROOT, ".github", "workflows", "*.yml"))):
            w = yaml.safe_load(open(f))
            top = w.get("permissions")
            for name, job in (w.get("jobs") or {}).items():
                text = yaml.dump(job)
                perms = job.get("permissions", top)
                write = perms is None or perms == "write-all" or (isinstance(perms, dict) and "write" in perms.values())
                if write or re.search(r"secrets\.|github\.token|GH_TOKEN|GITHUB_TOKEN", text):
                    yield os.path.basename(f), name, text

    def _run_or_imported(self, text):
        """Python files a job's text names (any path, 'cd DIR && python3 X.py', 'python -m a.b') and everything those
        import, resolved against every top-level folder and each file's own folder."""
        import re
        tops = [d for d in os.listdir(self.ROOT) if os.path.isdir(os.path.join(self.ROOT, d)) and not d.startswith(".")]
        found = {m for m in re.findall(r"[\w./-]+\.py\b", text)}
        for d, m in re.findall(r"cd\s+([\w./-]+)[^\n]*?python3?\s+(?:-I\s+)?([\w./-]+\.py)", text):
            found.add(os.path.join(d, m))
        for m in re.findall(r"python3?\s+(?:-I\s+)?-m\s+([\w.]+)", text):
            found.add(m.replace(".", "/") + ".py")
        seen = {os.path.normpath(p) for p in found if os.path.isfile(os.path.join(self.ROOT, p))}
        todo = list(seen)
        while todo:
            p = todo.pop()
            src = open(os.path.join(self.ROOT, p), errors="replace").read()
            names = set()
            for line in re.findall(r"^\s*(?:import|from)\s+([^\n#]+)", src, re.M):
                names |= {x.strip().split(" ")[0].split(".")[0] for x in re.split(r"[,\s]+import\s+|,", line) if x.strip()}
            for n in names:
                for d in [os.path.dirname(p)] + tops:
                    c = os.path.normpath(os.path.join(d, n + ".py"))
                    if os.path.isfile(os.path.join(self.ROOT, c)) and c not in seen:
                        seen.add(c); todo.append(c)
        return seen

    def test_no_privileged_job_runs_a_mergeable_file(self):
        tier = self._tier()
        jobs = list(self._privileged_jobs())
        ran = set().union(*(self._run_or_imported(t) for _, _, t in jobs))
        self.assertIn("worker/split.py", ran)   # controls: the scan finds what keyed jobs really run
        self.assertIn("router/probe.py", ran)
        self.assertIn("transfer/tasks.py", ran)   # via tasks/build.py in the scoring jobs (gate review 2026-11-21)
        self.assertEqual(sorted(p for p in ran if tier.merge_allowed(p)), [])
        # and no privileged job mentions experiments at all, apart from three that only name the folder: the record job
        # (copies verdict.json, runs no experiment code), the idle job (dispatch_experiments.py checks that a file exists),
        # anchor-test (record/experiments/anchors/, another folder). Gate review 2026-11-21: loops, PYTHONPATH, -s, and
        # variables escaped every narrower pattern.
        known = {("experiment.yml", "record"), ("worker.yml", "idle"), ("anchor-test.yml", "try")}
        for wf, name, text in jobs:
            if (wf, name) not in known:
                self.assertNotRegex(text, r"\bexperiments\b", f"{wf}:{name}")

    def test_known_cases(self):
        tier = self._tier()
        for p in ("router/probe.py", "worker/digest.py", "worker/classify.py", "worker/test_worker.py", "facts/check.py",
                  "router/bench6/illusion/illusion.py", "runner/proposals/judge.py", "runner/jail.py", ".github/x.yml",
                  "AGENTS.md", "README.md", "runner/CONTRACT.md", "router/bench/tasks/T1.md", "runner/proposals/PLAN.md",
                  "transfer/tasks.py", "transfer/meter.py", "runner/candidates/label-knn/train.py", "ledger/inbox/x.json",
                  "experiments/../worker/split.py", "experiments/run_one.py", "experiments/json.py", "experiments/yaml.py", "experiments/a.py\n",
                  "Experiments/a.py", "experiments/sub/a.py", ".agents/a.md", "experiments/A.py", ""):
            self.assertFalse(tier.merge_allowed(p), repr(p))
        for p in ("experiments/queue_drift.py", "experiments/test_experiments.py", "experiments/new_one.py"):
            self.assertTrue(tier.merge_allowed(p), p)

    def test_renames_and_removals_go_to_a_person(self):
        tier = self._tier()
        item = {"code": {"files": ["experiments/x.py"]}}
        f = {"filename": "experiments/x.py", "additions": 1, "deletions": 0}
        self.assertTrue(tier.merge_decision(item, [dict(f, status="modified")], "approve")[0])
        self.assertFalse(tier.merge_decision(item, [dict(f, status="renamed", previous_filename="worker/split.py")], "approve")[0])
        self.assertFalse(tier.merge_decision(item, [dict(f, status="removed")], "approve")[0])

if __name__ == "__main__":
    unittest.main()

