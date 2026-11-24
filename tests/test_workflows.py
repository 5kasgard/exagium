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



class TestArtifactApart(unittest.TestCase):
    """Audit 2026-11-21 (B F2, R1 F2): model code runs in worker.yml's test job. Its artifact holds only test.json and is
    unpacked into its own folder, never over the draft job's draft.json and bundle.json."""
    def test_worker_test_artifact(self):
        w = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".github", "workflows", "worker.yml")))
        up = [s for s in w["jobs"]["test"]["steps"] if "upload-artifact" in str(s.get("uses"))]
        self.assertEqual([s["with"]["path"] for s in up], ["tested/test.json"])
        down = [s for s in w["jobs"]["land"]["steps"] if "download-artifact" in str(s.get("uses")) and s["with"]["name"] == "test"]
        self.assertEqual([s["with"]["path"] for s in down], ["tested"])
        self.assertIn("cp tested/test.json job/test.json", yaml.dump(w["jobs"]["land"]))
        self.assertIn("sums", w["jobs"]["draft"]["outputs"])                       # land checks the draft job's own hashes
        self.assertEqual(w["jobs"]["land"]["env"]["DRAFT_SUMS"], "${{ needs.draft.outputs.sums }}")
        self.assertIn('[ "$got" != "$DRAFT_SUMS" ]', open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".github", "workflows", "worker.yml")).read())


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

    def _privileged_jobs(self, root=None):
        """Every job that holds a key or a write token, by parsing the workflow (Z.ai review 2026-11-22 F005: the scan read
        only each job's own text, so a secret set in the workflow-level env, a reusable-workflow call with 'secrets:',
        or a local action holding the token was invisible). Unparsed or unknown forms count as privileged."""
        import re
        root = root or self.ROOT
        key = r"secrets\.|secrets\[|github\.token|GH_TOKEN|GITHUB_TOKEN"
        for f in sorted(glob.glob(os.path.join(root, ".github", "workflows", "*.y*ml"))):
            w = yaml.safe_load(open(f)) or {}
            top = w.get("permissions")
            wenv = yaml.dump(w.get("env") or {})          # workflow-level env reaches every job
            for name, job in (w.get("jobs") or {}).items():
                job = job or {}
                text = yaml.dump(job) + "\n" + wenv + self._local_actions(job, root)
                perms = job.get("permissions", top)
                write = perms is None or perms == "write-all" or (isinstance(perms, dict) and "write" in perms.values())
                reusable = "uses" in job                     # a called workflow: its own jobs may hold anything
                if write or reusable or "secrets" in job or re.search(key, text):
                    yield os.path.basename(f), name, text

    def _local_actions(self, job, root):
        """The text of local composite actions a job uses (uses: ./path), which run in the job with its keys."""
        out = []
        for st in job.get("steps") or []:
            u = str((st or {}).get("uses", ""))
            if u.startswith("./"):
                for fn in ("action.yml", "action.yaml"):
                    p = os.path.join(root, u[2:], fn)
                    if os.path.isfile(p):
                        out.append(open(p, errors="replace").read())
        return "\n".join(out)

    def _run_or_imported(self, text, root=None):
        """Python files a job's text names (any path, 'cd DIR && python3 X.py', 'python -m a.b'), the shell scripts it runs
        and what those name in turn (Z.ai F005: a .sh wrapper hid the Python it ran), and everything those import,
        resolved against every top-level folder and each file's own folder."""
        import re
        root = root or self.ROOT
        tops = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")]

        def named(t, here=""):
            found = set(re.findall(r"[\w./-]+\.py\b", t))
            for d, m in re.findall(r"cd\s+([\w./-]+)[^\n]*?python3?\s+(?:-I\s+)?([\w./-]+\.py)", t):
                found.add(os.path.join(d, m))
            for m in re.findall(r"python3?\s+(?:-I\s+)?-m\s+([\w.]+)", t):
                found.add(m.replace(".", "/") + ".py")
            found |= set(re.findall(r"[\w./-]+\.sh\b", t))
            out = set()
            for x in found:
                for c in (x, os.path.join(here, x)):
                    c = os.path.normpath(c.lstrip("./") if c.startswith("./") else c)
                    if os.path.isfile(os.path.join(root, c)):
                        out.add(c)
            return out

        seen = named(text)
        todo = list(seen)
        while todo:
            p = todo.pop()
            src = open(os.path.join(root, p), errors="replace").read()
            nxt = set()
            if p.endswith(".sh"):
                nxt = named(src, os.path.dirname(p))
            else:
                names = set()
                for line in re.findall(r"^\s*(?:import|from)\s+([^\n#]+)", src, re.M):
                    names |= {x.strip().split(" ")[0].split(".")[0] for x in re.split(r"[,\s]+import\s+|,", line) if x.strip()}
                for n in names:
                    for d in [os.path.dirname(p)] + tops:
                        c = os.path.normpath(os.path.join(d, n + ".py"))
                        if os.path.isfile(os.path.join(root, c)):
                            nxt.add(c)
            for c in nxt - seen:
                seen.add(c); todo.append(c)
        return seen

    def test_guard_controls(self):
        """Known-answer controls for the guard (Z.ai F005): each fixture hides a keyed job that runs experiments/x.py one
        way the old scan missed. The guard must see every one; a keyless read-only job must stay unprivileged."""
        import tempfile, textwrap
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, ".github", "workflows")); os.makedirs(os.path.join(root, "experiments"))
            os.makedirs(os.path.join(root, "scripts")); os.makedirs(os.path.join(root, ".github", "act"))
            open(os.path.join(root, "experiments", "x.py"), "w").write("print(1)\n")
            open(os.path.join(root, "scripts", "wrap.sh"), "w").write("#!/bin/sh\npython3 experiments/x.py\n")
            open(os.path.join(root, ".github", "act", "action.yml"), "w").write(
                "runs:\n  using: composite\n  steps:\n    - run: echo ${{ secrets.K }}\n      shell: bash\n")
            wf = {
                "env.yml": "on: push\npermissions: {contents: read}\nenv:\n  K: ${{ secrets.K }}\njobs:\n  a:\n    runs-on: x\n    steps:\n      - run: python3 experiments/x.py\n",
                "inherit.yml": "on: push\npermissions: {contents: read}\njobs:\n  a:\n    uses: ./.github/workflows/env.yml\n    secrets: inherit\n",
                "wrap.yml": "on: push\npermissions: {contents: read}\njobs:\n  a:\n    runs-on: x\n    env: {K: '${{ secrets.K }}'}\n    steps:\n      - run: sh scripts/wrap.sh\n",
                "action.yml": "on: push\npermissions: {contents: read}\njobs:\n  a:\n    runs-on: x\n    steps:\n      - uses: ./.github/act\n      - run: python3 experiments/x.py\n",
                "clean.yml": "on: push\npermissions: {contents: read}\njobs:\n  a:\n    runs-on: x\n    steps:\n      - run: python3 experiments/x.py\n",
            }
            for n, t in wf.items():
                open(os.path.join(root, ".github", "workflows", n), "w").write(textwrap.dedent(t))
            jobs = {wfn: t for wfn, _, t in self._privileged_jobs(root)}
            self.assertEqual(sorted(jobs), ["action.yml", "env.yml", "inherit.yml", "wrap.yml"])
            self.assertIn("experiments/x.py", self._run_or_imported(jobs["wrap.yml"], root))   # through the .sh wrapper
            self.assertIn("experiments/x.py", self._run_or_imported(jobs["env.yml"], root))

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
class PauseStopsSpending(unittest.TestCase):
    """Audit 2026-11-21 (R1 F4, B F4): worker/PAUSE stopped the claim but not the idle job's spending (generator, Claude
    splitter, classifier, experiments), and router.yml and wiring.yml never looked at it."""
    WF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".github", "workflows")

    def test_every_keyed_self_starting_workflow_checks_pause(self):
        import re
        missing = []
        for f in sorted(glob.glob(os.path.join(self.WF, "*.yml"))):
            w, text = yaml.safe_load(open(f)), open(f).read()
            on = w.get(True) or w.get("on") or {}
            auto = {"schedule", "push", "workflow_run", "pull_request"} & set(on if isinstance(on, (dict, list)) else [on])
            keyed = re.search(r"secrets(\.(?!GITHUB_TOKEN)|\[)|secrets:\s*inherit", text)   # every form (review 2026-11-22)
            checks = re.search(r"-f worker/PAUSE|contents/worker/PAUSE", text) and re.search(r"paused", text)   # a real check, not a comment
            if auto and keyed and not checks:
                missing.append(os.path.basename(f))
        self.assertEqual(missing, [])

    def test_a_paused_worker_run_does_not_start_the_idle_job(self):
        w = yaml.safe_load(open(os.path.join(self.WF, "worker.yml")))
        step = [s for s in w["jobs"]["claim"]["steps"] if "worker/PAUSE" in str(s.get("run", ""))][0]["run"]
        self.assertIn('STOPPED=paused', step)
        idle = w["jobs"]["idle"]["if"]
        self.assertIn("needs.claim.outputs.stopped == ''", idle)   # so a stop reason (now: paused) keeps the idle job off


class PaidAllowanceMatchesReservation(unittest.TestCase):
    """Audit 2026-11-21 (R3 F1): a step that runs a paid caller sets ROUTER_PAID_CALLS equal to the calls it reserves;
    without it every paid request is refused and the job stops quietly (review 2026-11-22)."""
    def test_allowance_equals_reservation(self):
        import re
        wf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".github", "workflows")
        seen = 0
        for f in ("tier.yml", "make-work.yml", "worker.yml"):
            for job in yaml.safe_load(open(os.path.join(wf, f)))["jobs"].values():
                for st in job.get("steps", []):
                    run = st.get("run", "")
                    if not re.search(r"worker/(tier|split)\.py", run):
                        continue
                    seen += 1
                    allow = dict(x.split(":") for x in str((st.get("env") or {}).get("ROUTER_PAID_CALLS", "")).split(",") if ":" in x)
                    res = re.findall(r'ledger\.py reserve [^\n]*?"anthropic:(\d+):', run)
                    self.assertTrue(res, f"{f}: no anthropic reservation next to {run[:60]!r}")
                    self.assertEqual(allow.get("anthropic"), res[0], f"{f}: allowance {allow} vs reservation {res}")
        self.assertEqual(seen, 3)   # tier, the make-work split, the idle split

if __name__ == "__main__":
    unittest.main()

