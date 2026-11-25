"""P28 pilot step 3: one agent attempt inside the task's own image, graded by envcheck.attempt (sealed, Q82).

python3 -I agent.py extract HARNESS_IMAGE DEST            # copy bench-4's OpenHands venv and its Python out of the image
python3 -I agent.py run CANDS STATEMENTS ID AGENT OUTDIR HARNESS_DIR GW PORT SEAL_SH [MODEL CAP WALL_S MAXIT]

AGENT: gold  - no model: the gold patch is applied as if an agent wrote it (dry run of everything but the model)
       mock  - bench-4's base harness (OpenHands SDK) through bench-4's meter to a local mock model (no key)
       model - the same through the meter to OpenRouter (key file in BENCH4_KEY_FILE, read and deleted by the meter)

Order inside one container: start it with network; warm the task's own tests on the unfixed code (test_patch copied in,
applied, reversed and deleted, so the agent never sees it); seal (host iptables: containers reach only the meter's port;
gold disconnects the network instead); write the issue text; run the agent; take its diff from /testbed; remove the
container; grade the diff in a fresh sealed container (envcheck.attempt). Always writes OUTDIR/agent-ID-AGENT.json.
"""
import json, os, re, shutil, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import envcheck as E  # noqa: E402

BENCH4 = os.path.normpath(os.path.join(HERE, "..", "..", "bench4"))
LIMITS = ["--cpus", "2", "--memory", "6g", "--pids-limit", "2048"]   # bench-4's attempt container
TASK_TAIL = "\n\nThe repository is in /workspace."   # as bench-4 said where the code is, and nothing more
# seal.sh's IPv4 rules, removed by their exact text after the agent's container is gone (the grader seals its own)
UNSEAL = ("iptables -D DOCKER-USER -i docker0 -j REJECT; iptables -D DOCKER-USER -i docker0 -j LOG --log-prefix 'B4DROP ' --log-level 4; "
          "iptables -D INPUT -i docker0 -j REJECT; iptables -D INPUT -i docker0 -j LOG --log-prefix 'B4DROP ' --log-level 4; "
          "iptables -D INPUT -i docker0 -p tcp --dport {port} -j ACCEPT")
WARM = r"""
set -u
[ -f /opt/miniconda3/bin/activate ] && . /opt/miniconda3/bin/activate testbed 2>/dev/null
cd /
git -C /testbed apply /tmp/fw/test_patch.diff >/dev/null 2>&1 || { echo "test_patch did not apply (warm)"; exit 4; }
python -I /tmp/fw/helper.py test >/dev/null 2>&1
git -C /testbed apply -R /tmp/fw/test_patch.diff >/dev/null 2>&1 || { echo "test_patch did not reverse (warm)"; exit 4; }
rm -rf /tmp/fw /testbed/.pytest_cache
[ -z "$(git -C /testbed status --porcelain --untracked-files=no)" ] || { echo "warm-up left tracked changes"; exit 4; }
echo "=====FORGE-WARM====="
"""


def sh(cmd, timeout=600, **k):
    return E.sh(cmd, timeout) if not k else subprocess.run(cmd, timeout=timeout, capture_output=True, text=True, **k)


def extract(image, dest):
    """bench-4's /opt/oh venv points at a uv-managed Python under /root; copy both, to mount at the same paths."""
    rc, py, _ = sh(["docker", "run", "--rm", "--entrypoint", "readlink", image, "-f", "/opt/oh/bin/python"])
    py = py.strip()
    # the whole managed-Python folder: the venv's link goes through a version alias (cpython-3.12-...) to the full version
    # folder (cpython-3.12.16-...), and both live in it (first mock run, job 139143389636: "/opt/oh/bin/python: not found")
    root = os.path.dirname(py.rsplit("/bin/", 1)[0])
    if rc != 0 or not py.startswith("/"):
        raise SystemExit(f"cannot locate the venv's python: {py}")
    cid = sh(["docker", "create", image])[1].strip()
    try:
        mounts = []
        for path in ("/opt/oh", root):
            host = os.path.join(dest, path.strip("/").replace("/", "__"))
            rc, log, _ = sh(["docker", "cp", f"{cid}:{path}", host])
            if rc != 0:
                raise SystemExit(f"docker cp {path}: {log[-300:]}")
            mounts.append([os.path.abspath(host), path])
        json.dump({"image": image, "python": py, "mounts": mounts}, open(os.path.join(dest, "mounts.json"), "w"), indent=1)
        print(json.dumps(mounts))
    finally:
        sh(["docker", "rm", "-f", cid])


def git_diff(name, base):
    """The agent's change against the base commit, new files included (git add -A inside the container)."""
    p = subprocess.run(["docker", "exec", name, "sh", "-c",
                        f"cd /testbed && git -c core.hooksPath=/dev/null add -A >/dev/null 2>&1 && "
                        f"git -c core.hooksPath=/dev/null diff --cached --binary {base}"],
                       capture_output=True, timeout=300)
    return p.returncode, p.stdout


def run(cands, stmts, iid, agent, outdir, hdir, gw, port, seal_sh, model="", cap="80", wall="2100", maxit="1000"):
    c = next(x for x in json.load(open(cands))["candidates"] if x["instance_id"] == iid)
    text = json.load(open(stmts))["statements"][iid]["text"]
    os.makedirs(outdir, exist_ok=True)
    res = {"id": iid, "agent": agent, "model": model if agent == "model" else None, "cap": int(cap), "wall_cap_s": int(wall)}
    name = f"forge-agent-{os.getpid()}-{re.sub('[^a-z0-9]', '-', iid.lower())}"
    work = tempfile.mkdtemp(prefix="forge-agent-", dir=outdir)
    hout = os.path.join(work, "out")
    os.makedirs(hout)
    open(os.path.join(hout, "task.md"), "w").write(text + TASK_TAIL)
    procs, t0 = [], time.time()
    try:
        rc, log, _ = sh(["docker", "pull", "-q", c["docker_image"]], 900)
        if rc != 0:
            return finish(res, outdir, "start_failed", f"image pull: {log[-300:]}", t0)
        vols = ["-v", f"{hout}:/out"]
        if agent != "gold":
            for host, path in json.load(open(os.path.join(hdir, "mounts.json")))["mounts"]:
                vols += ["-v", f"{host}:{path}:ro"]
            vols += ["-v", f"{BENCH4}:/opt/bench4:ro"]
        rc, log, _ = sh(["docker", "run", "-d", "--name", name, *LIMITS, *vols, "--entrypoint", "sleep", c["docker_image"], "infinity"])
        if rc != 0:
            return finish(res, outdir, "start_failed", log[-300:], t0)
        fw = os.path.join(work, "fw")
        os.makedirs(fw)
        open(os.path.join(fw, "test_patch.diff"), "w").write(c["test_patch"])
        json.dump(c["FAIL_TO_PASS"] + c["PASS_TO_PASS"], open(os.path.join(fw, "tests.json"), "w"))
        open(os.path.join(fw, "test_cmd"), "w").write(c["test_cmd"])
        open(os.path.join(fw, "mode"), "w").write("warm")   # the helper reads mode and test_files at import
        json.dump(E.test_files(c["test_patch"]), open(os.path.join(fw, "test_files.json"), "w"))
        open(os.path.join(fw, "helper.py"), "w").write(E.HELPER.replace('"/w/', '"/tmp/fw/'))
        open(os.path.join(fw, "warm.sh"), "w").write(WARM)
        sh(["docker", "cp", fw, f"{name}:/tmp/fw"])
        shutil.rmtree(fw)
        rc, log, s = sh(["docker", "exec", name, "bash", "/tmp/fw/warm.sh"], 900)
        res["warm_s"] = s
        if rc != 0 or "=====FORGE-WARM=====" not in log:
            return finish(res, outdir, "warm_failed", log[-300:], t0)
        if agent == "gold":
            rc, log, _ = sh(["docker", "network", "disconnect", "bridge", name])
        else:
            rc, log, _ = sh(["sudo", "bash", seal_sh, str(port)])
        if rc != 0:
            return finish(res, outdir, "seal_failed", log[-300:], t0)
        menv = None
        if agent == "gold":
            open(os.path.join(hout, "gold.diff"), "w").write(c["patch"])
            rc, log, s = sh(["docker", "exec", name, "git", "-C", "/testbed", "apply", "/out/gold.diff"])
            os.remove(os.path.join(hout, "gold.diff"))
            ending = "exited" if rc == 0 else "start_failed"
        else:
            kf = os.path.join(work, "key")
            if agent == "mock":
                procs.append(subprocess.Popen([sys.executable, "-I", os.path.join(BENCH4, "mock_upstream.py"), "18099",
                                               os.path.join(outdir, f"mock-{iid}.jsonl")]))
                open(kf, "w").write("dummy")
                upstream, model = "http://127.0.0.1:18099/v1", "mock/model"
            else:
                shutil.move(os.environ["BENCH4_KEY_FILE"], kf)
                upstream = "https://openrouter.ai/api/v1"
            menv = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "OPENROUTER_API_KEY_FILE": kf, "BENCH4_UPSTREAM": upstream}
            procs.append(subprocess.Popen([sys.executable, "-I", os.path.join(BENCH4, "meter.py"), gw, str(port),
                                           os.path.join(outdir, f"meter-{iid}.jsonl"), model, str(cap)], env=menv,
                                          stdout=open(os.path.join(outdir, f"meter-{iid}.out"), "w"), stderr=subprocess.STDOUT))
            time.sleep(3)
            # bench-4's seal check, from inside the agent's own container: the meter is reachable, the outside is not
            probe = lambda host, prt: sh(["docker", "exec", name, "timeout", "8", "bash", "-c", f"echo > /dev/tcp/{host}/{prt}"], 30)[0]
            res["seal"] = {"meter": probe(gw, port) == 0, "outside": "reached" if probe("1.1.1.1", 443) == 0 else "blocked"}
            if not res["seal"]["meter"] or res["seal"]["outside"] != "blocked":
                return finish(res, outdir, "seal_failed", json.dumps(res["seal"]), t0)
            # bench-4's own launcher, unchanged: the same environment, iteration limit and output files as bench-4's base arm
            cmd = ["docker", "exec", "-e", f"BENCH4_BASE=http://{gw}:{port}", "-e", f"BENCH4_MODEL={model}",
                   name, "sh", "-c", "ln -sfn /testbed /workspace && sh /opt/bench4/run_in_container.sh openhands /out/task.md /out"]
            rc, log, s = sh(cmd, int(wall))
            ending = "timeout" if rc == "timeout" else "exited"
            try:
                res["harness_rc"] = int(open(os.path.join(hout, "rc")).read().strip())
            except (OSError, ValueError):
                res["harness_rc"] = None
            try:
                res["harness_end"] = json.loads(open(os.path.join(hout, "harness.out")).read().rsplit("@@END@@", 1)[1].splitlines()[0])
            except (OSError, IndexError, ValueError):
                res["harness_end"] = None
            if os.path.exists(os.path.join(outdir, f"meter-{iid}.jsonl")):
                calls = [json.loads(l) for l in open(os.path.join(outdir, f"meter-{iid}.jsonl")) if l.strip()]
                res["calls"] = sum(1 for x in calls if x.get("status") == 200)
                res["refused"] = sum(1 for x in calls if x.get("status") == "refused_cap")   # meter.py logs refusals so, not as 429 (model run 45601064588)
                if res["refused"] and ending == "exited":
                    ending = "call_cap"
        for f in ("harness.out", "harness.err"):   # the harness's own record, kept with the attempt
            if os.path.exists(os.path.join(hout, f)):
                shutil.copyfile(os.path.join(hout, f), os.path.join(outdir, f"agent-{iid}-{agent}-{f}"))
        rc, diff = git_diff(name, c["base_commit"])
        dpath = os.path.join(outdir, f"agent-{iid}-{agent}.diff")
        open(dpath, "wb").write(diff)
        res["diff_bytes"] = len(diff)
        if rc != 0:
            return finish(res, outdir, "diff_failed", "git diff failed", t0)
        sh(["docker", "rm", "-f", name])
        if agent != "gold":
            sh(["sudo", "bash", "-c", UNSEAL.format(port=int(port))])   # the grader's own warm-up needs the network back
        E.attempt(cands, iid, dpath, outdir)
        g = json.load(open(os.path.join(outdir, f"attempt-{iid}-file.json")))
        res.update(resolved=g["resolved"], grade_reason=g["reason"])
        return finish(res, outdir, ending, "", t0)
    finally:
        for p in procs:
            p.terminate()
        sh(["docker", "rm", "-f", name], 60)
        shutil.rmtree(work, ignore_errors=True)


def finish(res, outdir, ending, why, t0):
    res.update(ending=ending, why=why[-300:] if why else None, wall_s=round(time.time() - t0, 1))
    res.setdefault("resolved", None)
    json.dump(res, open(os.path.join(outdir, f"agent-{res['id']}-{res['agent']}.json"), "w"), indent=1)
    print(json.dumps({k: res.get(k) for k in ("id", "agent", "ending", "resolved", "calls", "diff_bytes", "wall_s", "why")}))
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "extract":
        extract(a[1], a[2])
    elif a and a[0] == "run":
        sys.exit(run(*a[1:]))
    else:
        print(__doc__); sys.exit(1)
