"""Code job for the queue worker (owner 2026-11-20: free and cheap models do code work, not only docs and research).

python3 worker/code_job.py context ITEM_JSON REPO OUT_DIR   # no keys: the bundle (item, the files it may change, its test)
python3 worker/code_job.py draft OUT_DIR                    # keys: the coder proposes edits, a reviewer of another family judges
python3 worker/code_job.py test REPO OUT_DIR                # NO SECRETS, read-only token: apply the edits, run the tests
python3 worker/code_job.py apply REPO OUT_DIR               # no keys: apply again and land as a PR only if the tests passed

An item opts in with "code": {"files": ["path.py", ...], "test": "path/test_x.py"}; every file is listed in "touches".
Edits are exact search-and-replace blocks: "old" must occur exactly once in its file. Model text is never run where a
key is: the edited code runs only in the test job, which has no secrets and a read-only token. Code never lands on main
directly: a passing change becomes a PR (the next tier or the owner merges), a failing one is released as failed.
Refused paths: workflows, the ledger and queue machinery, anything with secrets in its name.
"""
import json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "router"))
import doc_job as D  # noqa: E402  parse_json, write_calls

MAX_LINES = 600
# the sandbox that runs model-written code is never the code job's to change (PR #13 edited runner/jail.py, 2026-11-20)
REFUSE = re.compile(r"^\.github/|^worker/(ledger|queue_ops|transitions|code_job)\.py$|^audit/|secret|token|credential|(^|/)\.git(/|$)"
                    r"|^runner/(jail|limits|count)\.|^loop/|seccomp|landlock|sandbox", re.I)
TEST = re.compile(r"^[\w./-]+/test_[\w-]+\.py$")
NEW_EXT = (".py", ".md", ".json", ".c", ".h", ".txt")   # files an item may create (code.new); owner 2026-11-21: build, not only edit


def new_ok(repo, p):
    """A file an item may create: allowed, not there yet, in a folder that exists, of a plain source or text type."""
    return allowed(p) and p.endswith(NEW_EXT) and not os.path.exists(os.path.join(repo, p)) \
        and os.path.isdir(os.path.join(repo, os.path.dirname(p) or "."))
CI_TESTS = ["worker/test_worker.py", "worker/test_transitions.py", "runner/proposals/test_proposals.py", "facts/test_check.py",
            "runner/proposals/confirm/test_decide.py", "router/test_pick.py", "experiments/test_experiments.py"]

PROMPT = """You change code in a repository to do what one queue item asks. Change as little as possible.

ITEM {id}: {title}
NEXT STEP: {next}

FILES YOU MAY CHANGE (complete contents; an empty one is a new file you create):
{files}

FILES TO READ FOR CONTEXT (do not edit):
{context}

OTHER RELATED PARTS OF THE REPOSITORY (signatures only; from worker/repo_index.py):
{index}

TEST FILE (you may add tests to it; existing tests must keep passing): {test}
{test_src}

Reply with one or more edit blocks in exactly this form, and nothing else:

<<<<<<< SEARCH path/to/file.py
exact lines copied from the file (they must occur once)
=======
the replacement lines
>>>>>>> REPLACE

To add a new function, SEARCH for a line it should follow and repeat that line at the start of the replacement.
To write a new (empty) file, leave the SEARCH part empty: the SEARCH line, then ======= on the next line.
Keep the whole change under 80 changed lines (added plus removed): larger changes are not merged without a person.
Change the code the item is about, not only its test: the test shows the change works.
Finish with one line: NOTE: what you changed and why."""

REVIEW = """You review a proposed code change. Approve only if it does what the item asks, changes nothing else, adds no
network calls, secrets, file deletions or subprocess calls the item did not ask for, and keeps existing behaviour
unless the item asks to change it.

ITEM {id}: {title}
NEXT STEP: {next}

DIFF:
{diff}

Reply with JSON only: {{"verdict": "approve" or "reject", "reason": "one sentence"}}"""


def allowed(path):
    """F010: the path is checked as written AND normalised, so './.github/x' or 'worker//ledger.py' cannot slip past REFUSE."""
    if not isinstance(path, str) or path.startswith("/") or ".." in path.split("/") or "\\" in path:
        return False
    norm = os.path.normpath(path)
    return norm == path and not REFUSE.search(norm)


def context(item_path, repo, out):
    item = json.load(open(item_path))["item"]
    c = item.get("code") or {}
    files, test, new = c.get("files") or [], c.get("test", ""), set(c.get("new") or [])
    bad = [p for p in files if not allowed(p) or p not in item.get("touches", [])
           or not (os.path.isfile(os.path.join(repo, p)) or (p in new and new_ok(repo, p)))]
    test_new = test in new and TEST.fullmatch(test or "") and new_ok(repo, test)   # a new tests file (splitter gap, 2026-11-21)
    if bad or not files or not TEST.fullmatch(test) or not (os.path.isfile(os.path.join(repo, test)) or test_new):
        raise SystemExit(f"code item refused (files must exist, be listed in touches, not be refused paths; test must be "
                         f"an existing tests file): files {bad or files}, test {test!r}")
    # the item's test file is editable too: sub-items say "add a test", and the coder could not (PRs #10-#12, 2026-11-20)
    files = files + ([test] if test not in files else [])
    ctx = [p for p in (c.get("context") or [])[:2] if allowed(p) and os.path.isfile(os.path.join(repo, p)) and p not in files]
    src = {p: open(os.path.join(repo, p), encoding="utf-8").read() if os.path.isfile(os.path.join(repo, p)) else ""
           for p in files}   # a new file starts empty
    ctx_src = {p: open(os.path.join(repo, p), encoding="utf-8").read()[:20000] for p in ctx}
    if any(s.count("\n") > MAX_LINES for p, s in src.items() if p != test):
        raise SystemExit(f"a file is over {MAX_LINES} lines: split the item (worker-sized work only)")
    tsrc = open(os.path.join(repo, test), encoding="utf-8").read() if os.path.isfile(os.path.join(repo, test)) else ""
    os.makedirs(out, exist_ok=True)
    json.dump({"item": item, "files": src, "test": test,
               "prompt": PROMPT.format(id=item["id"], title=item["title"], next=item["next"][:2000], test=test,
                                       files="\n\n".join(f"--- {p} ---\n{s}" for p, s in src.items() if p != test),
                                       test_src=tsrc[-12000:],
                                       context="\n\n".join(f"--- {p} (read only) ---\n{s}" for p, s in ctx_src.items()) or "(none)",
                                       index=related(repo, item, set(files) | set(ctx)))},
              open(os.path.join(out, "bundle.json"), "w"), indent=1)
    return 0


def related(repo, item, skip, k=5):
    """Signatures of the modules most related to the item, beyond the files it already shows (owner 2026-11-21: give each
    job the slice it needs). Never fails the job: no index is better than no draft."""
    try:
        import repo_index as RI
        ms = [m for m in RI.rank(RI.build(repo), item["title"] + " " + item.get("next", ""), k + len(skip)) if m["path"] not in skip][:k]
        return RI.show(ms, max_parts=15) or "(none)"
    except Exception as e:
        return f"(index unavailable: {type(e).__name__})"


HEAD = re.compile(r"^<<<<<<< SEARCH[ \t]+(\S+)[ \t]*$")
SEP = re.compile(r"^=======[ \t]*$")
END = re.compile(r">>>>>>> REPLACE[ \t]*$")


def blocks_in(text):
    """(path, old, new) for each search/replace block, read line by line so no block can reach into the next. The
    regex this replaces let an empty SEARCH (a new file) run on to the NEXT block's =======: the new file got the test's
    replacement and the test edit vanished (13 of 39 'no edits' code runs 2026-11-20/21, e.g. K44 run 44953039259;
    no new-file item had landed). Returns (blocks, cut): cut lists the path of each block that ended before its REPLACE
    line (or whose text holds a line that looks like a block start), so the caller reports it instead of dropping it."""
    out, cut, cur, part = [], [], None, None
    for line in text.replace("\r\n", "\n").split("\n"):
        if cur is None:
            m = HEAD.match(line)
            if m:
                cur, part = (m.group(1), [], []), 1
            continue
        if part == 1 and SEP.match(line):
            part = 2
        elif part == 2 and END.search(line):
            head = line[:END.search(line).start()]
            out.append((cur[0], "\n".join(cur[1]), "\n".join(cur[2] + ([head] if head else []))))
            cur = None
        elif HEAD.match(line):   # a block cut off before its REPLACE is reported as cut, and this line starts a new one
            cut.append(cur[0])
            cur, part = (HEAD.match(line).group(1), [], []), 1
        else:
            cur[part].append(line)
    if cur is not None:
        cut.append(cur[0])
    return out, cut


def parse_reply(text):
    """Edits from a reply: search/replace blocks (code survives them; inside JSON strings it broke the JSON on 2 of 3
    live code jobs, 2026-11-20), or the older JSON form. Returns {"edits": [...], "note": ...} or None."""
    if not text:
        return None
    blocks, cut = blocks_in(text)
    if blocks:   # a reply with no whole block (cut off by an output limit) falls through, so the slot tries the next model
        note = re.search(r"^NOTE:\s*(.+)$", text, re.M)
        return {"edits": [{"path": p, "old": o, "new": n} for p, o, n in blocks], "note": note.group(1) if note else "",
                "cut": cut}
    j = D.parse_json(text)
    return j if isinstance(j, dict) else None


def check_edits(files, edits):
    ok, why = [], []
    if not isinstance(edits, list) or not edits:
        return [], ["no edits"]
    for e in edits:
        if not isinstance(e, dict) or not all(isinstance(e.get(k), str) for k in ("path", "old", "new")):
            why.append(f"malformed edit {str(e)[:100]}"); continue
        if e["path"] not in files:
            why.append(f"{e['path']}: not a file this item may change"); continue
        # a new (empty) file takes the replacement as its whole text, whatever the model put in SEARCH: 4 of 4 new-file
        # items on 2026-11-21 (K29, K35, K37, K40) failed with "'old' occurs 0 times" because the coder searched for text
        n = 1 if files[e["path"]] == "" else files[e["path"]].count(e["old"]) if e["old"] else 0
        if n != 1:
            why.append(f"{e['path']}: 'old' occurs {n} times (must be exactly once)"); continue
        if e["old"] == e["new"]:
            why.append(f"{e['path']}: no change"); continue
        ok.append(e)
    return ok, why


def apply_edits(files, edits):
    """Sequential, and each edit's "old" must still occur exactly once after the edits before it (F010: overlapping edits
    in one file could misapply); a conflicting edit raises, which the callers treat as no usable change."""
    new = dict(files)
    for e in edits:
        if new[e["path"]] == "" and files[e["path"]] == "":   # writing a new file (its SEARCH text is ignored)
            new[e["path"]] = e["new"] if e["new"].endswith("\n") else e["new"] + "\n"
            continue
        if not e["old"] or new[e["path"]].count(e["old"]) != 1:
            raise ValueError(f"{e['path']}: an earlier edit changed the text this edit replaces")
        new[e["path"]] = new[e["path"]].replace(e["old"], e["new"], 1)
    return new


def static_problems(files, new):
    """Problems a reviewer or the tests would find, found by reading the changed Python (nothing is run, so this is safe
    in the job that holds keys): syntax errors, and names used that nothing in the file binds and are not builtins.
    Missing imports were the commonest rejection on 2026-11-20/21 (K5 json, K7 re, G2 os/inspect/sys)."""
    import ast, builtins
    out = []
    for p, s in (new or {}).items():
        if not p.endswith(".py") or s == files.get(p):
            continue
        try:
            tree = ast.parse(s)
        except SyntaxError as e:
            out.append(f"{p}: syntax error at line {e.lineno}: {e.msg}")
            continue
        bound, used = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}, {}
        for n in ast.walk(tree):
            if isinstance(n, ast.Name):
                (bound.add(n.id) if isinstance(n.ctx, (ast.Store, ast.Del)) else used.setdefault(n.id, n.lineno))
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(n.name)
            elif isinstance(n, ast.arg):
                bound.add(n.arg)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                if any(a.name == "*" for a in n.names):
                    bound.add("*")
                bound.update((a.asname or a.name).split(".")[0] for a in n.names)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                bound.add(n.name)
            elif isinstance(n, (ast.Global, ast.Nonlocal)):
                bound.update(n.names)
            elif isinstance(n, ast.MatchAs) and n.name:
                bound.add(n.name)
        if "*" not in bound:
            out += [f"{p}: name '{k}' is used (line {ln}) but never imported or defined" for k, ln in sorted(used.items(), key=lambda kv: kv[1])
                    if k not in bound]
    return out


def arm(item_id):
    """Speculative drafts, measured (owner 2026-11-21: test every scaling shape where it could fit): items whose id hashes
    odd get a second draft from the next coder in the chain; the test job keeps whichever approved draft passes."""
    import hashlib
    return "two" if int(hashlib.sha256(str(item_id).encode()).hexdigest(), 16) % 2 else "one"


REPAIR = """{prompt}

YOUR PREVIOUS REPLY:
{reply}

IT HAS THESE PROBLEMS (found by reading the code; nothing was run):
{problems}

Reply again with the complete corrected set of edit blocks, in the same form, and the NOTE line."""


def draft(out):
    import engines, slots
    engines.TIMEOUT = 600
    b = json.load(open(os.path.join(out, "bundle.json")))
    item, calls = b["item"], []

    def ask(pool, model, prompt):
        r = slots.call(engines, pool, model, prompt)
        calls.append(r)
        return r
    obj = lambda t: (lambda j: j if isinstance(j, dict) else None)(D.parse_json(t))
    try:
        proposal, d, tries = slots.run("coder", b["prompt"], ask, parse_reply)
        fail = []
        if proposal is None:
            fail.append(("quota: " if all(t["status"] == 429 for t in tries) else "") + "coder gave no usable reply")
            proposal = {"edits": []}
        def checked(prop):
            e, w = check_edits(b["files"], prop.get("edits"))
            try:
                n = apply_edits(b["files"], e) if e else None
            except ValueError as x:
                n, e, w = None, [], w + [str(x)]
            only_test = bool(e) and {x["path"] for x in e} <= {b.get("test")} and any(p != b.get("test") for p in b["files"])
            no_test = bool(e) and b.get("test") in b["files"] and b.get("test") not in {x["path"] for x in e} \
                and not b.get("test", "").startswith("experiments/")   # K28, K39: the reviewer rejects a change with no test
            return e, w, n, (w if not e else []) + [
                f"{p}: a block ended before its >>>>>>> REPLACE line (or its text holds a line starting '<<<<<<< SEARCH'), "
                f"so it was not used; send every block whole" for p in prop.get("cut") or []] + static_problems(b["files"], n) + (
                [f"the edits change only the test file {b.get('test')}; the item needs the change itself in "
                 f"{', '.join(p for p in b['files'] if p != b.get('test'))}, with the test showing it"] if only_test else []) + (
                [f"the edits add no test: add one to {b.get('test')} that fails on the current code and passes with the change"]
                if no_test else [])
        edits, why, new, problems = checked(proposal)
        repaired = None
        if problems and (proposal.get("edits") or proposal.get("cut")):   # one repair round with the problems written in (owner 2026-11-21: code lands least)
            text = (d or {}).get("text") or ""
            fixed, d2, t2 = slots.run("coder", REPAIR.format(prompt=b["prompt"], reply=text[-12000:], problems="\n".join(problems[:20])),
                                      ask, parse_reply)
            tries += t2
            if fixed is not None:
                e2, w2, n2, p2 = checked(fixed)
                repaired = {"before": problems, "after": p2}
                if e2 and len(p2) < len(problems):
                    proposal, edits, why, new, problems = fixed, e2, w2, n2, p2
        review = None
        if edits and problems:   # still broken after the repair: not worth a review call, and not a change to land
            why = why + problems
            edits = []
        if edits:
            dd = D.diff(b["files"], new)
            review, _, _ = slots.run("code_reviewer", REVIEW.format(id=item["id"], title=item["title"], next=item["next"][:1500], diff=dd[:20000]), ask, obj)
            if review is None:
                fail.append("reviewer gave no usable reply")
        if not edits and proposal.get("edits") and problems:
            proposal = dict(proposal, edits=[])   # nothing that fails the static check reaches the test or land jobs
        alt = None
        if arm(item["id"]) == "two":
            chain = [f for f in slots.SLOTS["coder"] if f[0] != "openrouter_paid" or os.environ.get("PAID_FALLBACK") == "1"]
            other = [f for f in chain if not tries or (f[0], f[1]) != (tries[0].get("pool"), tries[0].get("model"))][:1]
            if other:
                try:
                    r2 = ask(other[0][0], other[0][1], b["prompt"])
                except Exception as x:   # the second draft is optional: its failure never costs the first
                    r2 = {"ok": False, "error": f"{type(x).__name__}: {str(x)[:200]}"}
                p2 = parse_reply(r2.get("text")) if r2.get("ok") else None
                e2, w2, n2, s2 = checked(p2) if p2 else ([], [], None, ["no usable reply"])
                rv2 = None
                if e2 and not s2:
                    rv2, _, _ = slots.run("code_reviewer", REVIEW.format(id=item["id"], title=item["title"], next=item["next"][:1500],
                                                                         diff=D.diff(b["files"], n2)[:20000]), ask, obj)
                alt = {"filler": list(other[0]), "proposal": p2 if e2 and not s2 else {"edits": []}, "review": rv2, "static": s2}
        json.dump({"proposal": proposal, "review": review, "pre_check": why, "model_fail": fail, "coder_tries": tries,
                   "static": problems, "repair": repaired, "arm": arm(item["id"]), "alt": alt},
                  open(os.path.join(out, "draft.json"), "w"), indent=1)
        print(f"coder edits={len(edits)} refused={len(why)} review={(review or {}).get('verdict')}")
        return 0
    finally:
        D.write_calls(out, calls)


def test(repo, out):
    """Runs where no secret is: apply the accepted edits to the checkout and run the item's test, then the CI suite. With
    a second draft (arm "two"), an approved second draft is tested when the first did not pass and was not approved, and
    the one that passes is chosen ("chosen" in test.json; the land job applies that one)."""
    b = json.load(open(os.path.join(out, "bundle.json")))
    d = json.load(open(os.path.join(out, "draft.json")))
    res = _test_one(repo, b, d["proposal"], d.get("model_fail"))
    res["chosen"] = "first"
    alt = d.get("alt") or {}
    first_ok = res["passed"] and (d.get("review") or {}).get("verdict") == "approve"
    if not first_ok and (alt.get("review") or {}).get("verdict") == "approve" and alt.get("proposal", {}).get("edits"):
        for p, s in b["files"].items():   # back to the bundle's files before the second draft
            path = os.path.join(repo, p)
            if s == "" and os.path.exists(path):
                os.remove(path)
            elif s != "":
                open(path, "w", encoding="utf-8").write(s)
        r2 = _test_one(repo, b, alt["proposal"], None)
        if r2["passed"]:
            res = dict(r2, chosen="second", first=res)
        else:
            res["second"] = r2
    json.dump(res, open(os.path.join(out, "test.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("runs", "first", "second")}), [(x["test"], x["rc"]) for x in res["runs"]])
    return 0


def _test_one(repo, b, proposal, model_fail):
    edits, why = check_edits(b["files"], proposal.get("edits"))
    res = {"passed": False, "edits": len(edits), "why": why, "runs": []}
    try:
        applied = apply_edits(b["files"], edits) if edits else None
    except ValueError as e:
        applied, res["why"] = None, why + [str(e)]
    if applied and not model_fail:
        for p, s in applied.items():
            open(os.path.join(repo, p), "w", encoding="utf-8").write(s)
        ok = True
        for t in [b["test"]] + [x for x in CI_TESTS if x != b["test"]]:
            r = subprocess.run([sys.executable, "-m", "unittest", t], cwd=repo, capture_output=True, text=True, timeout=600,
                               env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))   # no stale .pyc between two drafts of one size
            res["runs"].append({"test": t, "rc": r.returncode, "tail": (r.stdout + r.stderr)[-1500:]})
            ok = ok and r.returncode == 0
        res["passed"] = ok
    return res


def apply(repo, out):
    """In the land job (no keys): re-check, apply, and give a result the transition table knows."""
    b = json.load(open(os.path.join(out, "bundle.json")))
    d = json.load(open(os.path.join(out, "draft.json")))
    t = json.load(open(os.path.join(out, "test.json"))) if os.path.exists(os.path.join(out, "test.json")) else {}
    if t.get("chosen") == "second" and d.get("alt"):   # the second draft passed where the first did not
        d = dict(d, proposal=d["alt"]["proposal"], review=d["alt"]["review"])
    # defence in depth for refused paths and the item id (audit 2026-11-21, B F2): the bundle carries its own item, so
    # this cannot catch a whole forged bundle; the land job's hash check against the draft job's outputs does that
    it = b.get("item") or {}
    c = it.get("code") or {}
    own = set(c.get("files") or []) | ({c["test"]} if c.get("test") else set())
    bad = [p for p in b.get("files", {}) if not allowed(p) or p not in own or (p != c.get("test") and p not in it.get("touches", []))]
    if bad or not b.get("files") or (os.environ.get("ITEM") and str(it.get("id")) != os.environ["ITEM"]):
        res = {"result": "guard_blocked", "why": [f"bundle does not match the item: {bad or it.get('id')}"[:300]], "edits": 0}
        json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=1)
        print(json.dumps(res)); return 0
    files = {p: open(os.path.join(repo, p), encoding="utf-8").read() if os.path.isfile(os.path.join(repo, p)) else ""
             for p in b["files"]}   # a file the item creates is still absent here (and must be: else it is stale)
    if files != b["files"]:
        res = {"result": "stale", "why": ["a file changed since the bundle was made"], "edits": 0}
    elif d.get("model_fail"):
        quota = all(str(m).startswith("quota: ") for m in d["model_fail"])
        res = {"result": "quota" if quota else "failed_call", "why": d["model_fail"], "edits": 0}
    else:
        edits, why = check_edits(files, d["proposal"].get("edits"))
        verdict = (d.get("review") or {}).get("verdict")
        if not edits:   # Gemini review 2026-11-20 F009: was "pr", which parked the item in review with nothing to review
            res = {"result": "no_edits", "why": why or ["no edit proposed"], "edits": 0}
        elif verdict not in ("approve", "reject"):   # GLM F025: no review happened; a model failure, retried
            res = {"result": "failed_call", "why": [f"reviewer gave no verdict: {verdict!r}"[:300]], "edits": 0}
        elif verdict != "approve":   # F010: only an explicit approval goes on (a missing or odd verdict used to pass)
            res = {"result": "rejected", "why": [f"reviewer: {verdict}: {(d.get('review') or {}).get('reason', '')}"[:300]], "edits": 0}
        elif not t.get("passed"):
            res = {"result": "tests_failed", "why": [f"{x['test']} rc {x['rc']}" for x in t.get("runs", []) if x["rc"]] or ["tests did not run"], "edits": 0}
        else:
            try:
                new = apply_edits(files, edits)
            except ValueError as e:
                res = {"result": "no_edits", "why": [str(e)], "edits": 0}
                json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=1)
                return 0
            for p, s in new.items():
                open(os.path.join(repo, p), "w", encoding="utf-8").write(s)
            open(os.path.join(out, "change.diff"), "w").write(D.diff(files, new))
            # code always lands as a PR: "pr" result with the reviewer's verdict and the test record for the next tier
            res = {"result": "pr", "proposed": len(edits), "why": ["code change: tests passed; a PR for the next tier"],
                   "edits": len(edits), "review": d.get("review"), "tests": [(x["test"], x["rc"]) for x in t["runs"]], "code": True,
                   "arm": d.get("arm"), "chosen": t.get("chosen")}
    json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=1)
    print(json.dumps(res)[:600])
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit({"context": lambda: context(a[1], a[2], a[3]), "draft": lambda: draft(a[1]),
              "test": lambda: test(a[1], a[2]), "apply": lambda: apply(a[1], a[2])}[a[0]]())
