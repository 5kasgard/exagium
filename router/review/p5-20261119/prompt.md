You review a GitHub Actions workflow and scripts in a public repo. A free language model writes code (train.py in Python, infer.c in C) that the repo then compiles and runs to score it. Threat model: the model's reply is untrusted and may be hostile (prompt-injected or adversarial). Secrets: NVIDIA_API_KEY (in the propose job only); the GITHUB_TOKEN (write in reserve and record jobs, read in the others). Generated code must run only in the score job, which has no secrets and a read-only token, and there only inside a sandbox (runner/run.py trains and compiles inside bubblewrap with rlimits and seccomp, network unshared). Find any way the reply could: reach a secret or a write token; run outside the sandbox; inject into shell commands, file paths or git operations; write outside its own folder; poison the proposals branch, the ledger or main; or fake its own score. Also find logic errors in judge.py's validity rule versus PLAN.md. Numbered findings only, each with severity (major/minor), file and line, the concrete attack or bug, and the fix. If none, say "no findings". No praise.

## .github/workflows/propose.yml
```
# Queue P5: free models propose candidates; rules in runner/proposals/PLAN.md. Manual for now.
# reserve (ledger, no key) -> propose (key, runs no generated code) -> score (no secrets, read-only, generated code runs
# only in the jail) -> record (no key; writes the proposals branch and the ledger).
name: propose
on:
  workflow_dispatch:
    inputs:
      n:
        description: "proposals in this run (1-3)"
        default: "3"
      start:
        description: "task rotation start (0 optdigits, 1 text, 2 adult)"
        default: "0"
permissions: {}
concurrency:
  group: propose
  cancel-in-progress: false
env:
  RUN: propose-${{ github.run_id }}
jobs:
  reserve:
    runs-on: ubuntu-24.04
    timeout-minutes: 10
    permissions:
      contents: write   # the ledger on worker-runs
    outputs:
      n: ${{ steps.res.outputs.n }}
      start: ${{ steps.res.outputs.start }}
    steps:
      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262  # v4.4.0  # zizmor: ignore[artipacked] this job pushes the ledger with the checkout token and uploads no artifact
      - name: reserve one free call per proposal
        id: res
        env:
          N_IN: ${{ github.event.inputs.n }}
          START_IN: ${{ github.event.inputs.start }}
        run: |
          case "$N_IN" in 1|2|3) ;; *) echo "n must be 1, 2 or 3"; exit 1 ;; esac
          case "$START_IN" in 0|1|2) ;; *) echo "start must be 0, 1 or 2"; exit 1 ;; esac
          if [ -f worker/PAUSE ]; then echo "paused by worker/PAUSE"; exit 1; fi
          git config --global user.name exagium-worker; git config --global user.email worker@exagium.invalid
          git fetch -q origin worker-runs && git worktree add -q --detach "$RUNNER_TEMP/r" FETCH_HEAD
          python3 -I worker/ledger.py reserve "$RUNNER_TEMP/r" "$RUN" "nvidia:$N_IN"
          { echo "n=$N_IN"; echo "start=$START_IN"; } >> "$GITHUB_OUTPUT"
  propose:
    needs: reserve
    runs-on: ubuntu-24.04
    timeout-minutes: 45
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262  # v4.4.0
        with:
          persist-credentials: false
      - name: one call per proposal (Nemotron 3 Ultra, NVIDIA NIM free)
        env:
          NVIDIA_API_KEY: ${{ secrets.NVIDIA_API_KEY }}
          N: ${{ needs.reserve.outputs.n }}
          START: ${{ needs.reserve.outputs.start }}
        run: python3 -I runner/proposals/propose.py out "$N" "$START"
      - uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02  # v4.6.2
        if: ${{ !cancelled() }}
        with:
          name: replies
          path: out/
  score:
    needs: [reserve, propose]
    runs-on: ubuntu-24.04
    timeout-minutes: 330
    permissions:
      contents: read
    env:
      N: ${{ needs.reserve.outputs.n }}
    steps:
      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262  # v4.4.0
        with:
          persist-credentials: false
      - uses: actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093  # v4.3.0
        with:
          name: replies
          path: out
      - name: tools
        run: |
          sudo apt-get update -qq && sudo apt-get install -y -qq valgrind bubblewrap
          sudo sysctl -q -w kernel.apparmor_restrict_unprivileged_userns=0
          python3 -m venv "$RUNNER_TEMP/venv" && "$RUNNER_TEMP/venv/bin/pip" install -q numpy==2.5.3
          python3 -m unittest runner/proposals/test_proposals.py
      - name: extract, reproduce and score each proposal on dev, in the jail
        run: |
          PY="$RUNNER_TEMP/venv/bin/python"
          for k in $(seq 0 $((N - 1))); do
            task=$(cat "out/$k/task"); case "$task" in optdigits|text|adult) ;; *) echo "bad task"; exit 1 ;; esac
            name="prop-${GITHUB_RUN_ID}-$k"
            st=$(python3 -I runner/proposals/extract.py "out/$k/reply.md" "runner/candidates/$name" | tail -1) || true
            echo "$name ($task): $st"
            if [ "$st" = extracted ]; then
              [ -d "$RUNNER_TEMP/built/$task" ] || "$PY" -I tasks/build.py "$RUNNER_TEMP/built" --cache "$RUNNER_TEMP/dl" --only "$task"
              "$PY" runner/run.py reproduce --data "$RUNNER_TEMP/built" --task "$task" --cand "$name" \
                --out "out/$k/reproduce.json" > "out/$k/reproduce.txt" 2>&1 || true
              "$PY" runner/run.py evaluate --data "$RUNNER_TEMP/built" --task "$task" --cand majority label-linear "$name" \
                --split dev --rounds 3 --out "out/$k/dev.json" > "out/$k/dev.txt" 2>&1 || true
              tail -5 "out/$k/reproduce.txt" "out/$k/dev.txt"
            fi
            python3 -I runner/proposals/judge.py one "$name" "$st" "out/$k/reproduce.json" "out/$k/dev.json" "out/$k/result.json"
          done
      - uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02  # v4.6.2
        if: ${{ !cancelled() }}
        with:
          name: scored
          path: out/
  record:
    needs: [reserve, propose, score]
    if: ${{ always() && needs.reserve.result == 'success' }}
    runs-on: ubuntu-24.04
    timeout-minutes: 15
    permissions:
      contents: write   # the proposals branch and the ledger
    steps:
      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262  # v4.4.0  # zizmor: ignore[artipacked] this job pushes with the checkout token and uploads no artifact
      - uses: actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093  # v4.3.0
        continue-on-error: true
        with:
          name: scored
          path: scored
      - uses: actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093  # v4.3.0
        continue-on-error: true
        with:
          name: replies
          path: replies
      - name: ledger, proposals branch and leaderboard
        env:
          N: ${{ needs.reserve.outputs.n }}
        run: |
          git config --global user.name exagium-worker; git config --global user.email worker@exagium.invalid
          git fetch -q origin worker-runs && git worktree add -q --detach "$RUNNER_TEMP/r" FETCH_HEAD
          if [ -f replies/calls.jsonl ]; then python3 -I worker/ledger.py record "$RUNNER_TEMP/r" "$RUN" replies/calls.jsonl || echo "ledger record exit $?"; fi
          src=scored; [ -d scored ] || src=replies
          for t in 1 2 3; do
            if git ls-remote --exit-code origin proposals >/dev/null; then
              git fetch -q origin proposals && git worktree add -q --detach "$RUNNER_TEMP/p$t" FETCH_HEAD
            else
              git worktree add -q --orphan -b "proposals-$t" "$RUNNER_TEMP/p$t"
            fi
            P="$RUNNER_TEMP/p$t"
            for k in $(seq 0 $((N - 1))); do
              d="$P/proposals/prop-${GITHUB_RUN_ID}-$k"; mkdir -p "$d"
              for f in task prompt.md reply.md reproduce.txt dev.txt result.json; do [ -f "$src/$k/$f" ] && cp "$src/$k/$f" "$d/" || true; done
              [ -f "$d/result.json" ] || python3 -I runner/proposals/judge.py one "prop-${GITHUB_RUN_ID}-$k" "not scored" x x "$d/result.json"
            done
            python3 -I runner/proposals/judge.py board "$P/proposals"
            git -C "$P" add -A && git -C "$P" commit -qm "proposals: $RUN" && git -C "$P" push -q origin HEAD:proposals && break
            echo "push failed, retrying"; sleep 5
          done
          cat "$P/proposals/LEADERBOARD.md" >> "$GITHUB_STEP_SUMMARY"
```

## runner/proposals/extract.py
```
"""Take a proposal's files out of a model reply (queue P5). Runs with no secrets; the reply is untrusted text.
python3 -I extract.py REPLY.md CAND_DIR   -> writes CAND_DIR/{train.py, infer.c, README.md}; exit 1 with a reason if
the reply does not hold exactly what is asked: a ```hypothesis block, one ```python block for train.py and one ```c
block for infer.c (the first of each kind is taken), each within its size limit. Nothing in the reply is executed here."""
import os, re, sys

LIMITS = {"hypothesis": 3000, "python": 40000, "c": 40000}
FENCE = re.compile(r"^```[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n(.*?)^```[ \t]*$", re.S | re.M)
KINDS = {"hypothesis": "hypothesis", "text": "hypothesis", "python": "python", "py": "python", "c": "c"}


def blocks(reply):
    out = {}
    for lang, body in FENCE.findall(reply):
        kind = KINDS.get(lang.lower())
        if kind and kind not in out:
            out[kind] = body
    return out


def extract(reply, dst):
    b = blocks(reply)
    missing = [k for k in LIMITS if k not in b]
    if missing:
        return f"missing block(s): {', '.join(missing)}"
    for k, lim in LIMITS.items():
        if len(b[k].encode()) > lim:
            return f"{k} block is {len(b[k].encode())} bytes, over {lim}"
        if "\x00" in b[k]:
            return f"{k} block holds a NUL byte"
    if "def " not in b["python"] or "main" not in b["c"]:
        return "train.py has no function or infer.c has no main"
    os.makedirs(dst, exist_ok=True)
    open(os.path.join(dst, "train.py"), "w").write(b["python"])
    open(os.path.join(dst, "infer.c"), "w").write(b["c"])
    open(os.path.join(dst, "README.md"), "w").write("# Proposal (model-written, untrusted)\n\n" + b["hypothesis"].strip() + "\n")
    return None


if __name__ == "__main__":
    why = extract(open(sys.argv[1], encoding="utf-8", errors="replace").read(), sys.argv[2])
    if why:
        print(f"not extracted: {why}")
        sys.exit(1)
    print("extracted")
```

## runner/proposals/judge.py
```
"""Judge proposals by PLAN.md's fixed measures, and write the leaderboard (queue P5). No secrets; reads JSON only.
python3 -I judge.py one NAME EXTRACT_STATUS REPRODUCE.json DEV.json OUT.json   -> one proposal's verdict
python3 -I judge.py board DIR                                                 -> DIR/LEADERBOARD.md from DIR/*/result.json
Valid: extracted, trained twice to identical weights, and dev accuracy above the floor by at least 3 SE (paired with the
floor's items). Beats a baseline: above label-linear by more than 2 paired SE."""
import glob, json, math, os, sys


def paired(a, b):
    """mean and SE of the per-item difference a - b."""
    d = [x["score"] - y["score"] for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
    return m, sd / math.sqrt(n)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def one(name, extract_status, rep_path, dev_path):
    v = {"name": name, "extracted": extract_status == "extracted", "extract": extract_status,
         "reproduced": False, "valid": False, "beats_linear": False}
    if not v["extracted"]:
        return v
    if os.path.exists(rep_path):
        rep = [r for r in json.load(open(rep_path)) if r.get("candidate") == name]
        v["reproduced"] = bool(rep) and all(r.get("identical") for r in rep)
    if not v["reproduced"] or not os.path.exists(dev_path):
        return v
    dev = json.load(open(dev_path))
    c = dev["candidates"].get(name)
    if not c or "items" not in c:
        return v
    floor = dev["candidates"][dev["floor"]["baseline"]]
    lin = dev["candidates"].get("label-linear")
    mf, sf = paired(c["items"], floor["items"])
    v.update(task=dev["task"], mean=c["mean"], se=c["se"], ir_per_item=c.get("ir_per_item"),
             ns_per_item=c.get("ns_per_item"), over_floor=mf, over_floor_se=sf)
    v["valid"] = mf >= 3 * sf and mf > 0
    if lin:
        ml, sl = paired(c["items"], lin["items"])
        v.update(over_linear=ml, over_linear_se=sl, beats_linear=v["valid"] and ml > 2 * sl)
    return v


def board(d):
    rs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "*", "result.json")))]
    n, k = len(rs), sum(r["valid"] for r in rs)
    lo, hi = wilson(k, n)
    calls = sum(r.get("calls", 1) for r in rs)
    lines = ["# P5 proposals: leaderboard (generated by runner/proposals/judge.py; rules in runner/proposals/PLAN.md on main)", "",
             f"Proposals: {n}. Valid: {k} (rate {k / n if n else 0:.2f}, 95% Wilson {lo:.2f}-{hi:.2f}). "
             f"Beat label-linear: {sum(r['beats_linear'] for r in rs)}. Free calls per valid: "
             f"{calls / k if k else float('inf'):.1f}.", ""]
    if n >= 20:
        first = rs[:20]
        kv = sum(r["valid"] for r in first)
        lines += [f"First 20: {kv} valid -> {'KEEP the one-call proposer' if kv >= 2 else 'SWITCH to an agent harness'} "
                  "(rule fixed in PLAN.md).", ""]
    lines += ["| proposal | task | extracted | reproduced | valid | dev accuracy | vs linear | Ir/item | ns/item |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in rs:
        acc = f"{r['mean']:.4f} ± {r['se']:.4f}" if "mean" in r else ""
        vl = f"{r['over_linear']:+.4f} ± {r['over_linear_se']:.4f}" if "over_linear" in r else ""
        ir = f"{r['ir_per_item']:.0f}" if r.get("ir_per_item") else ""
        ns = f"{r['ns_per_item']:.0f}" if r.get("ns_per_item") else ""
        lines.append(f"| {r['name']} | {r.get('task', '')} | {r['extracted']} | {r['reproduced']} | {r['valid']} | {acc} | {vl} | {ir} | {ns} |")
    open(os.path.join(d, "LEADERBOARD.md"), "w").write("\n".join(lines) + "\n")
    return n, k


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "one":
        v = one(a[1], a[2], a[3], a[4])
        json.dump(v, open(a[5], "w"), indent=1)
        print(json.dumps(v))
    else:
        print("proposals %d, valid %d" % board(a[1]))
```

## runner/proposals/propose.py
```
"""Ask a free model for new candidates (queue P5). Holds the key; runs no generated code.
python3 -I runner/proposals/propose.py OUT_DIR N START
Writes OUT_DIR/<k>/{task, prompt.md, reply.md} for k in 0..N-1 (task k is TASKS[(START + k) % 3]) and appends one line
per call to OUT_DIR/calls.jsonl (for worker/ledger.py record). One call per proposal, no retry (PLAN.md)."""
import glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "router"))
import engines  # noqa: E402

ENGINE, MODEL, MAX_TOKENS = "nvidia", "nvidia/nemotron-3-ultra-550b-a55b", 16000
TASKS = ("optdigits", "text", "adult")
EXAMPLE = "label-linear"


def section(md, title):
    s = md.split("## " + title, 1)[1]
    return "## " + title + s.split("\n## ", 1)[0]


def families():
    out = []
    for d in sorted(glob.glob(os.path.join(ROOT, "runner", "candidates", "*"))):
        name = os.path.basename(d)
        if name.startswith("sat-"):
            continue
        doc = open(os.path.join(d, "train.py")).read().split('"""')[1].strip().split("\n")[0]
        out.append(f"- {name}: {doc}")
    return "\n".join(out)


def scores(task):
    """The latest committed dev lines for the task (P3 step 1 if present, else P2)."""
    for p in sorted(glob.glob(os.path.join(ROOT, "runner", "results", "*", "ci", f"dev-{task}.txt")), reverse=True):
        return "\n".join(l for l in open(p).read().splitlines() if l.strip()[:1].isalpha() or l.startswith("  "))[:3000]
    return "(none recorded)"


def prompt(task):
    contract = open(os.path.join(ROOT, "runner", "CONTRACT.md")).read()
    card = open(os.path.join(ROOT, "tasks", task, "card.json")).read()
    ex = os.path.join(ROOT, "runner", "candidates", EXAMPLE)
    return f"""You write a new candidate model for a public research repository that measures small machine-learning
models on accuracy and CPU cost. Your code is compiled and run in a sandbox with no network; it is judged on a held-back
dev split by a fixed checker.

# The contract
{section(contract, "A candidate (`runner/candidates/<name>/`): code only, no weights")}

Rules that the checks enforce:
- train.py: Python 3 with the standard library and numpy only; deterministic (fixed seeds); single thread; it must finish
  within 20 minutes on one CPU core and within 8 GiB of memory; it reads the card and training records and writes
  weights.bin. Running it twice must give byte-identical weights.bin.
- infer.c: C11, libc and libm only, compiled with gcc -O2 -std=c11 -ffp-contract=off; `./infer weights.bin in.rec N`
  reads N input records and writes exactly N output records to stdout (4-byte little-endian length, then the bytes).
  For this task an output record is one byte: the class index. Validate the weights file size; never read past buffers.
- If you choose settings, choose them on the card's tuning split only (last 20% of train.in, file order), then train
  on all of train.

# The task card
```json
{card}
```

# A working example candidate ({EXAMPLE}): follow its input and output handling exactly
```python
{open(os.path.join(ex, "train.py")).read()}
```
```c
{open(os.path.join(ex, "infer.c")).read()}
```

# Families that already exist (propose a different one)
{families()}

# Current dev results on this task (accuracy, instructions per item, ns per item)
{scores(task)}

# Your answer
Propose ONE new model family for this task that you expect to beat label-linear on dev accuracy, or to match it at a
lower cost per item. Answer with exactly three fenced blocks, in this order and nothing else:
```hypothesis
(at most 10 lines: the family, why it should do well here, and what result would show it does not)
```
```python
(the complete train.py)
```
```c
(the complete infer.c)
```
"""


def main():
    out, n, start = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    engines.TIMEOUT = 600   # Nemotron reasons before answering (worker release test 1)
    for k in range(n):
        task = TASKS[(start + k) % len(TASKS)]
        d = os.path.join(out, str(k))
        os.makedirs(d, exist_ok=True)
        p = prompt(task)
        open(os.path.join(d, "task"), "w").write(task)
        open(os.path.join(d, "prompt.md"), "w").write(p)
        r = engines.compat_call(ENGINE, MODEL, p, max_tokens=MAX_TOKENS)
        open(os.path.join(d, "reply.md"), "w").write(r.get("text") or f"(no reply: status {r.get('status')}: {r.get('error')})")
        with open(os.path.join(out, "calls.jsonl"), "a") as f:
            f.write(json.dumps({k2: r.get(k2) for k2 in ("t", "engine", "model", "ok", "status", "latency_s", "usage")}) + "\n")
        print(f"proposal {k}: task {task}, call ok {r.get('ok')}, status {r.get('status')}")


if __name__ == "__main__":
    main()
```

## runner/proposals/PLAN.md
```
# P5: free models propose candidates (plan, fixed before the first proposal)

Written 2026-11-19 before any proposal was requested. Commit time is the registration time. Design from the project doc
`forge-next-item-2026-11-18` (step 4), narrowed by the 2026-11-19 reviews.

## Hypothesis
A free model, given the task contract, one task card, one working candidate as an example, and the current dev scores,
can write new candidates (`train.py` + `infer.c`) that run under the contract and beat the task's floor, often enough
to be worth running unattended.

## How (`.github/workflows/propose.yml`)
1. **reserve** (no key; writes the worker ledger): one NVIDIA NIM call per proposal against the worker's daily caps
   (`worker/ledger.py`, shared with the queue worker).
2. **propose** (holds the key; runs no generated code): `runner/proposals/propose.py` builds the prompt and makes one
   call per proposal to Nemotron 3 Ultra (NVIDIA NIM, free). Tasks rotate over optdigits, text and adult.
3. **score** (no secrets; read-only token): `runner/proposals/extract.py` takes exactly the three fenced blocks
   (hypothesis, `train.py`, `infer.c`), with size limits, into `runner/candidates/prop-<run>-<k>/` on the runner only.
   `runner/run.py reproduce` (twice, in the jail) and `evaluate --split dev` with the card's baselines. Dev only: no
   proposal goes to the test gate (Q25: public test labels could be memorised).
4. **record** (no key; writes the `proposals` branch and the ledger): prompt, reply, files, result and a leaderboard.

## Measures, fixed now
Over the first 20 proposals:
- **valid**: extracted, compiled in the jail, trained twice to identical weights, and dev accuracy above the floor
  (majority) by at least 3 SE;
- **valid rate** per task and overall, with a 95% Wilson interval;
- **beats a baseline**: dev accuracy above label-linear by more than 2 paired SE;
- **free calls per valid candidate**;
- instructions and ns per item of every valid candidate, against the baselines.

## Rules, fixed now
- **Switch rule** (from the design): fewer than 2 valid of the first 20 (below 1 in 10) means the one-call proposer
  is dropped for an agent harness (OpenHands with Nemotron, bench-proven) before more runs.
- **Keep rule**: at least 2 valid of the first 20 keeps the one-call proposer running, with the same measures
  reported per batch.
- No proposal enters `runner/candidates/` on `main` without a separate check by Claude or the owner (code from a model).

## Forecasts (Claude's, for later calibration scoring)
- At least 2 valid of 20 (keep rule): p = 0.7.
- Overall valid rate at least 0.4: p = 0.35.
- At least one valid proposal beats label-linear on some task: p = 0.4.

## Not claimed
Anything on the test split; anything about models other than Nemotron 3 Ultra on NIM; agent harnesses.
```
