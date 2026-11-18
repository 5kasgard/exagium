<!-- cloudflare @cf/openai/gpt-oss-120b status=200 latency=71.95s -->

**Indicator script – adversarial review**  

Below each problem the *location* (function name or line range), *what is wrong*, *why it matters for a weekly audit*, *how to fix it*, and an *severity grade* are given.  All issues are presented without any praise.

---

## 1. `size()` – line‑count off‑by‑one & incomplete doc‑type coverage  

**Location**: `def size(repo, files):` (lines 30‑44)  

**What is wrong**  
* `read(...).count("\n")` counts only newline characters.  Files that do **not** end with a trailing newline lose one line from the total.  
* The *kind* mapping treats only `".md"` as *docs*.  Other common documentation files (`.rst`, `.txt`, `.yaml`, `.json`, `.toml`, `.cfg`) are classified as *data* and therefore mixed into the wrong bucket.  

**Why it matters**  
The “Lines per top‑level folder and kind” table is the primary size metric used by auditors to spot growth spikes.  Systematic under‑counting and mis‑categorisation distort trend lines and can hide a sudden surge of documentation debt.  

**Fix**  
```python
lines = read(repo, f).splitlines()
out[top][kind] += len(lines)          # counts every line, even the last one
```
Add the missing extensions to the docs test, e.g.:
```python
if f.endswith(('.md', '.rst', '.txt')):
    kind = "docs"
```

**Severity**: **Major**  

---

## 2. `duplication()` – docstrings and literal constants inflate duplicate bodies  

**Location**: `def duplication(repo, files):` (lines 47‑71)  

**What is wrong**  
The body hash is built from `ast.dump(ast.Module(body=node.body, …))`.  This includes *docstring* `Expr` nodes and any literal constants.  Two functions that differ only in their docstring will be reported as duplicates, while two functions that share the same implementation but have different docstrings will be missed.  

**Why it matters**  
Auditors rely on the *share* figure to gauge technical debt.  Inflated duplication masks real code reuse problems and generates unnecessary work for reviewers who must manually verify each group.  

**Fix**  
Strip docstring nodes before dumping:
```python
def _clean_body(body):
    return [stmt for stmt in body
            if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value,
                (ast.Str, ast.Constant)))]
clean = _clean_body(node.body)
body = ast.dump(ast.Module(body=clean, type_ignores=[]), include_attributes=False)
```

**Severity**: **Critical** (the main “duplication” signal is unreliable).

---

## 3. `unreferenced()` – heuristic too narrow, produces false positives  

**Location**: `def unreferenced(repo, files):` (lines 74‑92)  

**What is wrong**  
* Only files whose *extension* appears in **TEXT** files are examined.  Imports, `from … import …`, dynamic loading, and CLI entry‑point registrations in other `.py` files are ignored, so many truly referenced scripts are flagged.  
* The regular expression `(?<![\w-])stem(?![\w-])` also matches the *stem* inside longer identifiers (e.g. `run_test` matches `test`).  
* The check `base in t` looks for the full filename *including* the `.py` suffix, which rarely appears in documentation, creating unnecessary false positives.  

**Why it matters**  
The audit report currently lists **5 unreferenced code files** on the first run.  If the heuristic is noisy, the auditor will waste time chasing dead code that is actually used, and may incorrectly conclude that the repository has hidden dead code.  

**Fix**  
1. Scan **all** `.py` files for `import` / `from … import …` statements using `ast` and collect the imported module names.  
2. Search *both* code and documentation for those names (without the extension).  
3. Tighten the regex to word‑boundary matches and exclude matches that are part of longer identifiers:
```python
pat = re.compile(r'\b' + re.escape(stem) + r'\b')
```
4. Remove the `base in t` shortcut; rely solely on the regex.  

**Severity**: **Critical** (the indicator is fundamentally broken).

---

## 4. `cannot_fail()` – docstring mentions patterns that are never detected  

**Location**: `def cannot_fail(repo, files):` (lines 95‑108)  

**What is wrong**  
* The docstring lists three patterns: `"|| true"`, `"continue-on-error: true"`, **and** `"set +e"` without a later `exit`.  The implementation only checks the first two.  
* The `continue-on-error` regex is anchored to the exact string `"continue-on-error:\s*true"`; it fails on YAML variations such as `continue-on-error: true # comment` or on uppercase/lowercase differences.  

**Why it matters**  
Swallowing failures is a high‑risk practice.  Missing `"set +e"` lines means the audit silently ignores a whole class of failure‑hiding code, giving a false sense of safety.  

**Fix**  
```python
if "|| true" in s:
    hits.append(...)
elif re.search(r'^\s*continue-on-error\s*:\s*true\b', line, re.IGNORECASE):
    hits.append(...)
elif re.search(r'\bset\s*\+\s*e\b', line):
    # Verify there is no later `exit` or `set -e`
    hits.append(...)
```
Optionally parse the workflow YAML to detect the `continue-on-error` key structurally.

**Severity**: **Major**  

---

## 5. `ci()` – day‑boundary mis‑classification  

**Location**: `def ci(runs, days=7, now=None):` (lines 111‑124)  

**What is wrong**  
The condition `(now - t).days < days` truncates the time‑difference to whole days.  A run that happened 6.9 days ago is counted, but one that happened 7.1 days ago is excluded, even though both are within the *7‑day* window when measured in hours.  

**Why it matters**  
The “7‑day non‑success share” fluctuates artificially at the exact day boundary, making week‑over‑week comparisons noisy and potentially hiding a regression.  

**Fix**  
Replace the integer‑day test with a `timedelta` comparison:
```python
if now - t < dt.timedelta(days=days):
    …
```

**Severity**: **Major**  

---

## 6. `queue_state()` – git‑log quoting bug and age always zero  

**Location**: `def queue_state(queue, qgit=None, qpath="audit/2026-11-17/queue.json", today=None):` (lines 127‑147)  

**What is wrong**  
* The command `git log … -S '"id": "{i["id"]}"' …` passes the search string **with embedded double quotes** to the shell.  `git -S` expects the literal string *without* the surrounding quotes; the extra quotes cause the search to return no matches, so `log` is empty for every item.  
* Consequently `age[i["id"]]` is never set and defaults to `0` (the dict entry is missing, but the caller treats missing keys as 0).  The first‑run report therefore shows **open‑item ages all 0 days**, which is exactly the symptom described.  
* `today` is taken from `dt.date.today()` (local time) while `log` dates are UTC, producing off‑by‑one errors on the day of the run.  

**Why it matters**  
Age of open queue items is a core “staleness” indicator.  Reporting every item as brand‑new hides genuine backlog buildup and defeats the purpose of the audit.  

**Fix**  
```python
search = f'"id": "{i["id"]}"'          # no outer quotes
log = subprocess.run(
        ["git", "-C", qgit, "log", "--all", "--reverse",
         "--format=%ad", "--date=short", "-S", search, "--", qpath],
        capture_output=True, text=True, check=True).stdout.splitlines()
```
* Use `check=True` to surface git errors.  
* Compute `today` in UTC: `today = dt.datetime.now(dt.timezone.utc).date()`.  

**Severity**: **Critical**  

---

## 7. `stale_branches()` – timezone mismatch & fragile parsing  

**Location**: `def stale_branches(repo, days=14):` (lines 150‑160)  

**What is wrong**  
* `now` is UTC (`datetime.now(timezone.utc)`), but `git for-each-ref` emits ISO‑strict timestamps *without* a timezone offset (e.g. `2026-11-14T12:34:56+00:00`).  `datetime.fromisoformat` parses the offset, yielding an *aware* datetime, which is fine, but if the git config uses local time the offset may be missing, producing a *naïve* datetime that cannot be compared to an aware one and raises a `TypeError`.  

**Why it matters**  
A crash or silent omission of stale branches would leave the audit without a key risk indicator.  

**Fix**  
Normalize both sides to UTC naïve datetimes:
```python
now = dt.datetime.utcnow()
when = dt.datetime.fromisoformat(when.rstrip('Z')).replace(tzinfo=None)
age = (now - when).days
```
Add a fallback for missing offset.

**Severity**: **Major**  

---

## 8. `claims_without_evidence()` – over‑broad number detection, under‑detecting evidence  

**Location**: `def claims_without_evidence(repo, path="README.md"):` (lines 163‑176)  

**What is wrong**  
* `NUMBER` matches any isolated integer or float, catching version numbers, issue IDs, or even line numbers in code blocks that are not *claims*.  
* `EVIDENCE` does not recognise common evidence forms: GitHub issue/PR URLs, JIRA tickets, hyperlinks, markdown footnote references, or badge images (`![...]`).  

**Why it matters**  
The report currently lists **4 README lines** as lacking evidence.  If the regex flags non‑claims, the auditor will waste time; if it misses real evidence pointers, the audit will incorrectly flag missing provenance, inflating the “blind‑spot” metric.  

**Fix**  
* Refine `NUMBER` to require a surrounding context such as “% improvement”, “× faster”, or a measurement unit:
```python
NUMBER = re.compile(r'\b(?:\d+(?:\.\d+)?\s*(?:%|×|x|ns|µs|ms|s))\b')
```
* Expand `EVIDENCE` to capture URLs, issue/PR references, and markdown links:
```python
EVIDENCE = re.compile(
    r'(https?://\S+|#\d+|GH-\d+|JIRA-\d+|!\[.*?\]\(.*?\)|\b[0-9a-f]{7,40}\b|'
    r'\b\d{11}\b|[\w./-]+/[\w./-]+|\.(json|md|txt|py)\b|ledger entr|run \d|entry \d)'
)
```

**Severity**: **Major**  

---

## 9. `delta()` – incomplete change tracking  

**Location**: `def delta(new, old):` (lines 179‑186)  

**What is wrong**  
* Only six scalar fields are compared.  Important weekly signals such as `ci_7d`, `stale_branches`, and the *size* breakdown are omitted, so the “change” section silently hides regressions.  
* The key `code_lines` sums `v.get("code", 0)` but `size()` stores the *kind* under the key `"code"` **only** for files that also match `TEXT`.  If a new non‑text code file is added (e.g., a binary executable), it is ignored, breaking the delta.  

**Why it matters**  
Auditors look at the `change` object to spot trends.  Missing fields give a false impression that the system is stable.  

**Fix**  
Add the missing indicators:
```python
pick = lambda r: {
    "duplication_share": r["duplication"]["share"],
    "unreferenced": r["unreferenced"]["count"],
    "cannot_fail_lines": r["cannot_fail"]["count"],
    "claims_without_evidence": r["claims_without_evidence"]["count"],
    "stale_branches": len(r["stale_branches"]),
    "ci_not_success_share": sum(v["share"] for v in r["ci_7d"].values()),
    "code_lines": sum(v.get("code", 0) for v in r["size"].values()),
}
```

**Severity**: **Major**  

---

## 10. `main()` – fragile JSON loading & exclusive output creation  

**Location**: `def main(argv=None):` (lines 189‑215)  

**What is wrong**  
* `load = lambda p: json.load(open(p)) if p else None` does **not** catch `FileNotFoundError` or `JSONDecodeError`.  Supplying a non‑existent `--previous` or a corrupted `RUNS.json` aborts the whole audit.  
* The output file is opened with mode `"x"` (exclusive create).  CI pipelines that re‑run the same job will fail on the second attempt because the file already exists, forcing a manual cleanup.  

**Why it matters**  
A single missing optional file should not bring down the weekly audit; the script must be robust enough to run unattended.  

**Fix**  
```python
def _load(p):
    if not p:
        return None
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        sys.stderr.write(f"Warning: could not load {p}: {e}\n")
        return None
load = _load
```
Open the output with `"w"` (overwrite) or `"a"` if incremental logs are desired.

**Severity**: **Minor** (does not affect indicator correctness but makes the tool brittle).

---

## 11. Missing cheap‑to‑compute indicators that an “outside auditor of direction and debt” would expect  

| Missing indicator | Why it matters (weekly audit) | Approx. implementation cost |
|-------------------|------------------------------|-----------------------------|
| **Test‑coverage trend** (`.coverage` or `pytest --cov`) | Shows whether code growth is being protected by tests; a drop signals rising risk. | < 30 LOC: run `coverage json` and compute `covered / total`. |
| **Static‑analysis warnings count** (`flake8`, `pylint`) | Lint violations are a cheap proxy for code‑quality debt. | < 20 LOC: invoke the tool, parse line count. |
| **Dependency freshness** (`pip list --outdated`) | Stale dependencies increase security debt. | < 25 LOC: run the command, count outdated packages. |
| **Open‑vs‑closed PR ratio** (GitHub API) | A high backlog of open PRs indicates process bottleneck. | < 40 LOC: list PRs, compute ratio. |
| **Lines added / deleted per week** (`git log --since=7.days --shortstat`) | Direct measure of code churn; spikes may indicate rushed work. | < 15 LOC. |
| **Documentation‑to‑code ratio** (size output already present but not normalized) | Helps spot “code‑only” growth. | Adjust `size()` to output a ratio. |

The absence of any of these signals means the audit cannot give a holistic view of *direction* (are we improving or deteriorating) and *debt* (what categories are growing).  

**Severity**: **Minor** for each missing indicator, but collectively **Major** for the audit’s completeness.

---

# Overall severity summary  

| Issue | Severity |
|------|----------|
| `queue_state()` age bug | **Critical** |
| `duplication()` body includes docstrings | **Critical** |
| `unreferenced()` heuristic false positives | **Critical** |
| `cannot_fail()` missing pattern detection | **Major** |
| `ci()` day‑boundary mis‑classification | **Major** |
| `size()` line‑count & doc‑type bug | **Major** |
| `stale_branches()` timezone fragility | **Major** |
| `claims_without_evidence()` regex too broad / narrow | **Major** |
| `delta()` incomplete change set | **Major** |
| `main()` fragile JSON loading | **Minor** |
| Missing high‑value cheap indicators | **Major (collective)** |

---

## Merge recommendation  

**Do not merge** until all **Critical** and **Major** defects are fixed and the missing high‑value indicators are added (or a clear justification for their omission is provided).  The current script produces misleading numbers (queue ages, duplication share, unreferenced files) and omits several risk‑important signals, which defeats the purpose of a weekly, independent whole‑system audit.  

**Next steps**: address the critical bugs first, then the major ones, run the test suite on a representative repository, and verify that the first‑run baseline matches manual inspection before allowing the script into the CI pipeline.
