"""Build one findings register from every review series (Claude chat 2026-11-24). Status is evidence-based:
'fixed' cites a commit or PR; 'open' means no fix found on main at the build commit; 'pending-pr' names an open PR."""
import json, re, subprocess, sys
REPO = sys.argv[1]; OUT = sys.argv[2]
def show(ref, p): return subprocess.run(["git", "-C", REPO, "show", f"{ref}:{p}"], capture_output=True, text=True).stdout
MAIN = subprocess.run(["git", "-C", REPO, "rev-parse", "origin/main"], capture_output=True, text=True).stdout.strip()
rows = []
def add(series, fid, file, lines, sev, claim, status, evidence, source):
    rows.append({"id": f"{series}-{fid}", "series": series, "orig_id": fid, "file": file, "lines": lines, "severity": sev,
                 "claim": claim, "status": status, "evidence": evidence, "source": source, "dup_of": None, "queue": []})
# G1: GLM round 1 (83f06fd, committed jsonl); G2 statuses from the round-2 report (5301510d)
g1 = [json.loads(l) for l in show("origin/main", "audit/reviews/glm-20261120/findings.jsonl").splitlines() if l.strip()]
sev = {json.loads(l)["id"]: json.loads(l)["body"]["severity"] for l in show("origin/main", "audit/reviews/glm-20261120/rankings.jsonl").splitlines() if l.strip()}
carried = set("F001 F009 F012 F013 F014 F015 F020 F022 F026 F027 F028 F030 F031 F032 F035 F038 F040 F041 F044 F046 F047 F049 F050 F054 F060 F068 F070 F073 F074 F075 F076 F077 F078 F079 F080 F081 F082 F084 F085 F086 F087 F088 F089 F090 F091 F092 F093 F094 F095 F096 F097 F098 F099 F100 F101 F102 F103 F104 F105 F106 F108 F109 F110".split())
g1fix_after = {"F054": "89694c84/20408b04 (2026-11-21/22)", "F022": "worker/generate.py today_count counts research only (main)",
               "F032": "Z1-F024 fix 31daecbf (#100)", "F035": "worker/README.md now says paid pools are reserved (main)",
               "F088": "Z1-F065 fix 8bfe3ac7 (#99)", }
declined = {"F012": "declined with a registered reason (bench-1 disabled)", "F054": None}
for r in g1:
    f = r["id"]; b = r["body"]
    if f == "F071": st, ev = "superseded", "narrower hole filed as G2-F124 (fixed cf637e57)"
    elif f not in carried: st, ev = "fixed", "fixed by round 2 (GLM report 2026-11-21: 47 of 111; RESPONSE.md)"
    elif f in g1fix_after: st, ev = "fixed", g1fix_after[f]
    else: st, ev = "open", "carried at 5301510d; no fix found on main"
    if f == "F012" and st == "open": ev = declined["F012"]
    add("G1", f, r["file"], r["lines"], sev.get(f, "?"), b["claim"], st, ev, "audit/reviews/glm-20261120 (main)")
def tsv(path, series, fixed, src):
    for l in open(path):
        f, s, loc, claim = l.rstrip("\n").split("\t")
        file, _, line = loc.rpartition(":")
        add(series, f, file, [int(line)] if line.isdigit() else None, s, claim,
            "fixed" if f in fixed else "open", fixed.get(f, "no fix found on main at build"), src)
tsv("/tmp/reg/g2.tsv", "G2", {"F112": "c9e9b1f5", "F113": "c9e9b1f5", "F115": "7886d031", "F117": "315802b2", "F123": "315802b2",
                              "F124": "cf637e57 (#98)", "F125": "37ac4b37 (R3 F2)"}, "exagium-defects-audit-r2 (project doc, Z.ai GLM-4.6, 5301510d)")
z1fix = {k: v for k, v in [("F001", "799f846f #97"), ("F002", "799f846f #97"), ("F014", "924058e8 #100"), ("F016", "8de155e8 #99"),
         ("F017", "70a54479 #98"), ("F019", "a6b5b74f #100"), ("F024", "31daecbf #100"), ("F036", "8bce7d4e #98"), ("F038", "be136ee7 #98"),
         ("F041", "c6e3e26f #98"), ("F065", "8bfe3ac7 #99"), ("F066", "8de155e8 #99"), ("F067", "8bfe3ac7 #99"),
         ("F016", "8de155e8 #99: loop.py gate fails a witness pending 48h; record.py --max-pending-hours stays opt-in")]}
tsv("/tmp/reg/z1.tsv", "Z1", z1fix, "exagium-state-report (project doc, Z.ai, df7890aa) + review-20261122-df7890a.zip")
# GM: Gemini rounds (one series: round b repeats round a's 8)
gmfix = {"F007": "e4867d3f", "F008": "e4867d3f", "F009": "83f06fd4", "F010": "83f06fd4",
         "F011": "83f06fd4", "F012": "83f06fd4"}
for r in [json.loads(l) for l in show("origin/main", "audit/reviews/gemini-20261120b/findings.jsonl").splitlines() if l.strip()]:
    f = r["id"]
    add("GM", f, r["file"], r["lines"], "?", r["body"]["claim"], "fixed" if f in gmfix else "open", gmfix.get(f, "no fix found on main"),
        "audit/reviews/gemini-20261120b (main)")
add("D1", "F001", "research/cards/R32.md", [13], "low", "cited HN link dead live (quote archived on Algolia)", "open", "delta audit 2026-11-23", "exagium-delta-report-20261123")
# duplicates: same defect in more than one series (read by Claude; each pair names the same file and mechanism)
dups = {"Z1-F004": "G1-F104", "GM-F005": "G1-F104", "Z1-F060": "G1-F087", "Z1-F006": "G2-F122", "Z1-F065": "G1-F088", "Z1-F036": "G1-F082",
        "Z1-F024": "G1-F032", "Z1-F068": "G1-F106", "GM-F001": "G1-F101", "GM-F003": "G1-F102", "GM-F004": "G1-F103", "Z1-F030": "G1-F102",
        "GM-F006": "G1-F105"}
dups.pop("Z1-F030")   # audit 2026-11-24: same pattern, different file and consumer
by = {r["id"]: r for r in rows}
for a, b in dups.items():
    by[a]["dup_of"] = b
pending = {"Z1-F006": "#105", "G2-F122": "#105", "G1-F074": "#110, #104", "G1-F070": "#104", "Z1-F038": "#115 (candidate crash refinement)", "Z1-F009": "branch fix/audit-workflow-edges (403 on workflows)", "Z1-F011": "branch fix/audit-workflow-edges (403 on workflows)", "Z1-F033": "branch fix/audit-workflow-edges (403 on workflows)"}
for k, v in pending.items():
    if by[k]["status"] == "open": by[k]["status"] = "pending-pr"; by[k]["evidence"] = v if v.startswith("branch") else f"open PR {v}"
# queue links: any queue item naming the GLM id in its title or next (GLM round-1 bundles and G items)
q = json.loads(show("origin/pilot/lgn-20261115", "audit/2026-11-17/queue.json"))["items"]
for i in q:
    t = i["title"] + " " + (i.get("next") or "")
    if "GLM" in t or (i["id"].startswith("G") and i.get("origin") == "review"):
        for f in set(re.findall(r"\bF\d{3}\b", t)):
            if f"G1-{f}" in by: by[f"G1-{f}"]["queue"].append(f"{i['id']} ({i['status']})")
# corrections from the independent audit of 2026-11-24 (sampled 31 rows; scripts in its report)
notes = {"Z1-F068": ("open", "recurs: README 1,097 vs board 1,239 on 2026-11-24; fix = generate the row from the board"),
         "G1-F106": ("open", "same as Z1-F068"),
         "G1-F082": ("open", "future 'at' fixed (8bce7d4e); naive or malformed 'at' still crashes run() (fromisoformat unguarded)"),
         "GM-F006": ("open", "grade_canary3.py still hardcodes LINES and exact equality for E2/E4; e4867d3f only added E5"),
         "GM-F002": ("needs-read", "docstring still accepts per-run overshoot; unclear"),
         "G1-F073": ("open", "partly fixed: check_sigstore reads integratedTime from verified bytes; monitor's logIndex still from the unsigned wrapper"),
         "G2-F118": ("needs-read", "covered() treats the test file as optional since af002bba; old-items half unverified"),
         "G2-F117": ("fixed", "315802b2"),
         "Z1-F005": ("pending-pr", "open PR #117: guard parses workflow env, reusable calls, local actions, .sh wrappers; known-answer controls 4 of 4 (old guard 1 of 4)")}
for k, (st, ev) in notes.items():
    by[k]["status"], by[k]["evidence"] = st, ev
for k in ("G1-F063","G1-F032","G1-F088","G1-F016","G1-F019","G1-F069","G1-F010","G1-F043","G1-F008","Z1-F036","G2-F124","G2-F115","GM-F011","GM-F012",
          "G1-F028","Z1-F055","Z1-F051","G1-F047","G2-F127","G1-F075","Z1-F052","G1-F044","G1-F085","G2-F119","Z1-F045","G1-F038","G1-F031",
          "Z1-F003","Z1-F005","Z1-F010","Z1-F012","Z1-F016","GM-F006","Z1-F068","G1-F073","G2-F118","GM-F002"):
    by[k]["checked"] = "read in code by the independent audit 2026-11-24"
with open(OUT, "w") as fh:
    for r in rows: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
import collections
live = [r for r in rows if r["dup_of"] is None]
print("built at main", MAIN[:8], "rows", len(rows), "unique", len(live))
print("by status (unique):", dict(collections.Counter(r["status"] for r in live)))
print("open by series:", dict(collections.Counter(r["series"] for r in live if r["status"] in ("open", "pending-pr"))))
print("open by severity:", dict(collections.Counter(r["severity"] for r in live if r["status"] in ("open", "pending-pr"))))
print("open with a queue item:", sum(1 for r in live if r["status"] in ("open", "pending-pr") and r["queue"]))
