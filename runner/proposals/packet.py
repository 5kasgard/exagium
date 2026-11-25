"""Proposal packet (queue K68, P5d): what a proposer could learn from its parent's dev-fold run, read from artifacts that
already exist. Reads only out/dev.json, out/dev.txt, the dev counter record and the task's dev.key; never a test-split file,
a key or a token. Exactly four top-level keys."""
import collections, glob, json, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import records  # noqa: E402

MISSES = ("I1mr", "ILmr", "D1mr", "DLmr", "D1mw", "DLmw")   # counter/count.py EVENTS, the cache-miss counts


def build(out_dir, task_dir, champion_path, name):
    """out_dir: a loop/score job's out/ (dev.json, dev.txt, records/); task_dir: the task folder (its dev.key only)."""
    dev = json.load(open(os.path.join(out_dir, "dev.json")))
    c = dev["candidates"][name]
    n = dev.get("n") or len(c["items"])
    # runner/run.py:140-170 names records <name>-n<N>-<digest16>.json; test runs share the folder, so match N = dev n
    pat = re.compile(re.escape(name) + r"-n" + str(n) + r"-[0-9a-f]{16}\.json$")
    hits = sorted(p for p in glob.glob(os.path.join(out_dir, "records", "*.json")) if pat.search(os.path.basename(p)))
    cache = None
    if hits:
        counts = json.load(open(hits[0]))["reps"][0]["counts"]
        cache = {k: counts[k] for k in MISSES}
    try:
        tail = open(os.path.join(out_dir, "dev.txt"), errors="replace").read().splitlines()[-40:]
    except OSError:
        tail = []
    keys = records.read(os.path.join(task_dir, "dev.key"))
    clusters = collections.defaultdict(list)   # failed dev items grouped by their key label
    for i, it in enumerate(c["items"]):
        if it["score"] < 1:
            clusters[keys[i].decode("utf-8", "replace")].append(i)
    return {"stderr_tail": tail, "counts": {"instructions": c.get("ir_per_item"), "cache": cache},
            "clusters": dict(clusters), "champion_source": open(champion_path, errors="replace").read()}
