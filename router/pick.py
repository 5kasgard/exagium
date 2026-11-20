"""Spread review calls over the free pools instead of loading one (owner 2026-11-20: don't overwhelm any one agent or model;
Cloudflare refused on 2026-11-20 at 06:09Z: its allowance is a rolling 24 h, and the 24 h before held 10,528 neurons,
much of it reviews the ledger did not see; an earlier reading of this as same-day use was wrong, see worker/ledger.py).
python3 -I router/pick.py LEDGER_DAY_JSON K   -> prints K ENGINE:MODEL entries from different pools, least-used pool first.
python3 -I router/pick.py LEDGER_DAY_JSON check ENGINE:MODEL ...   -> prints the entries whose pool still has headroom."""
import json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "worker"))
from ledger import CAPS, ROLLING_NEURONS, rolling_neurons  # noqa: E402  the same caps the worker obeys, with their sources

# Reviewers qualified by the record (P31 known-answer runs; earlier blind reviews). One or two per pool.
REVIEWERS = [
    ("openrouter", "particle-ai/particle-1:free"),            # P31: 5/5 controls, 30/30 twice
    ("openrouter", "nvidia/nemotron-3.5-super-vl:free"),      # P31: 5/5, 29/30 twice
    ("nvidia", "nvidia/nemotron-3-ultra-550b-a55b"),          # worker reviewer and checker
    ("groq", "openai/gpt-oss-120b"),                          # worker drafter
    ("cloudflare", "@cf/openai/gpt-oss-120b"),                # blind reviews since 2026-11-17
]
NEURONS_PER_CALL = 600   # the worker's measured projection for one Cloudflare call (max seen 438)


def used_fraction(led, pool, rolling=0.0):
    cap, e = CAPS.get(pool, {}), led.get(pool) or {}
    f = (rolling + NEURONS_PER_CALL) / ROLLING_NEURONS if pool == "cloudflare" else 0.0
    # reservations held by runs in flight count against the cap, as in worker/ledger.py (GLM F037)
    held_calls = sum((e.get("pending_calls") or {}).values())
    held_neurons = sum((e.get("pending") or {}).values())
    if cap.get("calls"):
        f = max(f, (e.get("calls", 0) + held_calls + 1) / cap["calls"])
    if cap.get("neurons"):
        f = max(f, (e.get("neurons", 0) + held_neurons + NEURONS_PER_CALL) / cap["neurons"])
    return f


def main(a):
    led = json.load(open(a[0])) if os.path.exists(a[0]) else {}
    roll = rolling_neurons(os.path.dirname(os.path.dirname(os.path.abspath(a[0]))))   # the worktree holding ledger/
    uf = lambda pool: used_fraction(led, pool, roll)
    if a[1] == "check":
        print(" ".join(x for x in a[2:] if uf(x.split(":", 1)[0]) <= 1.0))
        return 0
    k, picked, pools = int(a[1]), [], set()
    for pool, model in sorted(REVIEWERS, key=lambda r: uf(r[0])):
        if pool in pools or uf(pool) > 1.0:
            continue
        picked.append(f"{pool}:{model}"); pools.add(pool)
        if len(picked) == k:
            break
    print(" ".join(picked))
    return 0 if picked else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
