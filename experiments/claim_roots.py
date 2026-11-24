"""Claim-roots experiment: a pure rule, fixed before any data is read.

PLAN: On the landed research cards, count the claims whose supporting sources share one root that a second source
repeats, and pass if that share is at least 20% of claims.
"""

PLAN = """On the landed research cards, count the claims whose supporting sources share one root that a second source
repeats, and pass if that share is at least 20% of claims."""


def rule(result):
    """Pure rule: holds if shared-root claims are at least 20% of claims; invalid if claims is 0 or missing."""
    claims = result.get("claims")
    if not claims:
        return {"verdict": "invalid", "why": f"claims is {claims!r}; need a positive count"}
    shared = result.get("shared_root_claims")
    if shared is None:
        return {"verdict": "invalid", "why": f"shared_root_claims missing; claims={claims}"}
    share = shared / claims
    verdict = "holds" if share >= 0.2 else "fails"
    why = f"{shared} of {claims} claims share a root ({share:.1%}); threshold 20%"
    return {"verdict": verdict, "why": why}
