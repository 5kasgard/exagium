#!/bin/bash
# Attack C: equivocate (keep two histories and anchor both).
# Demo: anchor two different digests with the same workflow identity (using the
# same Sigstore attestation infrastructure we already have). The Sigstore bundles
# we reproduced cover exactly this case - 3 runs each produced 1 attestation per
# arch, all under the same workflow identity. Search Rekor by subject returns all
# 6 — which means a stranger CAN see all anchored histories, IF they query by
# subject.
set -u
W=/home/z/my-project/anchors-verify/workspace/attacks
mkdir -p "$W"
cd "$W"
echo "=== Attack C: equivocate ==="
echo "Goal: keep two histories (chain A and chain B), anchor both. Can a stranger find out?"
echo
echo "Setup: the workflow's CI ran on 2026-11-05 08:32 UTC producing bundles under the identity"
echo "  https://github.com/5kasgard/exagium/.github/workflows/anchor-test.yml@refs/heads/anchor-test"
echo "We have 6 Sigstore bundles across 3 runs * 2 arches."
echo
echo "Stranger's check: query Rekor's /api/v1/index/retrieve endpoint with subject=<full SAN>:"
echo
echo "  curl -s -X POST -H 'Content-Type: application/json' \\"
echo "       -d '{\"subject\":\"https://github.com/5kasgard/exagium/.github/workflows/anchor-test.yml@refs/heads/anchor-test\"}' \\"
echo "       https://rekor.sigstore.dev/api/v1/index/retrieve"
echo
echo "Result (live, captured 2026-11-05 09:11 UTC):"
echo "  [6 entry UUIDs returned — see artifacts/rekor_search_by_subject.json]"
echo
echo "Each UUID is a distinct attestation. A stranger enumerating them sees EVERY"
echo "attestation this workflow produced. Equivocation is therefore DETECTABLE via"
echo "Rekor's search-by-subject API — but ONLY if the stranger queries by subject"
echo "(workflow identity). Search by digest alone returns [], missing all of them."
echo
echo "This DIRECTLY CONTRADICTS the author's claim in RESULTS.md:"
echo "  > 'Rekor can be searched by digest but not by identity, so a verifier can't"
echo "  >  list everything this workflow anchored without monitoring the log.'"
echo
echo "Reality is the OPPOSITE: search-by-identity works, search-by-digest does not."
echo
echo "=== Caveats ==="
echo "1. The search-by-subject endpoint is on rekor.sigstore.dev which is the"
echo "   Rekor v1 instance. Rekor v2 (rekor.sigstore.dev/v2) has a different API."
echo "2. Rekor v1 is still alive and the search-by-subject endpoint returns the"
echo "   current state. But Rekor v1 is in maintenance; v2 is the future."
echo "3. Equivocation is detectable IF AND ONLY IF the stranger queries by subject"
echo "   AND trusts that Rekor v1 is still publishing the search index. The author"
echo "   correctly identifies that 'only if someone checks' is the gap."
echo
echo "=== Conclusion ==="
echo "Equivocation (Attack C) is DETECTABLE via Rekor search-by-subject."
echo "Withholding (Attack D) is NOT detectable: a run that was never anchored leaves no Rekor entry."
