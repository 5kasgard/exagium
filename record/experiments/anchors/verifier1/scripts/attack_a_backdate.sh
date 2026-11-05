#!/bin/bash
# Attack demos — local, no GitHub fork needed.
# Each demo sets up a small scenario in workspace/, executes the attack, and
# shows the outcome from the stranger's POV. Files become regression tests.
set -u
cd /home/z/my-project/anchors-verify/workspace
W=/home/z/my-project/anchors-verify/workspace/attacks
mkdir -p "$W"
DIGEST=fe8d93a9dfb783350aa23a0c4471020a1b6cf9a17f6e5bd317b748e045b31de8

echo "=== Attack A: backdate a record (commit dates + TSA token -attime) ==="
echo "Goal: make a record appear older than it really is, against each anchor."
echo
echo "--- A.1 git: forge author/committer dates on a fresh commit ---"
rm -rf "$W/git_backdate"
mkdir -p "$W/git_backdate"
cd "$W/git_backdate"
git init -q
git config user.email verifier@exagium.test
git config user.name "verifier"
echo "real record at $(date -u +%FT%TZ)" > record.txt
git add record.txt
# Real commit time = now; forge commit dates to 2020-01-01
GIT_AUTHOR_DATE="2020-01-01T00:00:00Z" GIT_COMMITTER_DATE="2020-01-01T00:00:00Z" \
  git commit -q -m "record v1 (backdated to 2020)"
echo "git log shows:"
git log --pretty='format:  %h author=%ai committer=%ci subject=%s' | head -5
echo
echo "Stranger sees commit date 2020-01-01. The repo's GitHub push activity log, however, would show the real push time. So git dates alone = forgeable; GitHub's /repos/.../activity API is the independent clock."
echo
echo "--- A.2 RFC 3161: try to backdate a TSA token via openssl -attime ---"
echo "Take the DigiCert token (granted 2026-11-05 08:32:21 UTC) and verify at an earlier -attime."
TOK=/home/z/my-project/upload/exagium-anchors/anchors/41995230106/anchors-ubuntu-24.04/r2.tsr
ROOTS=/home/z/my-project/upload/exagium-anchors/anchors/41995230106/anchors-ubuntu-24.04/roots.pem
# Token's own time
echo "  verify at token's own time (epoch 1793867541 = 2026-11-05T08:32:21Z):"
openssl ts -verify -digest "$DIGEST" -in "$TOK" -CAfile "$ROOTS" -attime 1793867541 2>&1 | tail -1
echo "  verify at an EARLIER time (epoch 1577836800 = 2020-01-01T00:00:00Z), i.e. claim the record existed in 2020:"
openssl ts -verify -digest "$DIGEST" -in "$TOK" -CAfile "$ROOTS" -attime 1577836800 2>&1 | tail -1
echo "  => RFC 3161 token's gen_time is inside the signed TSTInfo; -attime only changes the cert-validity window, NOT the gen_time. So you cannot make a token *claim* an earlier time. The gen_time is what the token says; -attime just decides whether the TSA cert was valid when you check."
echo
echo "--- A.3 OTS / Sigstore: backdate attack surface ---"
echo "  OTS: the proof references a Bitcoin block hash. Backdating requires re-mining Bitcoin. Not feasible."
echo "  Sigstore: the bundle's integratedTime is the time Rekor added the entry. Forging an earlier time requires a fake Rekor entry that fails inclusion-proof verification."
echo
echo "=== Attack A summary ==="
echo "  git commit dates:    ATTACK SUCCEEDS (forgeable)."
echo "  GitHub activity log: ATTACK FAILS (server-side, read-only from outside)."
echo "  RFC 3161 token:      ATTACK FAILS (TSA's gen_time is signed; can't backdate)."
echo "  OTS proof:           ATTACK FAILS (Bitcoin block hash)."
echo "  Sigstore bundle:     ATTACK FAILS (Rekor inclusion proof)."
