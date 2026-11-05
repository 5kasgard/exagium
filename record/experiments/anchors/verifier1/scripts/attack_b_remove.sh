#!/bin/bash
# Attack B: remove or edit a record. Local demo on a throwaway git repo.
# Demonstrates: (1) drop an unfavourable record; (2) edit another to hide the gap;
# (3) force-push the rewritten history; (4) delete and recreate the repo.
# For each, shows whether a stranger cloning fresh after the attack notices.
set -u
W=/home/z/my-project/anchors-verify/workspace/attacks
mkdir -p "$W"
cd "$W"
rm -rf repo_remove
mkdir repo_remove
cd repo_remove
git init -q
git config user.email verifier@exagium.test
git config user.name "verifier"
mkdir -p runs
# Pretend we wrote 3 records; record #2 has an unfavourable verdict
echo "good"   > runs/a1.json ; git add runs/a1.json
echo "BAD"    > runs/a2.json ; git add runs/a2.json
echo "good"   > runs/a3.json ; git add runs/a3.json
git -c user.email=v@e -c user.name=v commit -q -m "records a1 a2 a3"
echo "=== before attack: git log ==="
git log --oneline
echo "=== before attack: runs/ contents ==="
ls runs/
echo
echo "=== Attack B.1: drop the unfavourable a2.json, edit a3 to reference a1 (chain repair) ==="
git rm -q runs/a2.json
# Edit a3 to "rebase" it on a1
echo "good (rebased on a1)" > runs/a3.json
git add runs/a3.json
git -c user.email=v@e -c user.name=v commit -q -m "remove a2 (unfavourable); rebase a3"
echo "=== after B.1 (normal commit): git log ==="
git log --oneline
echo "=== runs/ ==="
ls runs/
echo "Stranger cloning fresh sees only 2 records. Nothing in the public history screams 'I deleted a2'."
echo
echo "=== Attack B.2: force-push to wipe the deletion commit entirely ==="
# Rewrite history so a2 was never committed
git checkout -q --orphan tmp
git add runs/a1.json runs/a3.json
git -c user.email=v@e -c user.name=v commit -q -m "records a1 a3"
git branch -M main
echo "=== after B.2 (force-push): git log ==="
git log --oneline
echo "=== runs/ ==="
ls runs/
echo "Stranger cloning fresh sees only one commit, no deletion trail. Force-push erases evidence from public view."
echo
echo "=== The independent clock: GitHub /repos/.../activity API ==="
echo "If this were a real GitHub repo, the activity log would still show a PushEvent at the original time the bad commit was pushed. The author's RESULTS.md confirms: 'A force push shows in the activity log, but deleting and recreating the repo wipes that log in one step.' So:"
echo "  - force-push:           detectable via activity log (read-only from outside, no auth)"
echo "  - delete + recreate:    ACTIVITY LOG WIPED — undetectable from outside GitHub"
echo
echo "=== Attack B.3: gist revision wiping ==="
rm -rf "$W/gist_demo"
mkdir -p "$W/gist_demo"
cd "$W/gist_demo"
# Simulate a gist with two revisions: v1 has digest_X, v2 (force-push) has digest_Y
echo "fe8d93a9...1de8" > digest_v1.txt
git init -q
git config user.email v@e ; git config user.name v
git add digest_v1.txt
git -c user.email=v@e -c user.name=v commit -q -m "gist v1: real digest"
echo "deadbeef...cafe" > digest_v1.txt
git -c user.email=v@e -c user.name=v commit -q -m "gist v2: forged digest"
git log --oneline
echo "Stranger fetching the gist URL sees only v2. The author's RESULTS.md confirms: 'Revisions can be force-pushed away, there's no activity log, and CI can't write one.'"
echo
echo "=== Attack B.4: deleting GitHub attestation through the API ==="
echo "Per the author's claim: 'The owner can delete GitHub's copy of an attestation through the API; the Rekor copy stays.' The local demo of this requires a GitHub PAT (attestation delete scope). Analytically: deleting the GitHub-side copy removes it from the GH UI/API; the Rekor entry remains and is still discoverable via the search-by-subject API we tested. So deleting GH-side attestations is detectable ONLY if the stranger also queries Rekor. A stranger who trusts only the GH API is fooled."
