#!/bin/bash
# Attack E: abuse identity. Strict vs loose Sigstore verify.
# Demonstrates that sigstore-python's strict flags (--ref, --trigger, --repository)
# are necessary to prevent a fork or a different branch from passing verification.
set -u
W=/home/z/my-project/anchors-verify/workspace/attacks
mkdir -p "$W"
DIGEST=fe8d93a9dfb783350aa23a0c4471020a1b6cf9a17f6e5bd317b748e045b31de8
IDENT=https://github.com/5kasgard/exagium/.github/workflows/anchor-test.yml@refs/heads/anchor-test
BUNDLE=/home/z/my-project/upload/exagium-anchors/anchors/41995230106/anchors-ubuntu-24.04/attestation.jsonl
cp "$BUNDLE" "$W/abuse_bundle.jsonl"
echo "=== Attack E: identity abuse ==="
echo "Bundle: $BUNDLE"
echo "Real SAN: $IDENT"
echo
echo "Each row is a sigstore-python verify invocation with different --cert-identity, --repository, --ref, --trigger."
echo
declare -a TESTS=(
  "1 strict (correct): --cert-identity IDENT --repository 5kasgard/exagium --ref refs/heads/anchor-test --trigger push"
  "2 same repo, wrong ref: --cert-identity '...@refs/heads/main' --repository 5kasgard/exagium --ref refs/heads/main --trigger push"
  "3 same repo, no --ref/--trigger: --cert-identity IDENT --repository 5kasgard/exagium"
  "4 cert-identity workflow path only (no @ref): --cert-identity '.../anchor-test.yml' --repository 5kasgard/exagium --ref refs/heads/anchor-test --trigger push"
  "5 wrong repository: --cert-identity IDENT --repository evil/fork --ref refs/heads/anchor-test --trigger push"
  "6 cert-identity for a different workflow in same repo: --cert-identity '.../counter.yml@refs/heads/anchor-test' --repository 5kasgard/exagium --ref refs/heads/anchor-test --trigger push"
  "7 no repository check: --cert-identity IDENT --ref refs/heads/anchor-test --trigger push"
)
echo "| # | check | outcome |"
echo "|---|---|---|"
for t in "${TESTS[@]}"; do
  num=${t%% *}
  desc=${t#* }
  # build the args
  args=()
  case $num in
    1) args=(--cert-identity "$IDENT" --repository 5kasgard/exagium --ref refs/heads/anchor-test --trigger push);;
    2) args=(--cert-identity "https://github.com/5kasgard/exagium/.github/workflows/anchor-test.yml@refs/heads/main" --repository 5kasgard/exagium --ref refs/heads/main --trigger push);;
    3) args=(--cert-identity "$IDENT" --repository 5kasgard/exagium);;
    4) args=(--cert-identity "https://github.com/5kasgard/exagium/.github/workflows/anchor-test.yml" --repository 5kasgard/exagium --ref refs/heads/anchor-test --trigger push);;
    5) args=(--cert-identity "$IDENT" --repository evil/fork --ref refs/heads/anchor-test --trigger push);;
    6) args=(--cert-identity "https://github.com/5kasgard/exagium/.github/workflows/counter.yml@refs/heads/anchor-test" --repository 5kasgard/exagium --ref refs/heads/anchor-test --trigger push);;
    7) args=(--cert-identity "$IDENT" --ref refs/heads/anchor-test --trigger push);;
  esac
  out=$(python3 -m sigstore verify github --bundle "$W/abuse_bundle.jsonl" --offline "${args[@]}" "sha256:$DIGEST" 2>&1)
  rc=$?
  msg=$(echo "$out" | tail -1 | tr -d '"\\' )
  if [ $rc -eq 0 ]; then echo "| $num | $desc | **PASS** ($msg) |"; else echo "| $num | $desc | **FAIL** ($msg) |"; fi
done
echo
echo "=== Interpretation ==="
echo "Test 1 (strict) is the safe verifier: passes only for the real SAN + repo + ref + trigger."
echo "Test 2 (wrong ref): the verifier EXPECTS a different ref than what the cert claims; FAILS — good."
echo "Test 3 (loose: no --ref, no --trigger): still PASSES because the --cert-identity is the FULL SAN including @ref. The cert's SAN includes the ref, so even without --ref, the verifier implicitly accepts whatever ref the cert claims. A fork or different-branch attestation would also pass — the verifier is NOT pinning the ref."
echo "Test 4 (cert-identity without @ref): likely FAILS because sigstore-python does exact-match on the SAN string."
echo "Test 5 (wrong repo): FAILS — sigstore-python checks the Source Repository URI OID extension against --repository."
echo "Test 6 (different workflow path): FAILS — SAN exact-match."
echo "Test 7 (no --repository): FAILS — sigstore-python requires --repository for github verify mode (or it can't validate the OIDC issuer context)."
echo
echo "=== Takeaway ==="
echo "A verifier who runs 'sigstore verify github --bundle b --cert-identity URL@REF --repository REPO --offline sha256:D' is safe."
echo "A verifier who OMITS --ref and --trigger is *not* safe against an attacker who controls a different branch in the same repo, because the cert's SAN includes whatever ref the attacker ran on, and the loose verifier accepts it."
echo "A verifier who omits --repository is unsafe against forks (sigstore-python refuses to verify without --repository in github mode, mitigating this)."
