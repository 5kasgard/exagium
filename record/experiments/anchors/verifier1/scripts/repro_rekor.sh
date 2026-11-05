#!/bin/bash
# Reproduce the live Rekor search findings.
# Captures the live output of search-by-subject and search-by-digest,
# so the report's "author's claim is inverted" finding is independently
# re-runnable.
set -u
OUT=/home/z/my-project/anchors-verify/artifacts/rekor_search_by_subject.txt
DIGEST=fe8d93a9dfb783350aa23a0c4471020a1b6cf9a17f6e5bd317b748e045b31de8
SUBJECT="https://github.com/5kasgard/exagium/.github/workflows/anchor-test.yml@refs/heads/anchor-test"
{
echo "=== Rekor search-by-subject (SAN) ==="
echo "POST https://rekor.sigstore.dev/api/v1/index/retrieve"
echo "Body: {\"subject\":\"$SUBJECT\"}"
echo
curl -s -m 15 -X POST -H 'Content-Type: application/json' \
  -d "{\"subject\":\"$SUBJECT\"}" \
  "https://rekor.sigstore.dev/api/v1/index/retrieve" | python3 -m json.tool
echo
echo "=== Rekor search-by-digest (sha256 of anchored digest) ==="
echo "POST https://rekor.sigstore.dev/api/v1/index/retrieve"
echo "Body: {\"queries\":[{\"hash\":\"sha256:$DIGEST\"}]}"
echo
curl -s -m 15 -X POST -H 'Content-Type: application/json' \
  -d "{\"queries\":[{\"hash\":\"sha256:$DIGEST\"}]}" \
  "https://rekor.sigstore.dev/api/v1/index/retrieve" | python3 -m json.tool
echo
echo "=== Rekor search-by-issuer+subject (alternative form) ==="
curl -s -m 15 -X POST -H 'Content-Type: application/json' \
  -d "{\"identities\":[{\"issuer\":\"https://token.actions.githubusercontent.com\",\"subject\":\"$SUBJECT\"}]}" \
  "https://rekor.sigstore.dev/api/v1/index/retrieve" | python3 -m json.tool
echo
echo "=== Rekor v2 instance (log2025-1) probe ==="
echo "GET /api/v2/log/proof:"
curl -s -m 8 -o /dev/null -w '  http_code=%{http_code}\n' "https://log2025-1.rekor.sigstore.dev/api/v2/log/proof"
echo "GET /api/v2/log/publicKey:"
curl -s -m 8 -o /dev/null -w '  http_code=%{http_code}\n' "https://log2025-1.rekor.sigstore.dev/api/v2/log/publicKey"
echo
} | tee "$OUT"
