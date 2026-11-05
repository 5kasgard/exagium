#!/bin/bash
# Experiment for open item 3 (where does the record's outside anchor live?).
# Tries each candidate anchor route for one sha256 digest and writes one JSON line per attempt.
# Usage: try_anchors.sh <sha256-hex> <out-dir>
# Runs on a GitHub runner (open network); this workspace's proxy blocks most TSAs.
set -u
D="$1"; OUT="$2"; mkdir -p "$OUT"; cd "$OUT"
J=attempts.jsonl
now() { date -u +%s.%N; }
emit() { printf '%s\n' "$1" >> "$J"; }

# --- RFC 3161 time-stamp authorities (free, public). Query carries only the digest. ---
openssl ts -query -digest "$D" -sha256 -cert -no_nonce -out q.tsq 2>/dev/null
curl -s -m 20 -o freetsa_ca.pem https://freetsa.org/files/cacert.pem
cat /etc/ssl/certs/ca-certificates.crt freetsa_ca.pem > roots.pem 2>/dev/null
i=0
for url in https://freetsa.org/tsr http://timestamp.digicert.com http://timestamp.sectigo.com \
           http://ts.ssl.com http://timestamp.globalsign.com/tsa/r6advanced1 https://zeitstempel.dfn.de \
           http://time.certum.pl https://timestamp.apple.com/ts01; do
  i=$((i+1)); t0=$(now)
  code=$(curl -s -m 30 -o r$i.tsr -w '%{http_code}' -H 'Content-Type: application/timestamp-query' --data-binary @q.tsq "$url")
  t1=$(now)
  status=$(openssl ts -reply -in r$i.tsr -text 2>/dev/null | awk -F': ' '/^Status:/{print $2; exit}')
  gen=$(openssl ts -reply -in r$i.tsr -text 2>/dev/null | awk -F': ' '/^Time stamp:/{print $2; exit}')
  tsa=$(openssl ts -reply -in r$i.tsr -text 2>/dev/null | awk -F': ' '/^TSA:/{print $2; exit}' | tr -d '"\\')
  # Verify: token covers our digest and chains to a root we hold (system store + FreeTSA's own CA).
  vout=$(openssl ts -verify -digest "$D" -in r$i.tsr -CAfile roots.pem 2>&1 | tail -1 | tr -d '"\\')
  size=$(stat -c %s r$i.tsr 2>/dev/null || echo 0)
  emit "{\"route\":\"rfc3161\",\"url\":\"$url\",\"http\":\"$code\",\"seconds\":$(echo "$t1-$t0"|bc),\"status\":\"${status:-}\",\"gen_time\":\"${gen:-}\",\"runner_utc\":\"$(date -u -d @${t1%.*} +%FT%TZ)\",\"verify\":\"$vout\",\"token_bytes\":$size,\"tsa\":\"${tsa:-}\"}"
done

# --- OpenTimestamps (Bitcoin-anchored; pending until a block confirms). ---
pip install -q opentimestamps-client >/dev/null 2>&1
python3 -c "import sys;open('digest.bin','wb').write(bytes.fromhex(sys.argv[1]))" "$D"
t0=$(now); ots stamp digest.bin >ots.log 2>&1; rc=$?; t1=$(now)
emit "{\"route\":\"opentimestamps\",\"exit\":$rc,\"seconds\":$(echo "$t1-$t0"|bc),\"proof_bytes\":$(stat -c %s digest.bin.ots 2>/dev/null || echo 0)}"

# --- Gist written by CI's own token (expected: refused, since GITHUB_TOKEN has no gist scope). ---
code=$(curl -s -o gist.json -w '%{http_code}' -X POST -H "Authorization: Bearer ${GH_TOKEN:-}" \
  -H 'Accept: application/vnd.github+json' https://api.github.com/gists \
  -d "{\"public\":false,\"description\":\"exagium anchor test\",\"files\":{\"digest.txt\":{\"content\":\"$D\"}}}")
emit "{\"route\":\"gist_via_ci_token\",\"http\":\"$code\",\"message\":\"$(python3 -c 'import json;print(json.load(open("gist.json")).get("message",""))' 2>/dev/null)\"}"
cat "$J"
