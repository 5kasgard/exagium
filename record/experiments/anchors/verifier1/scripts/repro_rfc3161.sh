#!/bin/bash
# Reproduce RFC 3161 token verification for every CI run.
# Per the brief: do NOT trust the author's verify output. Re-derive.
set -u
ROOT=/home/z/my-project/upload/exagium-anchors/anchors
OUT=/home/z/my-project/anchors-verify/artifacts/rfc3161_repro.tsv
SYS_CA=/etc/ssl/certs/ca-certificates.crt
DIGEST=fe8d93a9dfb783350aa23a0c4471020a1b6cf9a17f6e5bd317b748e045b31de8
echo -e "run_id\tarch\ttoken\turl\thttp\tt_bytes\tgen_time\trunner_utc\tverify_author_roots\tverify_system_ca" > "$OUT"
for run in 41995168381 41995230106 41995534904 41995975317 ; do
  for arch in anchors-ubuntu-24.04 anchors-ubuntu-24.04-arm ; do
    d="$ROOT/$run/$arch"
    [ -d "$d" ] || continue
    # url order matches try_anchors.sh
    urls=(https://freetsa.org/tsr http://timestamp.digicert.com http://timestamp.sectigo.com \
          http://ts.ssl.com http://timestamp.globalsign.com/tsa/r6advanced1 https://zeitstempel.dfn.de \
          http://time.certum.pl https://timestamp.apple.com/ts01)
    i=0
    for u in "${urls[@]}"; do
      i=$((i+1))
      f="$d/r$i.tsr"
      [ -s "$f" ] || { echo -e "$run\t$arch\tr$i\t$u\t-\t0\t-\t-\tMISSING\tMISSING" >> "$OUT" ; continue ; }
      bytes=$(stat -c %s "$f")
      # gen_time and tsa name from the token itself
      gen=$(openssl ts -reply -in "$f" -text 2>/dev/null | awk -F': ' '/^Time stamp:/{print $2; exit}')
      tsa=$(openssl ts -reply -in "$f" -text 2>/dev/null | awk -F': ' '/^TSA:/{print $2; exit}' | tr -d '"\\')
      # verify against (a) the author's roots.pem (system CA + FreeTSA CA downloaded at runtime),
      #                (b) the bare system CA store alone (to test "independent root" claim for FreeTSA)
      v_auth=$(openssl ts -verify -digest "$DIGEST" -in "$f" -CAfile "$d/roots.pem" 2>&1 | tail -1 | tr -d '"\\')
      v_sys=$(openssl ts -verify -digest "$DIGEST" -in "$f" -CAfile "$SYS_CA" 2>&1 | tail -1 | tr -d '"\\')
      echo -e "$run\t$arch\tr$i\t$u\t-\t$bytes\t${gen:-}\t-\t${v_auth}\t${v_sys}" >> "$OUT"
    done
  done
done
column -t -s $'\t' "$OUT"
