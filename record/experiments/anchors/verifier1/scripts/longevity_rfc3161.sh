#!/bin/bash
# Longevity test for every RFC 3161 token.
# For each token: extract the TSA leaf cert's notAfter, then run openssl ts -verify
# at three reference times: token's own time, the day after cert expiry, and 2050.
set -u
DIGEST=fe8d93a9dfb783350aa23a0c4471020a1b6cf9a17f6e5bd317b748e045b31de8
ROOT=/home/z/my-project/upload/exagium-anchors/anchors
OUT=/home/z/my-project/anchors-verify/artifacts/rfc3161_longevity.tsv
echo -e "run_id\tarch\ttoken\turl\ttsa_subject\tcert_notBefore\tcert_notAfter\ttoken_time\tverify_at_token_time\tverify_at_day_after_expiry\tverify_at_2050" > "$OUT"
declare -A URLS
URLS[r1]=freetsa ; URLS[r2]=digicert ; URLS[r3]=sectigo ; URLS[r4]=ssl.com ; URLS[r5]=globalsign ; URLS[r6]=dfn ; URLS[r7]=certum ; URLS[r8]=apple
for run in 41995168381 41995230106 41995534904 41995975317 ; do
  for arch in anchors-ubuntu-24.04 anchors-ubuntu-24.04-arm ; do
    d="$ROOT/$run/$arch"
    [ -d "$d" ] || continue
    for i in 1 2 3 4 5 6 7 8 ; do
      f="$d/r$i.tsr"
      [ -s "$f" ] || { echo -e "$run\t$arch\tr$i\t${URLS[r$i]}\t-\t-\t-\t-\tMISSING\tMISSING\tMISSING" >> "$OUT" ; continue ; }
      # Extract the TSA cert chain from the token
      tok_time_str=$(openssl ts -reply -in "$f" -text 2>/dev/null | awk -F': ' '/^Time stamp:/{print $2; exit}')
      tok_time_epoch=$(date -u -d "${tok_time_str}" +%s 2>/dev/null)
      # extract TSA leaf cert (first cert in the SignedData certs)
      openssl ts -reply -in "$f" -token_out -out /tmp/tok$i.bin 2>/dev/null
      nb=$(openssl pkcs7 -inform DER -in /tmp/tok$i.bin -print_certs 2>/dev/null | awk '/BEGIN CERT/{n++; print > "/tmp/cert"i"."n} /END CERT/{n--}' i=$i ; n=1 ; awk '/BEGIN CERT/{n++; print > "/tmp/cert"i"."n} /END CERT/{n--}' i=$i /dev/stdin < <(openssl pkcs7 -inform DER -in /tmp/tok$i.bin -print_certs 2>/dev/null))
      # Just take the first cert in the chain (the signer leaf)
      openssl pkcs7 -inform DER -in /tmp/tok$i.bin -print_certs 2>/dev/null | awk '/BEGIN CERTIFICATE-----/{f=1} f{print} /END CERTIFICATE-----/{f=0; exit}' > /tmp/leaf$i.pem
      subj=$(openssl x509 -in /tmp/leaf$i.pem -noout -subject 2>/dev/null | sed 's/^subject=//')
      nb=$(openssl x509 -in /tmp/leaf$i.pem -noout -startdate 2>/dev/null | sed 's/notBefore=//')
      na=$(openssl x509 -in /tmp/leaf$i.pem -noout -enddate 2>/dev/null | sed 's/notAfter=//')
      na_epoch=$(date -u -d "${na}" +%s 2>/dev/null)
      day_after=$((na_epoch + 86400))
      yr2050=2524608000
      v1=$(openssl ts -verify -digest "$DIGEST" -in "$f" -CAfile "$d/roots.pem" -attime "$tok_time_epoch" 2>&1 | tail -1 | tr -d '"\\')
      v2=$(openssl ts -verify -digest "$DIGEST" -in "$f" -CAfile "$d/roots.pem" -attime "$day_after" 2>&1 | tail -1 | tr -d '"\\')
      v3=$(openssl ts -verify -digest "$DIGEST" -in "$f" -CAfile "$d/roots.pem" -attime "$yr2050" 2>&1 | tail -1 | tr -d '"\\')
      echo -e "$run\t$arch\tr$i\t${URLS[r$i]}\t${subj}\t${nb}\t${na}\t${tok_time_str}\t${v1}\t${v2}\t${v3}" >> "$OUT"
      rm -f /tmp/tok$i.bin /tmp/leaf$i.pem /tmp/cert$i.*
    done
  done
done
cat "$OUT"
