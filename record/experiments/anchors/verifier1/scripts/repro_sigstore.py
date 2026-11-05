#!/usr/bin/env python3
"""Reproduce Sigstore bundle verification offline.

For each CI run that produced a bundle, three checks are run:
  strict: --cert-identity=workflow@refs/heads/anchor-test --repository=5kasgard/exagium --ref --trigger --offline
  loose:  same but WITHOUT --ref and --trigger (tests the "abuse identity" attack surface)
  wrong:  strict params but a wrong digest (sanity check that verify actually checks the digest)
"""
import json, subprocess, os, pathlib, tempfile
ROOT = pathlib.Path("/home/z/my-project/upload/exagium-anchors/anchors")
OUT  = pathlib.Path("/home/z/my-project/anchors-verify/artifacts/sigstore_repro.tsv")
DIGEST = "fe8d93a9dfb783350aa23a0c4471020a1b6cf9a17f6e5bd317b748e045b31de8"
IDENT  = "https://github.com/5kasgard/exagium/.github/workflows/anchor-test.yml@refs/heads/anchor-test"
REPO   = "5kasgard/exagium"
REF    = "refs/heads/anchor-test"
TRIG   = "push"

def run_verify(bundle_path, *, digest, with_ref, with_trigger):
    cmd = ["python3","-m","sigstore","verify","github",
           "--bundle", str(bundle_path),
           "--cert-identity", IDENT,
           "--repository", REPO]
    if with_ref:     cmd += ["--ref", REF]
    if with_trigger: cmd += ["--trigger", TRIG]
    cmd += ["--offline", f"sha256:{digest}"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    msg = (r.stdout + r.stderr).strip().splitlines()
    return ("OK" if r.returncode==0 else "FAIL"), (msg[-1] if msg else "")

rows = []
for run in sorted(os.listdir(ROOT)):
    for arch in ("anchors-ubuntu-24.04","anchors-ubuntu-24.04-arm"):
        b = ROOT/run/arch/"attestation.jsonl"
        if not b.exists() or b.stat().st_size == 0:
            rows.append((run, arch, "MISSING","","",""))
            continue
        with tempfile.NamedTemporaryFile(suffix=".sigstore.json", delete=False) as tf:
            tf.write(b.read_bytes()); tmp = tf.name
        strict, m1 = run_verify(tmp, digest=DIGEST,     with_ref=True,  with_trigger=True)
        loose,  m2 = run_verify(tmp, digest=DIGEST,     with_ref=False, with_trigger=False)
        wrong,  m3 = run_verify(tmp, digest="0"*64,     with_ref=True,  with_trigger=True)
        # extract the source commit from the bundle's certificate (Fulcio OID 1.3.6.1.4.1.57264.1.12)
        try:
            bundle = json.loads(b.read_text())
            import base64
            cert_b64 = bundle["verificationMaterial"]["certificate"]["rawBytes"]
            cert_pem = "-----BEGIN CERTIFICATE-----\n" + base64.b64encode(base64.b64decode(cert_b64)).decode() + "\n-----END CERTIFICATE-----\n"
            with tempfile.NamedTemporaryFile(suffix=".pem", delete=False, mode="w") as cf:
                cf.write(cert_pem); cert_file = cf.name
            t = subprocess.run(["openssl","x509","-in",cert_file,"-noout","-text"], capture_output=True, text=True).stdout
            commit = ""
            for line in t.splitlines():
                if "Source Repository" in line and "URI" in line:
                    pass
                # The Fulcio OIDs map to ASN.1 string values; openssl prints them as UTF8String
            # The Sigstore cert encodes source repo, ref, commit as OID-tagged extensions.
            # Easier: pull the SAN URI (workflow path + ref), and use sigstore-python's debug.
            san_lines = [l.strip() for l in t.splitlines() if "URI:" in l]
            commit_line = [l for l in t.splitlines() if "github.com/5kasgard/exagium" in l]
            commit = "|".join(commit_line[:2])
            os.unlink(cert_file)
        except Exception as e:
            commit = f"ERR:{e}"
        rows.append((run, arch, strict, loose, wrong, m1, commit))
        os.unlink(tmp)

with open(OUT, "w") as f:
    f.write("run_id\tarch\tstrict_offline\tloose_no_ref_no_trigger\twrong_digest\tstrict_msg\tcert_claims\n")
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")
print(open(OUT).read())
