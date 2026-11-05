# Where should the record's outside anchor live? (item 3, experiment 1)

An anchor has to show that a record's digest existed by a given time, in a way the owner can't quietly backdate, edit or delete. This compares the candidates on that, measuring wherever possible. Run 2026-11-05 and checked by an independent reviewer agent; its corrections are folded in below.

**Method.**
- **Digest.** Every route anchors the same digest: the sha256 of `runs/count-2026-11-05T073638Z-0feb5e325c25.json`, which is `fe8d93a9…1de8`.
- **CI runs.** `try_anchors.sh` and `.github/workflows/anchor-test.yml` ran on the `anchor-test` branch, three times on two runners each (x86-64 and arm64). Results are on the `ci-runs` branch under `anchors/`.
- **Workspace checks.** Network reach, GH Archive, Software Heritage, offline verification of the CI tokens, and the public Rekor entries were checked from this workspace.
- **Caveat.** The first two runs' `attempts.jsonl` have invalid JSON numbers (`.22` with no leading zero, from `bc`). The values are readable, and the script is now fixed.

## Measured
| Route | Result |
|---|---|
| RFC 3161, 6 free TSAs | granted 4/4 each, verified |
| RFC 3161, DFN | granted 4/4, verify failed |
| RFC 3161, Apple | no connection 4/4 |
| OpenTimestamps | submitted 4/4, pending |
| Sigstore attestation | ok in every run with it, in Rekor |
| Gist via CI token | HTTP 403, 4/4 |
| GH Archive | none of our events |
| Software Heritage | repo not archived |

**Details:**
- **RFC 3161.** FreeTSA, DigiCert, Sectigo, SSL.com, GlobalSign and Certum all verified.
  - Each answered in 0.05–0.48 s with a 4.6–7.7 KB token, and each token's time matched the runner clock to within 1 s.
  - The DFN failure's cause wasn't captured; the script keeps only the last line.
  - FreeTSA's CA was downloaded from freetsa.org at run time, so it's trusted on first use, not an independent root. The other five chain to the system CA store.
  - **Offline re-check:** a DigiCert token re-verified in this workspace, and failed once one bit of the digest was flipped.
  - **After certificate expiry:** with `-attime` set to 2040, DigiCert's token fails ("certificate has expired"; its TSA cert ends 2037-11-04). Verifying at the token's own time passes, but that assumes the key wasn't compromised later. Long-term validity needs re-stamping before expiry, as in RFC 4998 evidence records.
- **OpenTimestamps.** Stamping took 1.7–5.7 s and gave a 0.7–0.9 KB proof. It's pending until a Bitcoin block confirms it, usually within hours.
  - The official client verifies against a Bitcoin node, though a pruned node is enough; in principle block headers alone are.
  - The opentimestamps.org web verifier works from a phone, at the cost of trusting that page.
  - It works from this workspace too.
- **Sigstore.** `actions/attest@v2` signed the digest with the workflow's identity.
  - Rekor's index lookup by this digest returns 4 entries, and the GitHub API lists 4 attestations. Entry 3692913518 has integratedTime 2026-11-05T08:32:34Z.
  - The certificate names the repo, `anchor-test.yml@refs/heads/anchor-test`, commit `4ea7f2c…` and run 41995230106, and is valid for 10 minutes.
  - **Verified without GitHub** by the reviewer, with sigstore-python 4.5.1, both online and with `--offline`:
    `python3 -m sigstore verify github --bundle b.sigstore.json --cert-identity "https://github.com/5kasgard/exagium/.github/workflows/anchor-test.yml@refs/heads/anchor-test" --repository 5kasgard/exagium --ref refs/heads/anchor-test --sha <commit> --trigger push sha256:<digest>`.
    A flipped digest or a wrong repository fails.
- **Gist via CI token.** Refused with "Resource not accessible by integration", so a gist needs a personal token stored as a CI secret, or a manual step.
- **GH Archive.** It now holds about 4–15 k events an hour, a small sample of GitHub. None of our pushes appear in the 21 hours scanned, including 07:40–07:59 when we pushed; hour 8 wasn't published yet. Not a witness.
- **Software Heritage.** The repo isn't archived. "Save code now" is possible but untested.
- **GitHub repo activity.** `/repos/…/activity` returns GitHub's server-side push times without auth, including force pushes. Commit dates, by contrast, are set by the committer.
- **Cost:** every route here is free.

## What each route can and can't prove
| Route | Backdate | Delete or hide |
|---|---|---|
| Git commit/tag | owner (commit dates) | owner |
| Gist | owner | owner |
| RFC 3161 token | TSA only | owner (holds the token) |
| OpenTimestamps | nobody practical | owner (holds the proof) |
| Sigstore / Rekor | Sigstore only | detectable, not impossible |

**Notes on the table:**
- **Git.** A force push shows in the activity log, but deleting and recreating the repo wipes that log in one step. The activity log is GitHub's clock, so it's independent of the owner but not of GitHub.
- **Gist.** Revisions can be force-pushed away, there's no activity log, and CI can't write one.
- **RFC 3161.** You trust one company's key per token. It needs re-stamping before the cert expires.
- **OpenTimestamps.** It's slow, but there's no company to trust.
- **Sigstore.** The owner can delete GitHub's copy of an attestation through the API; the Rekor copy stays. Removing a Rekor entry would show up through inclusion proofs and signed checkpoints, but only if someone checks. The public-good instance has no SLA, and the move from Rekor v1 to v2 bears on how long it lasts. An attestation says "this workflow at this commit vouched for D at T", not that D's content is true.

**Findings:**
- **Repo plus gist gives no independence.** Both sit under one account and one custodian, and both can be rewritten by the same person. A gist is weaker than the repo: no activity log, and CI can't write to it.
- **Existence isn't completeness, and a hash chain alone doesn't fix that.** Every route proves only that a digest existed. A chain where each record includes the previous digest proves internal consistency, but the owner could still:
  - keep two forked chains and anchor both;
  - run privately and never append;
  - re-run until a good result comes up.

  Rekor can be searched by digest but not by identity, so a verifier can't list everything this workflow anchored without monitoring the log.
- **Mitigations to test:**
  - CI is the only writer.
  - Every run is anchored automatically, with a sequence number and the previous head.
  - The test plan is committed before the run (pre-registration).
  - Verifiers pin the workflow commit and ref.
  - An identity monitor watches Rekor for forks.

## Hypothesis for the next step (to test, not yet decided)
- **The log lives in git:** hash-chained records, canonically encoded (RFC 8785 JCS, or hash the stored bytes) so any language can recompute them. CI is the only writer.
- **Each chain head gets independent witnesses:**
  1. a Sigstore attestation from CI: public, identity-bound, verifiable without GitHub;
  2. RFC 3161 tokens from two commercial TSAs whose roots are in the system store: offline-verifiable, re-stamped before expiry;
  3. optionally, OpenTimestamps (no company to trust), and a periodic deposit outside GitHub (Software Heritage or Zenodo).

**Still to measure:** whether the OTS proofs complete and verify; whether `gh attestation verify` agrees with sigstore-python; and whether a Rekor identity monitor can see every anchor this workflow makes.
