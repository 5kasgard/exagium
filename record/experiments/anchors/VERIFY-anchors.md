# Verifier brief: what actually anchors an honest record?

You're an independent verifier for exagium (https://github.com/5kasgard/exagium, public). Don't trust the author. Your job:
- find out which outside anchor does the job described below, or whether any does;
- follow the question down to its base, even if the answer is "none of these", "a different design", or "an anchor isn't the real fix".

Report what you measured, not what you expect.

## The job
exagium records claims about measurements and models, together with the verdict on each claim. The owner's rule is that **results can never be lied about**. The record is an append-only log in a public git repo, written by CI. An *anchor* is something outside the owner's control that a stranger can check.

Treat the owner, and anyone holding the repo or the CI token, as the adversary. A good anchor lets a stranger with no trust in the owner check each of these:

| # | Requirement | Question a stranger must be able to answer |
|---|---|---|
| R1 | Existence by time | Did record D exist no later than time T? |
| R2 | No backdating | Could the owner have made D after T and claimed T? |
| R3 | No silent edit/removal | If D was changed or deleted later, will I notice? |
| R4 | Completeness | Am I seeing *every* record, including the unfavourable ones? |
| R5 | No equivocation | Does everyone see the same history? |
| R6 | Longevity | Will this still verify in 10–20 years, if the provider dies or its certs expire? |
| R7 | Practical | Is it free, automatable from CI and VMs, free of personal names, checkable from a phone, and verifiable in another language? |

**R4 and R5 are the hard part.** The author's own finding is that every route tested proves existence, R1 to R3. None proves completeness. A hash chain doesn't help against an owner who keeps two chains, runs privately, or re-runs until a result is good. Find out what does, if anything.

## What's already measured (reproduce it, then break it)
- **Results:** `record/experiments/anchors/RESULTS.md` on branch `anchor-test`.
- **Script:** `try_anchors.sh`. **Workflow:** `.github/workflows/anchor-test.yml`.
- **Raw results:** branch `ci-runs`, `anchors/<run_id>/`. They include the RFC 3161 tokens (`r*.tsr`), the query, `digest.txt`, OTS proofs and the Sigstore bundle (`attestation.jsonl`).
- **The digest anchored everywhere:** `fe8d93a9dfb783350aa23a0c4471020a1b6cf9a17f6e5bd317b748e045b31de8`, the sha256 of `runs/count-2026-11-05T073638Z-0feb5e325c25.json` on `main`.

The author's claims, in short:
- **Repo and gist:** both under one owner, so no independence. The gist is the weaker of the two: it has no activity log, and CI's token gets a 403 when writing one.
- **RFC 3161:** six free TSAs grant and verify. A token fails after its certificate expires unless re-stamped.
- **Sigstore:** the attestation sits in public Rekor, with the identity bound to repo, workflow, ref and commit. sigstore-python verifies it offline.
- **OpenTimestamps:** works, but takes hours to confirm and needs Bitcoin data to verify.
- **GH Archive and Software Heritage:** neither holds anything of ours.

The proposed next design, a hypothesis you should test: a hash-chained log in git, with CI as its only writer. Each chain head gets a Sigstore attestation plus tokens from two RFC 3161 TSAs.

## Tasks
**1. Reproduce.** Verify the committed tokens, OTS proofs and Sigstore bundle on your machine, without trusting the author's verify output.
- Record exact commands and tool versions.
- The OTS proofs were pending at commit time; they should be complete by now. Upgrade and verify them, and say how: a node, block headers or an explorer.

**2. Attack each route as a dishonest owner.** Use your own throwaway public repo or fork, since you can't push to exagium. For every route, try to:
- **a. Backdate:** make a record now and get the anchor to say an earlier time.
- **b. Remove or edit:** drop an unfavourable record and change another so a stranger doesn't notice. Include: force push, deleting and recreating the repo, deleting a GitHub attestation through the API, deleting a gist, not showing a token.
- **c. Equivocate:** keep two histories and anchor both. Can a stranger find out?
- **d. Withhold:** run privately, or re-run until good, and anchor only the good run. Can a stranger find out?
- **e. Abuse the identity:** can another workflow, branch or fork produce an attestation that passes a verifier who checks only the repo?

Score each route against R1–R7: **pass**, **fail** or **conditional**, with the condition. Give the evidence for each.

**3. Dig to the base.** For R4 and R5, research and test, where feasible, what actually gives completeness and non-equivocation. Candidates are below; add any we missed.
- **Transparency logs with witness cosigning:** Certificate Transparency's gossip, sigsum, transparency-dev witnesses, Rekor monitors such as rekor-monitor watching a workflow identity. Can a stranger list *every* Rekor entry for one workflow identity? How?
- **Pre-registration:** anchor the test plan before the run, so a plan without a published verdict is visible. Does that close withholding? What about re-runs?
- **CI as the only writer,** with branch protection, required workflows and the activity log. What can the repo owner still do?
- **Independent archives as witnesses:** Software Heritage "save code now", Zenodo deposits, Internet Archive.
- **Bitcoin directly,** versus via OpenTimestamps.
- **Reframing.** Maybe R4 can't be had from an anchor at all, and the real defence is independent re-measurement: anyone re-runs the pre-registered test and gets the same verdict. Is that enough, and what remains?

**4. Longevity and practice.**
- Test RFC 3161 tokens with `openssl ts -verify` and `-attime` past the cert's expiry.
- Find what re-stamping (RFC 4998 evidence records) would take.
- Check terms of service and rate limits for the free TSAs and for Sigstore's public instance, including the move from Rekor v1 to v2.
- Check whether each route can be verified from a phone browser.
- Check how a non-Python verifier would recompute a record digest. Python's `json.dumps(sort_keys)` isn't a cross-language canonical form; RFC 8785 JCS is the candidate.

## Rules
- Don't spam free services: at most about 20 requests to any one TSA or calendar in total.
- Put nothing personal in what you anchor. Use digests of test files only.
- Don't push to exagium, and don't open issues there.
- If a claim of the author's is wrong, say exactly which one and show it.
- If you can't test something, say so and why. Don't infer a pass.

## Report (return as a zip)
- `report.md`:
  - a requirement × route table (pass, fail or conditional, with evidence);
  - your attacks and their outcomes;
  - corrections to the author's claims;
  - what you found at the base of R4 and R5;
  - one recommended design with its remaining known holes, and the cheapest experiment that would test it.
- Every script, command log and artifact you used. Attack files are especially wanted, since they become regression tests.
- Your machine and tool versions: OS, openssl, sigstore, ots, gh, python.
