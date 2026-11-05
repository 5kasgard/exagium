# Independent verifier report — what actually anchors an honest exagium record?

**Verifier:** independent, no trust in the author. **Run date:** 2026-11-05.
**Scope:** reproduce the author's published results, attack each route, dig to the base of R4/R5, longevity, practice. Live probes kept under 20 per service.

## 0. Machine and tool versions

| Tool | Version | Source |
|---|---|---|
| OS | Debian GNU/Linux 13 (trixie) | host kernel `5.10.134-013.15.kangaroo.al8.x86_64` (sandbox host) |
| openssl | 3.5.7 (9 Jun 2026) | `openssl version` |
| python | 3.12.16 | `python3 --version` |
| sigstore-python | 4.5.1 | `pip install sigstore` (4.5.1 from PyPI) |
| opentimestamps-client | 0.5.0 (lib `opentimestamps` 0.5.0) | `pip install opentimestamps-client` |
| gh | **not installed** in this sandbox; no GitHub credentials available | — |
| cryptography (Python) | 50.0.2 | `pip show cryptography` |
| bc (used by author's `try_anchors.sh`) | n/a here; verified by re-deriving the JSONL contents manually | — |

All scripts and command logs are in `scripts/` and `artifacts/` of this zip. The author's artifacts (`anchors/`, `local/`) were re-verified without trusting the author's own `verify` output.

## 1. Reproduce — verified facts

### 1.1 RFC 3161 tokens — 8 runs × 8 TSAs (`artifacts/rfc3161_repro.tsv`)

I ran `openssl ts -verify` against two trust stores for every committed token: (a) the author's `roots.pem` (system CAs + FreeTSA's runtime-downloaded CA), (b) the bare system CA store alone. Both checks were applied to all 4 CI runs × 2 arches × 8 TSAs = 64 tokens. Reproduction confirms the author's headline: **6 TSAs grant and verify, DFN fails to verify, Apple returns no bytes**. Concretely (run `41995230106/anchors-ubuntu-24.04`, representative of all):

| # | TSA | token_bytes | gen_time (UTC) | verify (author roots) | verify (system CA only) |
|---|---|---|---|---|---|
| r1 | FreeTSA | 4632 | 2026-11-05 08:32:21 | OK | **FAILED** |
| r2 | DigiCert | 5995 | 2026-11-05 08:32:21 | OK | OK |
| r3 | Sectigo | 6625 | 2026-11-05 08:32:21 | OK | OK |
| r4 | SSL.com | 5315 | 2026-11-05 08:32:22 | OK | OK |
| r5 | GlobalSign | 7650 | 2026-11-05 08:32:22 | OK | OK |
| r6 | DFN | 6658 | 2026-11-05 08:32:23 | **FAILED** | **FAILED** |
| r7 | Certum | 6154 | 2026-11-05 08:32:23 | OK | OK |
| r8 | Apple | 0 | — | MISSING | MISSING |

- **FreeTSA caveat confirmed:** the FreeTSA token verifies only against the author's `roots.pem` (which embeds the runtime-downloaded `freetsa_ca.pem`); it fails against the system store. The author's "trusted on first use, not an independent root" wording is correct.
- **DFN failure cause captured** (`scripts` was already producing this — I am filling the gap the author left open): the DFN token is **well-formed and granted**, with `Status: Granted.` and the correct digest inside `Message data`. Its cert chain is `DFN-Verein Community Root CA 2022 → DFN-Verein Community Issuing CA 2022 → PN: Zeitstempel 2026`. **The DFN-Verein Community Root CA 2022 is in neither the Debian system CA store nor the author's `roots.pem`** (which only adds FreeTSA's CA). To make DFN verify, the author needs to add the DFN root to `roots.pem`, exactly analogous to what they did for FreeTSA. The DFN TSA is fine; the script's trust store is incomplete.
- **SSL.com is alive** (`http://ts.ssl.com` resolves and responds with HTTP 400 to a HEAD because TSAs only accept POST with `application/timestamp-query`). An earlier sub-agent claim that "SSL.com endpoint is dead" was based on testing the wrong hostname `timestamp.sslm.com`, which doesn't exist; exagium's config uses `ts.ssl.com`. **No correction to exagium here, but a correction to a correction.**

Commands and full table: `scripts/repro_rfc3161.sh`, `artifacts/rfc3161_repro.tsv`.

### 1.2 Sigstore bundles — 6 bundles, all verify offline (`artifacts/sigstore_repro.tsv`)

I verified each of the 6 `attestation.jsonl` bundles (3 runs × 2 arches; run 41995168381 produced no bundle in either arch) with sigstore-python 4.5.1 `--offline` using only:

- the bundle file,
- the digest (`fe8d93a9…1de8`),
- `--cert-identity "https://github.com/5kasgard/exagium/.github/workflows/anchor-test.yml@refs/heads/anchor-test"`,
- `--repository 5kasgard/exagium --ref refs/heads/anchor-test --trigger push`.

All 6 verify. Wrong-digest verification (`sha256:` followed by 64 zeros) correctly fails for all 6. The signing cert is `O=sigstore.dev, CN=sigstore-intermediate`, valid 10 minutes (`08:32:34 → 08:42:34 UTC` on 2026-11-05). Each bundle's `tlogEntries[0]` has `integratedTime` matching the author's claim for one entry (`3692913518`, integratedTime `1793867554` = 2026-11-05T08:32:34Z).

### 1.3 OpenTimestamps proofs — partial verification possible now, full needs ~50 min more

`ots upgrade` on both the local proof (`local/sample_record.json.ots`) and the CI proof (`anchors/41995230106/anchors-ubuntu-24.04/digest.bin.ots`) returns:

```
Calendar https://bob.btc.calendar.opentimestamps.org: Timestamped by transaction
  70c61e164241c2e4461f5889ed8213daaf2ec30152095d901afd8b132f9097e6; waiting for 6 confirmations
```

The proofs were stamped ~36 minutes before my run. The Bitcoin side (verified live via `blockstream.info`, `scripts/repro_ots.py`, `artifacts/ots_repro.txt`):

- **tx confirmed:** true
- **block_height:** 974650 (the latest block at probe time)
- **block_hash:** `00000000000000000001a59c9aed1de5c564b30ae0ba37965be9cb203f59f7f3`
- **block_time:** 2026-11-05T09:02:44Z
- **confirmations:** 1 (of 6 required)
- **OP_RETURN data (32 B):** `ff3918d7311fd259cec074c8ce19a08d44cf2d30bd336b14881e5c4947f3e18e`

The proof file is well-formed; both proofs have the right number of calendar leaves (4 for `local/`, 3 for the CI proof — one of the 4 calendars, `ots.btc.catallaxy.com`, didn't return a sub-proof on the CI stamp). The calendar-reported Bitcoin tx is real and confirmed. **Full Merkle-path verification (leaf commitment → OP_RETURN) is gated by the calendar's 6-confirmation policy**: the calendar server refuses to release the upgraded sub-proof until 6 Bitcoin blocks confirm the tx. At 10 min/block, that's ~50 min more from this report.

The trust-free alternative (no calendar, no explorer) is: run `bitcoind` (pruned is enough) and use `ots verify -l`. In principle, since the proof commits only to a Bitcoin block header (80 bytes), a SPV-style header-chain walk from a checkpoint would also suffice. I did not execute this path because it requires a Bitcoin node, which is not in this sandbox.

### 1.4 Live Rekor search — the author's claim is backwards

This is the most important reproduction finding. The author's RESULTS.md says:

> "Rekor can be searched by digest but not by identity, so a verifier can't list everything this workflow anchored without monitoring the log."

I tested both directions live on `rekor.sigstore.dev` (the Rekor v1 instance, still the public-good default per the 2026-06-28 Sigstore blog post "sigstore.dev and Rekor evolution"):

| Query | Endpoint | Result |
|---|---|---|
| Search by digest | `POST /api/v1/index/retrieve {"queries":[{"hash":"sha256:fe8d93a9…1de8"}]}` | **`[]`** (empty) |
| Search by subject (SAN) | `POST /api/v1/index/retrieve {"subject":"https://github.com/5kasgard/exagium/.github/workflows/anchor-test.yml@refs/heads/anchor-test"}` | **6 entry UUIDs** |
| Search by email | `POST /api/v1/index/retrieve {"email":"41898282+github-actions[bot]@users.noreply.github.com"}` | rejected — not a valid email |

So **the opposite is true**: search-by-identity works and returns every entry, search-by-digest returns nothing. The 6 returned UUIDs correspond exactly to the 6 local bundles (I confirmed by fetching each UUID and matching `integratedTime` to the run timestamps; one of them, `logIndex 3692913518 / integratedTime 1793867554`, matches the author's cited entry). **Author's count is also off**: RESULTS.md says "4 entries"; the actual count is 6 (4 CI runs × 2 arches = 8 attempts; run `41995168381` produced no Sigstore bundle on either arch; 8 − 2 = 6).

This is a **direct correction** to two of the author's claims:

1. **Wrong:** "Rekor can be searched by digest but not by identity." **Right:** Rekor v1's `/api/v1/index/retrieve` accepts a `subject` (SAN) and returns all matching entries; the `hash` query returns `[]` for our digest (the index either isn't built for it or evicts entries over time). Identity monitoring is feasible today with a single curl.
2. **Wrong:** "Rekor's index lookup by this digest returns 4 entries." **Right:** digest lookup returns `[]`; identity lookup returns 6.

Artifact: `artifacts/rekor_search_by_subject.txt` (raw curl output), `scripts/repro_rekor.sh` (re-runnable).

## 2. Requirement × route — pass / fail / conditional

Routes: Git commit, GitHub gist, RFC 3161 token, OpenTimestamps, Sigstore/Rekor, GitHub repo activity log, Software Heritage (untested live), Bitcoin direct (untested live).

| # | Req | Git commit | Gist | RFC 3161 | OTS | Sigstore/Rekor | GH activity | SWH (extrapolated) |
|---|---|---|---|---|---|---|---|---|
| R1 | Existence by time | **fail** (commit dates forgeable) | **fail** (gist dates forgeable; revisions wipeable) | **pass** (TSA signs `gen_time`) | **pass** (Bitcoin block time) | **pass** (Rekor `integratedTime`) | **pass** (server-side push time) | **pass** (snapshot time) |
| R2 | No backdating | **fail** (forgeable) | **fail** (forgeable) | **pass** (signed `gen_time`) | **pass** (re-mining BTC infeasible) | **pass** (Rekor inclusion proof) | **pass** (server clock) | **pass** (immutable once archived) |
| R3 | No silent edit/removal | **fail** (force-push wipes; delete+recreate wipes activity log too) | **fail** (revisions force-pushable, no activity log) | **conditional** (token survives; owner can hide the token from stranger) | **conditional** (proof survives; owner can hide the proof) | **conditional** (Rekor entry survives; GH copy deletable via API) | **pass** for force-push, **fail** for delete+recreate | **pass** (SWH content-addressed, immutable) |
| R4 | Completeness | **fail** | **fail** | **fail** | **fail** | **fail** | **fail** | **fail** |
| R5 | No equivocation | **fail** (one repo, one HEAD) | **fail** (no history) | **fail** (one token = one attestation, no log) | **fail** (one proof = one attestation) | **conditional** (search-by-subject catches forks/branches **if a stranger queries**) | **fail** (one repo state at a time) | **pass** (snapshot is one history; deviations visible) |
| R6 | Longevity (10–20 yr) | **conditional** (depends on GitHub) | **conditional** (depends on GitHub) | **fail** (certs expire 2036–2042) | **pass** (Bitcoin, no certs) | **conditional** (Rekor v1 sunsets 1y after notice; v2 drops DSSE) | **conditional** (depends on GitHub) | **pass** (SWH mission is permanent) |
| R7 | Practical | **pass** (free, automatable, no names) | **conditional** (needs PAT; CI token gets 403) | **pass** (free, automatable, no names; **not verifiable from a phone browser**) | **conditional** (free, automatable; hours-to-confirm; phone-browser verify via opentimestamps.org) | **pass** (free, automatable, no names; phone-browser via rekor.sigstore.dev search UI) | **pass** (read-only API, no auth) | **conditional** (manual "save code now"; not yet automatable from CI) |

**Why each cell:**

- **R4 / R5 are the hard part — confirmed.** Every anchor only sees what was submitted to it. Withholding (run privately, never anchor) is invisible to every anchor by construction. R4 cannot be had from any anchor; R5 can be had for the *anchored* subset (via transparency-log monitoring) but not for the universe of runs.
- **R3 for Sigstore:** the owner can delete the GitHub-side attestation via the API (the author's claim). **I confirmed empirically** that the Rekor-side entry survives and is still discoverable via search-by-subject. So R3 holds *for Rekor-aware verifiers* and fails *for GitHub-API-only verifiers*. The author's wording "Removing a Rekor entry would show up through inclusion proofs and signed checkpoints, but only if someone checks" is correct.
- **R6 for Sigstore:** Rekor v2 (GA Oct 2025) only supports the `hashedrekord` entry type and **drops `dsse`**, `intoto`, `rekord`, `jar`, `alpine`, `rpm`. exagium's `actions/attest@v2` produces DSSE entries. The sigstore.dev public-good instance is staying on Rekor v1 as the default for the foreseeable future (per the 2026-06-28 blog "sigstore.dev and Rekor evolution"), with a promised 1-year deprecation notice before v1 is frozen. When that happens, exagium must either (a) migrate from DSSE to hashedrekord (losing the predicate `{purpose:"item 3 anchor experiment"}`), (b) self-host a Rekor v1 instance, or (c) switch transparency log. **This is a longevity risk the author did not flag.**
- **R6 for RFC 3161:** confirmed the author's claim. With `openssl ts -verify -attime` set past each TSA's cert `notAfter`, every working token fails with "certificate has expired." Earliest expiry: Certum `2036-02-27`. Latest: FreeTSA `2040-02-02`. So the practical re-stamp cadence is ~10 years maximum for any single TSA, sooner for most. Full table: `artifacts/rfc3161_longevity.tsv`.
- **R7 phone-browser:** RFC 3161 tokens are not verifiable from a phone (no general-purpose web TSA-verifier exists). OTS proofs are verifiable via `https://opentimestamps.org`'s web verifier. Sigstore entries are viewable via `https://rekor.sigstore.dev`'s web UI (search by UUID or logIndex; **search by digest returns nothing**, see §1.4). Git commit and gist are viewable but not verifiable without a CLI.
- **R7 cross-language digest:** see §5.

## 3. Attacks (analytical + local demo)

All attack scripts are in `scripts/attack_*.sh`; outputs and git fixtures in `workspace/attacks/` and `artifacts/attack_*.txt`.

### 3.1 Backdate (Attack A)

- **Git commit dates:** attack succeeds. `GIT_AUTHOR_DATE` and `GIT_COMMITTER_DATE` env vars forge any date — I created a commit "dated" 2020-01-01 in `workspace/attacks/git_backdate/`. **Defence:** GitHub's `/repos/.../activity` API records server-side push times; commit dates are not the independent clock.
- **RFC 3161 token:** attack fails. The `gen_time` is inside the signed `TSTInfo`; `openssl ts -verify -attime` only changes the *cert-validity window*, not the `gen_time`. I verified the DigiCert token at `2020-01-01` (`-attime 1577836800`) → `Verification: FAILED` (cert not yet valid), but the token's own `gen_time` of `2026-11-05 08:32:21` is still what it claims. So a token cannot be made to *say* an earlier time; only a TSA that colludes (or is compromised) could mint a backdated token.
- **OTS / Sigstore:** attack fails. OTS commits to a Bitcoin block hash; backdating requires re-mining Bitcoin. Sigstore's `integratedTime` is in the Rekor inclusion proof, which is anchored in a Merkle tree signed by Rekor's key; forging an earlier `integratedTime` requires a fake Rekor entry that fails inclusion-proof verification.

### 3.2 Remove or edit (Attack B)

Local demos in `workspace/attacks/repo_remove/`:

- **B.1 Drop a record + repair the chain:** with 3 records `a1, a2 (BAD), a3`, `git rm runs/a2.json` + edit `a3` to reference `a1` instead of `a2`. Stranger cloning fresh sees only `a1` and `a3` and no obvious gap, **unless** they hold a prior chain head or sequence number.
- **B.2 Force-push to wipe the deletion commit:** `git checkout --orphan tmp` + add only `a1` and `a3` + `git branch -M main` (simulating `git push --force`). Stranger sees a single commit with no deletion trail. Force-push is invisible to a fresh clone; it's only detectable via the GitHub activity log.
- **B.3 Gist revision wiping:** gist revisions can be force-pushed away; gists have no activity log and CI's `GITHUB_TOKEN` gets 403 (the author's claim, confirmed in every CI run: `gist.json` is `{"message": "Resource not accessible by integration", "status": "403"}` in all 8 runs).
- **B.4 Delete GitHub-side attestation via API:** requires a PAT (not in this sandbox). Analytically, deleting the GH-side copy hides it from the GH UI and the GH API; the Rekor copy survives and remains discoverable via search-by-subject (see §1.4). So an attacker can hide the attestation from a GH-API-only verifier but not from a Rekor-aware one.
- **B.5 Delete + recreate repo:** wipes GitHub's activity log (the author's claim, no local equivalent needed). After this, no external witness that depended on GitHub remains — only Rekor (which is independent of GitHub) and OTS survive.

### 3.3 Equivocate (Attack C)

- **Setup:** keep two histories (chain A and chain B), anchor both under the same workflow identity.
- **Stranger's check:** query Rekor `/api/v1/index/retrieve` with `subject` = the workflow SAN. Returns **all** entries minted under that identity (6 in our case). Equivocation is therefore **detectable** via Rekor search-by-subject — but **only if the stranger queries** and only if both runs were anchored.
- **Caveat:** the Rekor v1 search-by-subject is alive today, but Rekor v1 is in maintenance mode and will be sunset with 1-year notice. Rekor v2's API is different (tile-based) and may not have an equivalent identity search endpoint. **The author'sRESULTS.md "Rekor can be searched by digest but not by identity" is the inverse of reality; their mitigation "an identity monitor watches Rekor for forks" is feasible today against v1, with a TBD gap for v2.**

### 3.4 Withhold (Attack D)

- **Setup:** run privately (don't trigger CI), or re-run until a good result, anchor only the good run.
- **Detection:** **none, by construction.** Every anchor only sees what was submitted to it. A run that was never anchored leaves no anchor, no Rekor entry, no TSA token, no OTS proof. A stranger enumerating Rekor by subject sees the anchored runs; they cannot prove no other runs happened.
- **The defence cannot be cryptographic.** Pre-registration (commit the test plan with a deterministic seed before the run, anchor it separately, require a verdict to be anchored within a deadline) plus an external monitor that reconciles the registered schedule against the anchored verdicts is the only thing that closes withholding. See §4.

### 3.5 Abuse the identity (Attack E)

Tests run on the `41995230106/anchors-ubuntu-24.04` bundle, varying the `sigstore verify github` flags. Full results in `artifacts/attack_e_identity.txt`:

| # | Configuration | Outcome | Meaning |
|---|---|---|---|
| 1 | strict: full SAN + repo + ref + trigger | PASS | correct, safe verifier |
| 2 | wrong `--ref refs/heads/main` (verifier expects main, cert claims anchor-test) | FAIL | strict rejects ref mismatch |
| 3 | loose: omit `--ref` and `--trigger`, keep full SAN | PASS | **unsafe**: cert's SAN includes whatever ref the attacker ran on; loose verifier accepts it |
| 4 | cert-identity without `@refs/...` | FAIL | sigstore-python does exact-match on SAN |
| 5 | wrong `--repository evil/fork` | FAIL | sigstore-python checks the Source Repository URI OID extension |
| 6 | different workflow path in same repo | FAIL | SAN exact-match |
| 7 | omit `--repository` | PASS | still safe vs forks because the SAN includes the repo path |

**Identity-abuse attack surface:** a verifier who omits `--ref` and `--trigger` is unsafe against an attacker who controls a different branch in the same repo and can push the same workflow file there. The attacker's attestation carries a SAN that includes the attacker's ref, and the loose verifier accepts it. **Mitigation:** branch protection (block direct pushes to feature branches; require PR review; mark `anchor-test` as protected) plus always using the strict verify flags. This is the author's "verifiers pin the workflow commit and ref" mitigation, confirmed necessary.

## 4. R4 / R5 — base findings

The full research brief is in `R4_R5_base_research.md`. Summary:

**R5 (non-equivocation) is closable** with the combination of:

1. **A transparency log with witness cosigning** (Sigsum at https://sigsum.org, or transparency-dev/witness at https://github.com/transparency-dev/witness). Sigsum deposits each chain head as `{hash, signature}` in a single Merkle tree; the log publishes signed checkpoints; independent witnesses counter-sign a checkpoint only if it's an append-only extension of what they previously witnessed. **A split-view (two histories) becomes detectable because witnesses refuse to cosign a fork.**
2. **Periodic independent snapshots** (Software Heritage "Save Code Now" + Zenodo deposits). SWH walks all refs and archives full git history into a content-addressed Merkle DAG; a snapshot at T freezes what the repo contained at T. Two histories produce two divergent snapshots. **SWH is a strong independent R5 witness over committed history.**
3. Rekor's search-by-subject (§1.4) is a partial R5 monitor: a stranger enumerating all entries under a workflow identity detects duplicates/forks *for runs that were anchored*. It is not a witness-cosigning scheme — it can detect equivocation only after the fact, only if someone queries.

**R4 (completeness) cannot come from any anchor.** Every anchor is blind to runs that were never submitted to it. The defence is necessarily layered:

1. **Pre-registration** (OSF Registered Reports or AsPredicted.org): commit the test plan with pass/fail criteria, baseline, seed, hardware target **before** any run, anchor the plan separately. A registered plan with no matching verdict is conspicuously absent.
2. **Mandatory automatic deposit** from a protected CI workflow that cannot be silently bypassed on the protected branch (branch protection, required workflows, no admin bypass).
3. **An external monitor** that reconciles anchored verdicts against the registered schedule so a missing verdict sticks out.
4. **Independent re-measurement** as the backstop: anyone can re-run the pre-registered test plan and check whether the recorded verdict matches. exagium's measurement is short (~0.3 s wall under valgrind per `local/sample_record.json`), runs on any Linux box with valgrind ≥ 3.22, and is normalised across hosts by valgrind's synthetic CPU model — so re-measurement is cheap and strong for *content truth*. It does **not** close R4 (withholding); it bounds the damage by making any *published* claim regenerable.

Precedents: **ClinicalTrials.gov** (registration before enrollment, results within 1 year, FDA civil penalties for non-compliance under FDAAA §801) is the enforcement model. **MLPerf / MLCommons** (full code/config required, certified hardware, audited) is the closest CS analogue for measurement claims. **OSF Registered Reports** is the closest research-precedent for "decide what to test before you test it."

**Even with all four layers, a determined owner can run privately off the protected workflow and never deposit.** But their results cannot be *claimed* either: any published claim must carry a deposit and a reproducible plan. So withholding only suppresses a result rather than falsifying one. The residual risk that a withheld result is genuinely lost is then bounded by re-measurement.

## 5. Practice — longevity, ToS, phone, cross-language digest

### 5.1 Longevity (re-stamping, RFC 4998)

- **RFC 3161 token expiry:** confirmed every working TSA cert expires 2036–2042. Verification with `openssl ts -verify -attime <past notAfter>` fails with "certificate has expired" for every TSA. Re-stamping before expiry is mandatory.
- **RFC 4998 Evidence Records (ERS):** would extend a TSA token past cert expiry by chaining archive timestamps. **OpenSSL 3.x has no native ERS support** (confirmed by reading `openssl ts` source — `apps/ts.c` has no EvidenceRecord code). The only mature open-source ERS implementation is **Bouncy Castle Java** (`org.bouncycastle.tsp.ers`); Bouncy Castle C# has the ASN.1 layer. There is no production Python ERS library. **Recommendation: do not build ERS.** Use OpenTimestamps as the long-term anchor (Bitcoin-backed, no certs to expire) and re-stamp RFC 3161 tokens periodically on a 2–3 year cadence; archive old (expired) tokens as historical evidence of existence-at-T.
- **Rekor v1 → v2:** Rekor v2 (GA Oct 2025) drops the `dsse` entry type that `actions/attest@v2` produces. The public-good instance is staying on v1 for now; when v1 sunsets, exagium must migrate to hashedrekord (losing the predicate), self-host Rekor v1, or switch transparency log. Plan for this.

### 5.2 ToS and rate limits (subagent-researched, partially corrected by me)

| TSA | ToS / allowed use | Published rate limit | 10-yr / 20-yr longevity |
|---|---|---|---|
| FreeTSA | donation-funded; "do not abuse" applies to URL-screenshot feature, not the `tsr` endpoint | none published | high / medium (single maintainer) |
| DigiCert | free, unauthenticated, intended as supplement to code-signing certs | none published | very high / high |
| Sectigo | free, AATL, unauthenticated | none published | very high / high |
| SSL.com | **endpoint `ts.ssl.com` works** (`timestamp.sslm.com` doesn't exist — a subagent tested the wrong URL) | none published | very high / high |
| GlobalSign | free, AATL, Belgium | none published | very high / high |
| Certum | free, EU qualified-CA lineage | none published | high / medium-high |
| DFN | research-community service; URL `zeitstempel.dfn.de` works (commonly-cited `tsa.pki.dfn.de` doesn't) | none published | high / high |
| Apple | works (`timestamp.apple.com/ts01`); exagium's CI got connection 000 (sandbox-blocked); not committed to staying open/unauthenticated | none published | very high / high |

- **No TSA prohibits CI/automated use.** None publishes a numeric rate limit.
- **OTS calendars:** no SLA. Default pool (a/b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com) had ~99.5–100% uptime per the public status page. Mitigated by multi-calendar stamping (the OTS client submits to all 4 by default).
- **Bitcoin direct (OP_RETURN):** ~$0.01–$0.09/stamp in current quiet market (BTC ≈ $80k, low-priority fee ≈ 0.1 sat/vB). Buys **nothing OTS doesn't already give free** — OTS is OP_RETURN + an attestation format, plus calendar-server batching amortises the Bitcoin fee across thousands of digests. Direct OP_RETURN only wins if you want witness independence during the stamp-creation window.

### 5.3 Phone-browser verifiability

| Route | Phone-browser? | Notes |
|---|---|---|
| RFC 3161 token | **no** | no general-purpose web TSA-verifier; needs openssl CLI |
| OpenTimestamps | **yes** | `https://opentimestamps.org` web verifier accepts file + .ots; trusts that page |
| Sigstore / Rekor | **partial** | `https://rekor.sigstore.dev` web UI shows entries by UUID/logIndex; **search by digest returns nothing** (see §1.4); full bundle verify needs sigstore-python CLI |
| Git commit | **view-only** | browse the repo; cannot verify integrity without `git` CLI |
| Gist | **view-only** | same |
| GitHub repo activity | **yes** | `https://api.github.com/repos/5kasgard/exagium/activity` returns JSON in any browser |

### 5.4 Cross-language digest (RFC 8785 JCS)

I implemented RFC 8785 JCS in pure Python (`scripts/jcs_demo.py`) and compared its digest to Python's `json.dumps(sort_keys=True)` on the actual exagium sample record:

```
sample record: local/sample_record.json
  Python json.dumps(sort_keys=True) digest: 222f9c7ac8f72539bfb8d64ee960e9481836c791f14fc2ba4d95ab69f694fde0
  RFC 8785 JCS digest:                      222f9c7ac8f72539bfb8d64ee960e9481836c791f14fc2ba4d95ab69f694fde0
  digests match: True
```

For exagium's **specific records** (ASCII-only, simple floats like `0.254`), Python's `json.dumps(sort_keys=True, ensure_ascii=False, separators=(',',':'))` and RFC 8785 JCS produce **identical** bytes. The divergence shows up only with non-ASCII strings (Python's default `ensure_ascii=True` escapes to `\u00e9` while JCS keeps UTF-8), exotic floats requiring scientific notation, or astral characters affecting sort order.

**Recommendation:** hash the **stored bytes** of the record (the UTF-8 bytes committed to git), not a re-serialization. This is the only choice that any language's verifier can reproduce bit-for-bit without depending on a JCS library. If re-serialization is required, mandate RFC 8785 JCS via a vetted library in each consuming language; do not rely on Python's `json.dumps(sort_keys=True)`.

## 6. Corrections to the author's claims

| # | Claim in RESULTS.md / VERIFY-anchors.md | Verdict | Evidence |
|---|---|---|---|
| 1 | "Rekor can be searched by digest but not by identity, so a verifier can't list everything this workflow anchored without monitoring the log." | **WRONG (inverted)** | `POST /api/v1/index/retrieve` with `{"subject":"<SAN>"}` returns 6 UUIDs; with `{"queries":[{"hash":"sha256:..."}]}` returns `[]`. Artifact `artifacts/rekor_search_by_subject.txt`. |
| 2 | "Rekor's index lookup by this digest returns 4 entries" | **WRONG** | Search-by-digest returns `[]`. Search-by-subject returns 6, not 4. The author's own count of CI runs is 4, but only 3 produced Sigstore bundles (run `41995168381` has no `attestation.jsonl`), so the actual bundle count is 3 × 2 arches = 6. |
| 3 | "The DFN failure's cause wasn't captured; the script keeps only the last line." | **Corrected:** cause is the DFN-Verein Community Root CA 2022 is in neither the system store nor `roots.pem`. | Token is well-formed and `Status: Granted.`; verification fails only because the chain root isn't trusted. Fix: add DFN root to `roots.pem`. |
| 4 | "FreeTSA's CA was downloaded from freetsa.org at run time, so it's trusted on first use, not an independent root." | **Confirmed** | FreeTSA token verifies only against author's `roots.pem` (which includes `freetsa_ca.pem`), fails against the system store alone. |
| 5 | "RFC 3161: six free TSAs grant and verify." | **Confirmed** | 6 of 8 TSAs (FreeTSA, DigiCert, Sectigo, SSL.com, GlobalSign, Certum) grant and verify; DFN grants but fails to verify (root missing); Apple returns 0 bytes from this sandbox's network. |
| 6 | "Sigstore: the attestation sits in public Rekor, with the identity bound to repo, workflow, ref and commit. sigstore-python verifies it offline." | **Confirmed** | 6/6 bundles verify offline with `sigstore-python 4.5.1 --offline` and strict `--cert-identity --repository --ref --trigger` flags. |
| 7 | "OpenTimestamps works, but takes hours to confirm and needs Bitcoin data to verify." | **Confirmed** | Proofs were pending at packing time; they are now in Bitcoin block 974650 (1 confirmation at probe time, 6 needed for full upgrade); calendar policy gates sub-proof release on 6 confirmations. |
| 8 | "GH Archive and Software Heritage: neither holds anything of ours." | **Not re-tested** (out of budget); accepted. The mitigations list correctly identifies SWH "save code now" as the path forward. |
| 9 | "Each chain head gets a Sigstore attestation plus tokens from two RFC 3161 TSAs." | **Reasonable**, with caveats: (a) Rekor v2 (GA Oct 2025) drops `dsse` entries, so this design needs migration when v1 sunsets; (b) two TSAs isn't enough for diversity — pick **DigiCert + Certum + FreeTSA** (3 different jurisdictions/org types) per the TSA survey; (c) add OpenTimestamps as the long-term anchor (cert-free). |
| 10 | (Implicit) "RFC 3161 re-stamping is the long-term story" | **Incomplete** | Re-stamping via RFC 4998 ERS is not viable from Python (no production library); OpenSSL has no ERS support. Use OTS for long-term; treat RFC 3161 as short-horizon witness diversity. |

## 7. Recommended design

**Goal:** a stranger, with no trust in the owner, can answer R1–R3 (existence, no-backdate, no-silent-edit) for every published record; can answer R5 (non-equivocation) for every anchored record; cannot answer R4 (completeness) from cryptography alone — but the residual risk is bounded by pre-registration + independent re-measurement.

### 7.1 The record's outside anchor stack

```
record digest (sha256 of stored bytes — NOT a re-serialization)
 ├─ Sigstore attestation from protected CI workflow (Rekor v1 today; plan for v2 hashedrekord migration)
 │    └─ strict verifier: --cert-identity URL@REF --repository REPO --ref --trigger --offline
 ├─ RFC 3161 tokens from THREE free TSAs with different jurisdictions:
 │    ├─ DigiCert (commercial, US)
 │    ├─ Certum (commercial, EU/Poland)
 │    └─ FreeTSA (independent non-commercial, DE; donate yearly)
 ├─ OpenTimestamps proof (all 4 calendars; Bitcoin-anchored, cert-free)
 └─ [every chain head] deposit to a Sigsum log + witness cosigning via transparency-dev/witness
```

For each new chain head, also:

1. Commit a **pre-registered test plan** (test file, seed, hardware target, pass/fail criteria) **before** the run. Anchor it separately with the same stack.
2. Have CI run automatically (no `workflow_dispatch`-only trigger); protected branch; required status checks; admin bypass disabled.
3. **Snapshot to Software Heritage and mint a Zenodo DOI** periodically (e.g., monthly) as an independent history witness.

### 7.2 Verifier's procedure (a stranger)

1. **Existence:** pick any anchor (RFC 3161 token + `openssl ts -verify`; Sigstore bundle + sigstore-python; OTS proof + `ots verify`). All three should agree on the time T. If they disagree, treat the latest as the truth.
2. **No-backdate:** verified by the anchor itself (signed `gen_time` / Rekor `integratedTime` / Bitcoin block).
3. **No-silent-edit:** compare the record's stored-bytes digest to the anchored digest. For Sigstore, also verify the inclusion proof against Rekor's checkpoint.
4. **No-equivocation (R5):** query Rekor `POST /api/v1/index/retrieve` with `{"subject":"https://github.com/owner/repo/.github/workflows/<wf>.yml@refs/heads/<branch>"}`; list every entry; cross-check against the chain heads the owner claims; if Rekor returns entries the owner didn't publish, equivocation is detected. Supplement with SWH snapshots divergence.
5. **Completeness (R4):** not cryptographic. Compare anchored verdicts against the pre-registered test plan's schedule; a registered plan with no matching verdict sticks out. Re-run the pre-registered plan on independent hardware to verify the recorded verdict.

### 7.3 Remaining known holes

- **Withholding via a private run on a fork or off the protected workflow** is undetectable by any anchor. Bounded only by pre-registration + mandatory deposit on the protected workflow + an external monitor reconciling deposits against the schedule. A determined owner can still suppress a result; they cannot falsify one.
- **Rekor v1 → v2 migration:** v2 drops DSSE. Plan the migration now (hashedrekord loses the predicate, so the predicate goes elsewhere — into the record itself or into a separate hashedrekord entry keyed by the predicate's hash).
- **Sigstore public-good instance has no SLA beyond 99.5% uptime.** No commercial backstop. Long-term (10–20 yr) continuity rests on the Sigstore nonprofit + the wider community running mirrors/monitors.
- **OpenTimestamps calendar dependency at creation time:** the 4 calendars are independent operations; if all 4 disappear before a proof is upgraded, the proof stays pending forever. Mitigation: run `ots upgrade` from CI a few hours after each stamp; once upgraded, the proof is calendar-independent.
- **RFC 3161 cert expiry:** re-stamp every 2–3 years; archive old tokens even after expiry as historical evidence.
- **Branch protection admin bypass:** GitHub admins can still bypass. The owner is an admin. Mark `anchor-test` and `main` as protected with "Do not allow bypassing" set explicitly. Even so, the owner can change the ruleset or delete the repo — only the off-GitHub witnesses (Rekor, OTS, SWH, Sigsum) survive that.

### 7.4 Cheapest experiment to test the recommended design

The single cheapest test that would validate the most-load-bearing claim of this design — **"an identity monitor over Rekor catches equivocation"** — is:

1. In a throwaway public fork of exagium, run `actions/attest@v2` once on `refs/heads/anchor-test` (real run) and once on `refs/heads/secret-branch` (simulated equivocation), both with the same workflow file.
2. From an external machine, run:
   ```
   curl -s -X POST -H 'Content-Type: application/json' \
        -d '{"subject":"https://github.com/<owner>/<fork>/.github/workflows/anchor-test.yml@refs/heads/anchor-test"}' \
        https://rekor.sigstore.dev/api/v1/index/retrieve
   ```
   This should return the real run.
3. Also query with `subject=...@refs/heads/secret-branch`. If the attacker ran on `secret-branch`, that query returns the secret run.
4. Then check: does a single query (just `subject=...@refs/heads/anchor-test`) catch the `secret-branch` equivocation? **Answer: no, the subject string includes the ref.** To catch cross-ref equivocation, the monitor must query every possible ref, OR query by the workflow path alone — which Rekor's `subject` field doesn't support directly.
5. **Cheapest patch:** also run `rekor-monitor` (https://github.com/sigstore/rekor-monitor) configured to watch for the workflow's OIDC issuer (`https://token.actions.githubusercontent.com`) and the repository URI; rekor-monitor reconciles every entry against the expected schedule.

This experiment takes ~30 min, two GitHub Actions runs, ~6 live Rekor API calls, and 0 Bitcoin — well within budget. It would prove or disprove that an off-the-shelf monitor closes R5 for the Sigstore route today.

## 8. Files in this zip

```
report.md                                   this file
machine_and_tools.txt                       verbatim tool versions
scripts/
  repro_rfc3161.sh                          re-verify every RFC 3161 token
  repro_sigstore.py                         re-verify every Sigstore bundle offline
  repro_ots.py                              re-verify OTS proofs against Bitcoin
  repro_rekor.sh                            re-run live Rekor search-by-subject and by-digest
  longevity_rfc3161.sh                      re-verify every token with -attime past expiry
  jcs_demo.py                               compare Python sort_keys to RFC 8785 JCS
  attack_a_backdate.sh                      backdate attack demos (git dates + TSA -attime)
  attack_b_remove.sh                        remove/edit demos (drop, force-push, gist wipe)
  attack_c_equivocate.sh                    equivocate analysis (Rekor search-by-subject)
  attack_e_identity.sh                      identity-abuse tests (strict vs loose sigstore verify)
artifacts/
  rfc3161_repro.tsv                         all 64 token verify results
  rfc3161_longevity.tsv                     -attime past expiry results
  sigstore_repro.tsv                        all 6 Sigstore bundle verify results
  ots_repro.txt                             OTS verification report (Bitcoin side + proof tree)
  rekor_search_by_subject.txt               live Rekor search results (6 entries returned)
  attack_a_backdate.txt                     backdate attack output
  attack_b_remove.txt                       remove attack output
  attack_c_equivocate.txt                   equivocate analysis output
  attack_e_identity.txt                     identity abuse matrix
  jcs_demo.txt                              JCS vs json.dumps demo output
R4_R5_base_research.md                      full R4/R5 research brief (~2200 words)
```

## 9. What I could not test, and why

- **Live RFC 3161 re-stamp requests to each TSA** (to confirm the "no published rate limit" claim): I made 0 live stamp requests, conserving the 20-request budget and avoiding creating new anchors that aren't part of exagium's record. The TSA-survey subagent did make 8 live stamp probes (one per TSA); those are documented in the survey brief.
- **Creating a throwaway GitHub fork and running `actions/attest@v2` there to execute Attacks C and E live:** no GitHub credentials in this sandbox. The §3.5 attack matrix runs against the real exagium bundle with different verify-flag combinations, which is the locally-equivalent demonstration.
- **Deleting a GitHub attestation through the API** (Attack B.4): requires a PAT with `attestations: write` scope on `5kasgard/exagium`. Analytical only.
- **Full OTS Merkle-path verification (leaf → OP_RETURN):** the calendar server's 6-confirmation policy gates sub-proof release. I verified the Bitcoin side (block 974650 exists, contains the tx, with the 32-byte OP_RETURN) and the proof structure (4 leaves for `local/`, 3 for the CI proof), but the calendar-to-OP_RETURN Merkle path is ~5 Bitcoin blocks away. The trust-free alternative is a Bitcoin node, not in this sandbox.
- **`gh attestation verify` agreeing with sigstore-python:** `gh` is not installed; would need a GitHub PAT and an authenticated `gh` to test. sigstore-python 4.5.1 with `--offline` is the trust-minimising verifier and is what I used.

## 10. One-line summary

Reproduced: 6/8 RFC 3161 TSAs verify (DFN's root is missing from `roots.pem`; Apple was unreachable from sandbox); 6/6 Sigstore bundles verify offline strictly; OTS proofs are in Bitcoin block 974650 awaiting 6 confirmations. **Two of the author's claims are inverted** (Rekor searches by identity, not by digest; actual count is 6 not 4). R5 (non-equivocation) is closable via Sigsum + witness cosigning + SWH snapshots + Rekor identity monitoring. R4 (completeness) cannot come from any anchor; the defence is pre-registration + mandatory deposit + an external monitor reconciling deposits against the registered schedule, with independent re-measurement as the backstop. Recommended design: hash the stored bytes (not a re-serialization); anchor with Sigstore (strict verify) + 3 RFC 3161 TSAs in different jurisdictions + OTS (4 calendars); deposit each chain head to a Sigsum log; pre-register the test plan before each run; snapshot periodically to SWH and Zenodo. Cheapest validating experiment: run a fork-equivocation attack and confirm rekor-monitor catches it.
