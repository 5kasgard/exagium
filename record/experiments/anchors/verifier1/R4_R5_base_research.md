# R4 (completeness) and R5 (non-equivocation) for exagium — base research

Exagium's anchors (RFC 3161, OTS, Sigstore/Rekor, GitHub gist, the repo itself) all prove R1–R3: a record existed by a time, wasn't backdated, and wasn't silently edited. None proves **R4 — am I seeing every record, including unfavourable ones?** (completeness) or **R5 — does everyone see the same history?** (non-equivocation). This digs to the base.

---

## 1. Transparency-log fundamentals

A transparency log is an **append-only, Merkle-structured** sequence of entries. The machinery that turns "append-only on paper" into "append-only you can check" is, since RFC 6962 (Certificate Transparency, https://www.rfc-editor.org/rfc/rfc6962):

- **Merkle tree** over entries (§2.1). The root commits to the whole set; an inclusion (audit) path proves a leaf is in it (§2.1.1); a **consistency proof** proves one root is an append-only superset of another (§2.1.2).
- **Signed Tree Head (STH)** (§3.5): the log signs `{timestamp, tree_size, root_hash}`. Tree size + root are the public, checkable state of the log.
- **Monitors** (§5.3) fetch *every* new entry between consecutive STHs and rebuild the root; **auditors** (§5.4) check arbitrary pairs of STHs for consistency. §5: "All clients should gossip with each other, exchanging STHs at least; this is all that is required to ensure that they all have a consistent view."

This splits the two properties cleanly:

- **Completeness (R4)** is enforced by *monitors that know what should be in the log* and check it all appears. The log alone cannot prove a withheld entry is missing — that requires a second party with an expectation of presence.
- **Non-equivocation (R5)** is enforced by *gossip/witnessing of STHs*: if the log shows different roots to different parties (a **split-view**, two histories neither a prefix of the other), comparing STHs (or their cosignatures) detects it.

Chase & Meiklejohn, *"Transparency for and with Users"* (https://eprint.iacr.org/2019/1295), formalise this. They separate **transparency *for* users** (the subject of a logged entry can find entries about itself — detectability of misissue) from **transparency *with* users** (users carry the log state they saw and cross-check it with others — which is what yields **non-equivocation**: no log server can present conflicting views without some client noticing). Load-bearing for the rest of this brief: **non-equivocation is a property of the *distribution* of STHs, not of the log alone.** A log run by one operator, with no one watching, equivocates trivially.

The modern, generalised form of an STH is a **checkpoint** (https://c2sp.org/tlog-checkpoint): a signed note `<origin>\n<size>\n<root>\n` plus signatures. Witness cosigning is specified at https://c2sp.org/tlog-witness; the reference implementation is https://github.com/transparency-dev/witness.

---

## 2. Concrete candidates for exagium-style completeness

**Rekor identity monitoring.** Rekor's `searchIndex` endpoint (https://github.com/sigstore/rekor, `openapi.yaml`) accepts a `subject` field — a SAN such as `https://github.com/owner/repo/.github/workflows/build.yml@refs/heads/main` — plus `email`, `publicKey`, `hash`. So **yes, a stranger can list every Rekor entry minted under a given GitHub workflow identity**, provided `searchIndex` is enabled (it is on sigstore.dev) and entries were made via keyless OIDC. An external monitor watching that identity sees the full deposited set. **But this proves completeness only over runs that touched Rekor.** A run on a fork, a private runner, or with the workflow edited to skip the deposit produces no entry and is invisible. Rekor enforces completeness *conditional on every run being funnelled through it*; it cannot detect a run that never entered it.

**Sigsum** (https://www.sigsum.org/, getting-started). Log-only: one Merkle tree, one signed checkpoint per tree head, entries are `{hash, signature}` — exactly the shape of "chain head hash signed by exagium's key." Witness cosigning is first-class: the log reaches out to witnesses (https://witness-network.org) which counter-sign a checkpoint only if it is an append-only extension of what they last saw. Onboarding is a signing key + a `sigsum-submit` step; the public log is free. Exagium depositing each `runs/*.json` digest to a Sigsum log would give R1–R3 *plus* witness-cosigned checkpoints — i.e. R5 comes for free. Cost: trivial; onboarding: a day. Completeness, again, only over deposits actually made.

**CT gossip (RFC 6962 §5).** §5 prescribes STH gossip, but in deployed CT this collapsed to *monitors* (Google/Apple/Mozilla) plus browsers requiring SCTs with a Maximum Merge Delay; true P2P gossip among TLS clients never really shipped. The reusable artefact for exagium is the **pattern**, not the code: publish checkpoints, have independent witnesses cosign, and require verifiers to check cosignatures.

**transparency-dev/witness.** As the README states: a witness "verifies that tlogs are evolving in an append-only manner, and counter-signs checkpoints that represent an append-only evolution from any previously witnessed checkpoints." This **closes split-view equivocation (R5)** for any client that demands cosignatures: a log cannot fork without the witness refusing to cosign one branch. It **does not close completeness (R4)** — a log can simply decline to add an entry; the witness only attests "this tree grew monotonically," never "this tree contains everything that should exist." That is the load-bearing distinction for the rest of this brief.

---

## 3. Pre-registration

- **OSF** (https://help.osf.io/article/330-welcome-to-registrations): a registration is a "time-stamped, read-only version of your study plan … before beginning data collection," with a DOI, embargoable up to 4 years. A plan published now is frozen and dated, so a later verdict with no matching plan is conspicuously absent. It does **not** force anyone to run it.
- **AsPredicted.org** (https://aspredicted.org): a free-form form assigned an ID, private until the author shares it. Lightweight, but voluntary and post-hoc-publication is the author's choice.
- **ClinicalTrials.gov** is the enforcement model: registration **before** enrollment (42 USC 282(j)) and **results within 1 year of completion**, with FDA civil monetary penalties (FDAAA §801; statutory cap $10,000/day per violation, inflation-adjusted) for non-compliance. Notably, enforcement lagged for years; FDA began issuing pre-notices only ~2021. The lesson is that completeness needs both a *mandate* and an *enforcer with teeth*.
- **MLPerf** (https://github.com/mlcommons/policies, `submission_rules.adoc`): the review committee sees "All results, all code" confidentially during review, and there is a "Final opportunity to withdraw some or all results" — but withdrawing is itself visible to the committee. Fairness rests on the community norm "if you submit a system, you submit all runs," policed by competing submitters, not a cryptographic guarantee.

Does pre-registration close re-runs? **No.** If the owner runs the test 100 times privately and pre-registers only the run they liked, pre-registration is silent. It only closes the gap when combined with *mandatory automatic deposit on every run* (so a registered plan with a missing verdict sticks out) and *a monitor that reconciles deposits against the plan's schedule*.

---

## 4. CI as the only writer

With branch protection (https://docs.github.com/en/repositories/.../about-protected-branches): force-push and deletion are blocked on the protected branch by default, and required status checks/workflows can gate merges. But:

- Admins **bypass by default**; "Do not allow bypassing the above settings" must be set explicitly, and even then the admin who owns the repo can change the ruleset.
- The owner can **disable the workflow**, **edit the workflow file** to skip recording (admin write), **force-push non-protected branches**, or **delete and recreate the repo** (wipes the activity/audit log — the audit log is Enterprise-only for orgs anyway).
- Running the same workflow path in a **fork** yields a *different* Rekor OIDC identity (the SAN includes the repo), so a monitor on the original repo sees nothing. Withholding via fork is invisible.

So GitHub gives R5-ish guarantees *against non-admins* (a protected branch's history can't be silently rewritten by a third party), but the owner is an admin and controls the settings — GitHub gives no guarantee against the owner. The record's own caveat is honest: the counter "can't show that a record was made on the host it names, or when … that needs the record and its outside anchor." GitHub is part of the trust base, not a witness over it.

---

## 5. Independent archives as witnesses

- **Software Heritage** (https://www.softwareheritage.org/, Save Code Now): the git loader walks **all refs** of an origin and archives every reachable commit into a content-addressed Merkle DAG; a **snapshot** records `branch → revision`. So it preserves *full history and branches*, not just HEAD. Once loaded, content is permanent and immutable. A periodic "Save Code Now" on `github.com/5kasgard/exagium` yields a series of immutable snapshots at T₁, T₂… each freezing the repo tree (and thus every committed `runs/*.json` head) at that instant. A run committed then silently removed is absent from a later snapshot → detectable by diff. SWH is a strong **independent R5 witness** for committed history.
- **Zenodo** (https://help.zenodo.org, https://developers.zenodo.org): deposit → DOI minted on publish (reservable pre-publication), records immutable (new version → new linked DOI). CI can deposit record JSON via the REST API; the GitHub–Zenodo integration auto-archives releases as DOIs. So each chain head can get a citable, immutable, dated DOI.
- **Internet Archive Wayback** (https://web.archive.org, Save Page Now API): captures the *rendered HTTP view* of a URL at a time — good for "what the public could see," weak for git-object integrity.

Could periodic SWH + Zenodo deposits act as an independent completeness witness? They freeze **what was in the repo at time T** — closing R5 (no silent edit of committed history; a fork of history would diverge from the snapshot) — but they **cannot see a run that was never committed**, so they do not close R4 over withheld runs. They are witnesses over the published log, not over the universe of runs.

---

## 6. Bitcoin directly, versus via OpenTimestamps

An OP_RETURN output carrying a 32-byte hash is ~40 vB; a minimal 1-in/1-out+change tx is ~110 vB. Live fees today (block ~974,653, BTC ≈ $80,290): blockstream.info reports next-block ~1.07 sat/vB, low-priority ~0.10 sat/vB; blockchair's 24h median fee ≈ $0.014. So a direct OP_RETURN anchor costs **on the order of $0.01–$0.09** (≈11–110 sats) in the current, quiet market, rising with congestion. Direct commitment gives **the same guarantee as OTS** — OTS *is* an OP_RETURN commitment plus an attestation/receipt format; OTS just removes the need to run a wallet and can aggregate many stamps per Bitcoin tx. Direct is more transparent operationally; it is not stronger cryptographically. Neither addresses R4/R5: timestamping is R1–R3. (OTS stamps already exist in this repo's `workspace/`.)

---

## 7. The reframe: completeness can't come from an anchor

An anchor only ever sees what was **submitted to it**. Withholding — running privately and never depositing — is invisible to every anchor by construction. So R4 cannot be had from an anchor at all. The real defence is **independent re-measurement**: anyone re-runs the pre-registered test plan and gets the same verdict.

- **Cost.** From `runs/count-…7b1ad95d.json`, `python3 mlp.py 20` under `valgrind --tool=cachegrind` is ~2.9 s wall per rep (3 reps). Per `counter/README.md`, the program's own work (`./mlp 1000` minus startup) is **identical — 35,765,911 `Ir` — across six x86-64 CPU models**, and 30,898,727 on arm64. So a re-run is seconds of compute and reproduces bit-for-bit on the same architecture; the barrier is "any Linux box with valgrind ≥ 3.22," not money.
- **Bias channels.** The record fixes `PYTHONHASHSEED=0`, `MKL_NUM_THREADS=1`, etc. — these are recorded, so a verifier checks them. The deeper caveat (recorded): valgrind presents a fixed synthetic CPU (Intel `0x306c3`, AVX2-only), so **CPU dispatch is normalised away** — code that dispatches on features runs its AVX2-or-lower path under valgrind even on an AVX-512 host. That buys cross-host comparability, but it means the measured counts reflect the *AVX2 path*, not native performance: a content-level bias, independent of completeness.
- **What it closes.** Re-measurement closes **truth of content** (the recorded numbers are reproducible by a third party). It does **not** close R4: the owner could still withhold a bad run. But it *de-risks* withholding, because a claim that's reproducible can be regenerated by anyone — so withholding can only *deny the world a result*, not *falsify one*.

---

## What actually closes R4 and R5 for exagium

**R5 (non-equivocation) is closable:** publish each chain head as a **Sigsum checkpoint** (one Merkle tree, free, trivial onboarding), get it **witness-cosigned** by the transparency-dev witness network (https://github.com/transparency-dev/witness), and require verifiers to check cosignatures; supplement with **periodic SWH + Zenodo snapshots** as additional independent witnesses over committed history. A split-view (two chains, re-run-until-good) then becomes detectable, because witnesses refuse to cosign a fork and SWH snapshots diverge. **R4 (completeness) is not closable by any anchor**, because every anchor is blind to runs never submitted to it; completeness needs the *combination* of (a) a **pre-registered, dated test plan** (OSF/AsPredicted) that fixes what should run and on what schedule, (b) a **protected CI workflow that deposits every run** to the Sigsum/Rekor identity and cannot be silently bypassed on the protected branch, and (c) an **external monitor that reconciles deposits against the registered schedule** so a missing verdict sticks out. Even then a determined owner can run privately off the protected workflow and never deposit — but their results cannot be *claimed* either, because any published claim must carry a deposit and a reproducible plan, so withholding only suppresses a result rather than falsifying one; the residual risk that a withheld result is genuinely lost is then bounded by **independent re-measurement**, which regenerates any published verdict on cheap independent hardware. The honest summary: anchors and witnesses close R5 and the truth-of-content half of R4; the completeness half of R4 is closed not by cryptography but by a *social contract enforced in the open* — pre-register the plan, deposit every run, let strangers monitor and re-measure.
