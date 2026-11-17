# Rulings by data on the 23 "needs the owner" items (draft for review, 2026-11-18)

Rule from the owner (2026-11-18): owner decisions also follow the data. Claude stops only before a step that cannot be undone.
Each line: ruling | the data it rests on | new status | next step.

## A. Pre-bench gate (no new bench runs until these are done; none has any effect today)
Data for the whole group: all benches are complete (0 open cells), the crons are removed (main 05328a4), so no grade step or agent container runs. The defects are reproduced (S2), so they are real for any future bench.
- Q01 | fix before the next bench: build the diff without the agent's .git, or run git inside the no-network container | A C1 reproduced by S2 and the orchestrator | open, blocked by Q24 (no next bench until a task kind is chosen) | as before
- Q02 | fix before the next bench: results go to a per-run nonce file the agent process cannot predict | A M1, S2 confirmed | open, blocked by Q24 | as before
- Q03 | decided: every future bench uses the bench-4 design (a host-side meter holds the key) | bench-4 already runs that way; A M2, S2 confirmed | open, blocked by Q24 | write bench-4's meter into the next bench's plan
- Q05 | decided: pin the SDK, the agent-server image and uv by commit/digest in the next bench | A M4 | open, blocked by Q24
- Q17, Q23 | park: they are amendments to complete, registered benches; carry MODE validation and an idle timeout into the next bench design | B F6, E E-09; benches complete | parked
- (Q19 already parked with the same reason.)

## B. Can run now (decided by the data, reversible)
- Q04 | do it: hash-pinned lock (pip --require-hashes) for record.yml's two installs, tested first on record-e2e.yml | 37 packages resolved, 2 pinned, in a job holding OIDC + attestations + contents:write (A M3, S2); no plan freezes workflow files; a lock can be reverted in one commit | open | build the lock on ubuntu-24.04 x86_64 / Python 3.12, run record-e2e, then record
- Q06 | do the doc half now: state in README and record/README that `by` is free text and independence rests on who can push (only the owner today); binding to the GitHub actor waits until anyone else can push | A M5, severity lowered by S2 because only the owner can push | open (doc)
- Q38 | do it: replace "Results can't be lied about" with "Results can't be changed afterwards without it showing" | the ledger is tamper-evident (hash chain + Sigstore + OTS), not tamper-proof: a writer can still record false values (A M5) | open (doc)
- Q39 | do it: fix VERIFY.md facts (counter tests are 23, not 16; one numbering sequence); the word cap is raised to fit the ~21 items or the items are cut, whichever keeps every contract item | E E-13 (counted) | open (doc)
- Q40 | do the factual half: add router/ and the benches to the README status table; leave the word-history order alone (taste, no data either way) | E E-16, C F7 | open (doc)
- Q37 | do it: erratum file next to the registered bench-4 plan: arXiv 2607.22585 is a workshop preprint, and its direction differs from bench-4's | D citations (checked) | open (doc)
- Q28 + Q09 | do it in two steps: (1) on a throwaway repo, test that deleting a branch after tagging keeps its commits and that nothing reads the deleted ref; (2) tag each stale branch (anchor-test, q2-main, q2-shadow, e2e-main?) and delete it. CHECK: e2e-main is the ledger writer's test branch used by record-e2e.yml; keep it | C F3, F4; A m4: stale branches carry push-triggered privileged workflows, record-q2 lacks 5 security steps | open | step 2 is the one step to confirm with the owner only if step 1 shows something is lost
- Q29 | do the reversible half: turn on branch protection (no force-push, no deletion) for router-runs and ci-runs; move raw outputs to release assets later, only if the size starts to cost something | 94% of unique bytes, cited as evidence, unprotected (C F14, S2 93.7%); protection is a setting, undoable | open | needs an admin token: check what the session's gh token can do; if it cannot, this is the one owner action (a settings click)

## C. Decided for the next plan (frozen today, so park with a trigger)
- Q26 | decided: new candidates use one train.py + config; registered plans keep frozen copies | C F1, F16: 29% duplicated function lines on the pilot branch; frozen digests require copies for registered plans | parked until the next plan
- Q31 | decided: version the frozen tools separately (a tools tag per plan) | C F12 | parked until the next plan
- Q42, Q43 | decided: fix in the next plan's baselines (frozen by plan 2; the harness supplies N, so not exploitable today) | F findings 2, 3 | parked until the next plan
- Q30 | decided: wire probe/ into the per-image meter work (the standing next item), else delete it | C F8 | parked until the per-image meter work

## D. The 24/7 question
- Q24 | decided by the data: the free compute that is plentiful and measured is GitHub Actions CPU (loop, counter: 6-runner matrix, every run green); the free LLM supply is thin (4 pools answer; OpenRouter 50/day unless 10 credits were bought; Cloudflare 10k neurons/day fits reviews, not agent runs; 7 of 10 compat pools have no key). So the first work queue is candidate measurement on Actions (the per-image meter, then the MNIST plan), with LLM pools used for reviews (Cloudflare, NIM). Model routing waits for more supply. | E E-03, E-11, E-20, Q49 | open | turn the per-image meter item into the first queue job
- Q25 | still holds, for model routing only: no routing claim without a held-out task set; not needed for the Actions measurement queue | E E-04 | open, blocked by Q24 (unchanged)
- Q49 | done as input: folded into Q24's ruling | parked/done
- New owner-only action surfaced by the data (cannot be done by Claude): creating keys for Groq, Cerebras and Mistral is the cheapest way to add LLM capacity (E E-20). This is an account action, not a decision.
