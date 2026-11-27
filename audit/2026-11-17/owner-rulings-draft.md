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

# Rulings by data on the owner lane (2026-11-26, owner session)

Data for the whole group: the queue on 2026-11-26 holds 363 items (185 done, 88 parked, 77 open, 8 escalated, 3 review, 2 waiting on the owner: Q29 and P16). PR #213 (propose.yml dispatch retry, guard path) was guard-merged by the owner session 2026-11-26 after review: diff read line by line, unit CI green, stub-gh retry check in the PR body.

- P28 | go: run the pre-registered separation test exactly as the item's next field fixes it (rerun the 6 call-cap tasks once, cap 160 calls / 4200 s wall, same model, harness, seal and grader; resolved counts reported for every attempt, deciding nothing by itself) | the rule was written before the pilot (queue P28, "DIRECTION BY DATA"); the pilot's result is its trigger: 6 of 8 hit the 80-call cap, iceberg-python had a resolving diff when capped, the 2 kept ended at 33 and 48 calls; 6 x 160 = 960 calls fits one day's free allowance; Q82, the last blocker, is done | open, needs free | the rerun's fixed rule decides: at least 4 of 6 end by themselves -> H1 (cap becomes 160, proceed to the 36 attempts); at most 1 of 6 -> H2 (the 8 rerun with the next free model at cap 80); 2 or 3 -> both
- Q29 | confirmed as the one owner action, nothing else blocks it: Settings > Branches (or Rules), for main, router-runs and ci-runs, block force pushes and deletions; no required reviews or checks | re-verified 2026-11-26: all three branches report protected:false; the repo proxy 403s on the protection API path (write test 2026-11-25); the owner session's GitHub client has no branch-protection operation | waiting on the owner | the click
- P16 | hold unchanged; no data bears on the framing and nothing downstream starves | the hold is the owner's own 2026-11-19 instruction; blocker P2 is done and blocker P11 is open but unstarted by the owner's own 2026-11-19 instruction | waiting on the owner | the owner frames the scope when ready

# Owner-lane executions (2026-11-27, owner session)

- P28 | executed: the 160-cap separation run finished 2026-11-26 (3 of 6 ended by themselves -> the rule's middle case, both; 1 of 6 resolved; runs 45704843848, 45736434826; results/2026-11-26-cap160.md) and PR #219 was guard-merged as 50f2942 after the README conflict was cleared: the PR's README changes (Nov 26 diary row, step-4 sentence) moved to main commit b9d65d38, because git conflicts with #218's diary-row insertion and #248's adjacent step-3 change; content unchanged, reviewed as part of the PR. Next stands as the queue fixes it: the 8 usable tasks with ernie-5.5-flash at cap 80 (640 calls), dispatched by the machinery after the 00:00 UTC 2026-11-27 allowance reset (already past), now from main.
- K75 | executed: PR #208 merged as afd5fcd (worker/tier.py is the tier's own code, so a person merges). Diff matches the item spec line by line; the acceptance test ran on the merged tree in the owner session: python3 -m unittest worker.test_worker, 146 tests OK (CI never ran on the PR: github-actions[bot] PRs do not trigger workflows).
- Owner lane after this: Q29 (the settings click) and P16 (the hold) only. Note: p28/step4-cap160, an empty branch the owner session created at main's HEAD while preparing the merge (no unique commits), can go with the Q09/Q28 stale-branch cleanup.
