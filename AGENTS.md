# Working on exagium (for any agent or person)

## Before claiming anything
- Every number points to its evidence: a file, a run ID, or a ledger entry. If there is none, the claim is a hypothesis and is labelled as one.
- Say on which task, host and data split a result holds. Results on one task are not results about AI in general.
- A separate reviewer checks consequential work: another agent, or a free model through `review.yml`. Its points are verified, then applied or rejected with evidence.

## Use the cheapest worker that does the job well
See the README's "Who does what". Measurement and scoring run on CI. Reviews and proposals go to free models. Scarce paid models do design, integration and final checks.

## Closing every session
1. **Update the public face**, honestly:
   - the README's "Where it is now", "Progress", "Best result so far" and "Next";
   - any part's README whose behaviour changed.

   Remove claims the evidence no longer supports. Say what failed or was dropped, not only what worked. Keep the direction clear: what comes next and why.
2. **Update the queue** (`audit/2026-11-17/queue.json` on the pilot branch, until it moves to `main`): done items with where, new items with a next step.
   - Every new item says what it writes (`touches`: files, paths, branches).
   - `render.py` then lists any other item that writes the same place with no dependency between them. Put such items in one batch, or add a blocker, so related work lands together and nothing is edited twice.
3. **Write a handoff:** what was done, decided, tested or still hypothesis; the open list in order; anything left running; and spend.
