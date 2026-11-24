# Handoff: Claude chat 2026-11-24 ~11:00–11:50Z

Copied from the owner's project (claude/HANDOFF-chat-2026-11-24-queue-closeout.md) so other agents can read it.

## Done
- **Queue close-out** (via worker/queue_ops.py move, actor claude-chat, with evidence):
  - Closed as built: P5, P7, P8, P27, P30, P5b, Q58 and Q59.
  - Confirmed: R26 and R32.
  - Parked: 15 empty research escalations, for source gaps.
  - Wrote the 11:00Z shift's moves: R30, R62 and R43 escalated with recommendations; R36 redone.
- **Findings register:** audit/findings/ on this branch.
  - Covers five review series with prefixed IDs.
  - An independent audit corrected about 12% of the "fixed" labels and 5–10% of the "open" labels.
- **Guard rewrite (#117, merged):** the privilege guard now sees workflow env, reusable calls, local actions and .sh wrappers. Controls: old guard 1 of 4, new guard 4 of 4.
- **High findings → queue:**
  - Q82 seals the bench-6 grader (Z1-F003, Z1-F012). P28 waits on it.
  - Q83 is the bench.yml key exposure (Z1-F010).
- **Auto-merge widening: measured, not done.**
  - Keyed jobs run 51 of the 129 .py files.
  - The other 78 are almost all tests, graders, scoring or verdict code.
  - The lever is P33, the key broker.
- **New directions, blind review 45385736919 (gpt-oss-120b, Nemotron):**
  - P33 key broker: KEEP.
  - P34 Lean 4 domain: DEFER.
  - P35 connection graph: KEEP.
  - P36 chat-independent interface: kept, after P33.
- **API keys audit:**
  - 14 of 27 repository secrets are referenced by no workflow or code.
  - Workflows reference 6 secrets that do not exist (COHERE, CROWLLM x2, INFERBASE, MISTRAL, SAMBANOVA; all in router.yml's probe).
  - Semantic Scholar: the code reads S2_API_KEY, but the secret is SEMANTIC_SCHOLAR_API_KEY → PR #120 wires it to the search step.
- **Workflow quick wins (PR #121):** Z1-F009 keeper date, Z1-F011 bench-2 credentials, Z1-F033 guard suite always runs. Z1-F006 is left to #105.
- **Fault caused and fixed:** P36 was filed with empty `touches`, so render.py refused every queue write from 11:38Z to 11:44Z.

## Open, in order
1. **Owner:** merge or decide #120, #121, #104–#115, #92, #102, #96 and #94. Refresh the scheduled shift's token.
2. **The 13 unused keys:** each needs a free probe (authenticates? what is the free allowance?) and a role. Probe first, then route.
3. **P33 design step:** list the call sites and files per keyed job, and the hop count. Then one end-to-end run.
4. **Plan items toward distillation:** cut P11/P25 and P19 into typed items.
5. **P35:** the 10 dependency questions; Graphify vs P21 v0.
