# Reader probe on R4's bundle (2026-11-20, scheduled Claude shift), rules fixed before any call

Why: R4's second run (worker run 44642929224) read 28 sources, including the TabPFN-2.5 and TabPFN-3 abstracts, which
state win rates against tuned/default boosted trees up to 100K rows, and returned 0 claims (only open questions). The
question has two parts (CPU time at ~30,000 rows; accuracy and cost against tuned boosting); the abstracts answer the
second. Hypothesis: the reader prompt (after 7f417b8) drops every claim when one part of the question is unanswered.

Same model as the worker's reader (gpt-oss-120b; the worker uses Cloudflare, whose account allocation is spent today,
so OpenRouter's free endpoint), exact prompt the job builds from the saved search.json.
- current.txt: READ_PROMPT as on main. parts.txt: one added sentence ("If the question has several parts, extract
  claims for every part the sources do answer, and list only the parts they leave unanswered under open_questions.").
- 3 calls each. Scored by script: a reply counts if it has >= 1 claim whose quote occurs verbatim (whitespace
  normalised) in the cited source's abstract.

Rule:
1. current.txt counts in >= 2 of 3: the 0 was chance; R4 is redone unchanged.
2. Else, parts.txt counts in >= 2 of 3 and current.txt in <= 1: the added sentence goes into READ_PROMPT; R4 is redone.
3. Else: neither; R4 goes to the owner tier with the result (escalate).
