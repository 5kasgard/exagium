# Round 2 grading rule, fixed before any round-2 reply (2026-11-19 20:10Z)
Round 1 (runs 44602813904, 44602831482) is void: Claude's prompt said a direct commit "where a pull request was required" is not acceptable
without saying where one was required, and two graders failed every directly committed item for it; all three accepted R4, a card with no claims.
Changes, all to the instrument: each item states its required delivery; card items must name their best claim or NONE; item ids are neutral and
shuffled (seed 20261120); five control items with known answers are mixed in: {"C1": "ACCEPTABLE", "C2": "NOT", "C3": "NOT", "C4": "NOT", "C5": "ACCEPTABLE"}.
- Graders: cloudflare llama-4-scout, mistral-small-3.1, qwen3-30b-a3b; openrouter gemma-4-31b-it:free, ernie-5.5-flash-preview:free
  (none drafted, reviewed or checked the items). A grader that gets any control wrong, or misses a line, is excluded from the vote.
- An item is acceptable if a strict majority of the remaining graders say ACCEPTABLE. Fewer than 3 graders left: round inconclusive,
  and grading goes to a separate agent with the same prompts.
- Test 2's rule (worker/RELEASE-TEST-2.md, unchanged): pass on acceptance needs >= 25 of 30.
- Mapping (neutral id -> item): {"I01": "C2", "I02": "D8", "I03": "D13", "I04": "C4", "I05": "D9", "I06": "R4", "I07": "D3", "I08": "D18", "I09": "D21", "I10": "D16", "I11": "D17", "I12": "D20", "I13": "D1", "I14": "C5", "I15": "D2", "I16": "D22", "I17": "D6", "I18": "D11", "I19": "D12", "I20": "D15", "I21": "R8", "I22": "D10", "I23": "C1", "I24": "R6", "I25": "R5", "I26": "D4", "I27": "D14", "I28": "D5", "I29": "DB", "I30": "D7", "I31": "Q38", "I32": "D19", "I33": "R3", "I34": "C3", "I35": "R7"}
