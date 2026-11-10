# Outside audit of the "Two meters" section, 2026-11-11

**Auditor:** NVIDIA Nemotron 3 Ultra (`nvidia/nemotron-3-ultra-550b-a55b:free`, via OpenRouter), a non-Claude model with no other context. OpenRouter generation id `gen-1794354707-z8IpfFJw09UgDb15wvgj`. Briefed by Claude (the builder), with only the card's claims, the code and the raw numbers.

`prompt-as-built.txt` is the prompt as first assembled. The prompt actually sent was a trimmed version (some redundant pair rows dropped, the same numbers otherwise); `response.md` is the reply, verbatim.

**Result:** of 21 claims, 13 supported, 5 not checkable from the data given, 1 partly checkable, 1 overstated, and the proposed rule ambiguous. The builder checked each point against the data and fixed `loop/README.md`:
- "An MLP instruction took 1.9–2.5× as long": marked supported, but the auditor's own span was 1.9–2.6× (EPYC mlp32: 2.56×), so it now says that.
- "With startup removed mlp16 is 1.9–2.3× faster" → 1.8–2.6× on the six CI hosts.
- The rule for future plans didn't say how hosts combine or give the per-correct-answer formula. Both are now explicit.

The builder found one more while fixing these: "not the 4–5× its instruction count suggests" → 3.5–5.4×.

**Not covered:** the auditor saw summaries, not the JSON files; it couldn't check the code-reading claims or the ledger history.
