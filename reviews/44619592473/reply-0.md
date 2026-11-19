<!-- cloudflare @cf/openai/gpt-oss-120b status=200 latency=20.75s -->

**KEEP** – the probe‑driven fact register directly tackles the concrete, recurring breakages with minimal added cost.  

It exists because **data called for it** (multiple recent run failures were traceable to drifting external facts).  

**Biggest flaw:** the rule’s success threshold (“≥ 2 changes caught before a break”) is arbitrary and can discard a useful safeguard that prevents even a single costly failure.  

**Cheapest fix:** change the criterion to “keep if the register prevents ≥ 1 break” (or any prevented break), eliminating the arbitrary numeric bar.
