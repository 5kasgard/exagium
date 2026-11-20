<!-- openrouter particle-ai/particle-1:free status=200 latency=21.91s -->

DEFER after a ranked failure-frequency table names the top bottleneck. Reason: five-mechanism capability scan is still machinery-first; test only the bottleneck that actually dominates current failures.

**Data-driven or a pleasing idea?** Pleasing idea: the 44/43/12/92 inventory is availability-driven, and the pass thresholds are set without baseline rates.

**Biggest flaw in the tests as written:** no control arm on the same fixed items, so passing small absolute thresholds (4/5 duplicates, 0/20 parse failures, top-half rerank) does not show the mechanism beats the current path. **Cheapest fix:** run the current pipeline on identical items and require a predeclared improvement; for log-probabilities, predeclare a numeric discrimination threshold.
