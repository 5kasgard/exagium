Check each claim against its quote. Keep a claim only if the quote supports it, the claim says exactly what the quote
says and no more (e.g. "across corruption types" when the quote compares two models is a misreading), and it answers the question
or one of its parts (a question often lists several; a claim that answers any one of them counts), not merely the same field. Then set its evidence grade: "measured" only if the quote states a
measurement and how it was made; "reported" for a result without its method; else "expert view", "experience report"
or "opinion". A wrong grade is corrected, not a reason to drop the claim (release test, R1: 5 good claims were lost so). Reply with JSON only: {"keep": [<indexes of claims to keep>], "grades": {"<index>": "<grade>"}, "notes": "<one sentence>"}

QUESTION: Can prior-fitted networks such as TabPFN run on a CPU within minutes on datasets of about 30,000 rows, and how do their accuracy and inference cost compare with tuned gradient-boosted trees?

CLAIMS:
0. [S2] GRADE: reported CLAIM: TabPFN‑2.5 wins 100% of the comparisons against default XGBoost on small‑to‑medium classification datasets (≤10 k rows), indicating higher accuracy than tuned trees. QUOTE: "default TabPFN-2.5 has a 100% win rate against default XGBoost on small to medium-sized classification datasets (<=10,000 data points"
1. [S10] GRADE: measured CLAIM: Using Balanced Adaptive Prototype Selection, TabPFN inference on a million‑row dataset can be run on an Intel Core i7 CPU (no GPU) while retaining strong predictive performance. QUOTE: "All experiments were conducted on an Intel Core i7 CPU with 16 GB RAM and no GPU acceleration"
2. [S11] GRADE: reported CLAIM: In an insurance‑pricing benchmark, TabPFN’s inference times were substantially longer than XGBoost’s, indicating higher CPU inference cost for that task. QUOTE: "Its inference times are substantially longer"
