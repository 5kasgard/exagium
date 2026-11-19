# Is the research checker consistent? (Q60 escalation of R10, R11, R13), fixed before any reply
Same model as the worker's checker (Nemotron 3 Ultra, here through OpenRouter's free tier, since review.yml has no NIM key),
same prompt the worker sent, 3 independent calls each on R10's 8 script-verified claims (the worker's checker kept 0) and on
R9's 3 (it kept 3). Measure: per claim, how many of 3 calls keep it; and agreement with the worker's original verdict.
Reading, fixed now: if the 3 calls disagree with each other on 2 or more claims of 11, the checker is a noisy gate (a
single call decides nothing) and the fix is a script gate plus a majority of calls; if they agree with each other and
with the original verdicts, the checker is consistent and the cause is in the reader's claims.
