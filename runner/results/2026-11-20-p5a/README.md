# P5a result: does label-mlp2l's depth help beyond digits? (runner run 44626963486; rule fixed in queue P5a before results)

| task | mlp2l | vs mlp16 (paired) | depth helps (> 2 SE) | on the accuracy-cost front |
|---|---|---|---|---|
| digits (dev) | 0.9864, 35,422 Ir | +0.0094 ± 0.0040 | yes | yes (P5b on held-out data: ties gbdt, beats mlp16 by +0.0175 ± 0.0060) |
| UCI Adult | 0.8531, 11,368 Ir | −0.0060 ± 0.0031 | no | no: gbdt, mlp16 and linear dominate |
| next-character text | 0.2760, 24,710 Ir | −0.0500 ± 0.0062 | no | no: below linear (0.3375) at 3x its cost |

The proposal's gain is specific to digits, as P1 found for families in general: its depth helps on 8x8 images and hurts on
tabular and text data. Training reproduces byte for byte on all three. Forecasts (Claude, before): helps on text p 0.5
(wrong), on adult p 0.3 (it did not); Brier 0.17. `decide.py` here recomputes `decision.txt` from the stored JSON.
