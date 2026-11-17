# Note on weights.bin (added in the 2026-11-17 audit)

`weights.bin` here was trained on **the first 2,867 rows** of `loop/task/optdigits.tra` (`train75.csv`: the training file without the dev split), so the pilot's dev scores are out of sample.

`loop.py reproduce` retrains on the **whole** training file, so it will not reproduce these exact bytes. To reproduce them:

    head -n 2867 loop/task/optdigits.tra > train75.csv
    python3 -s -P train.py train75.csv weights.bin

The 75% recipe above was verified byte for byte for lgn_d only (2026-11-17 audit); for this directory it is untested. Full-file retrains were reproduced in the jail for lgn_b (pilot 1 README) and lgn_d (pilot 2 README).
