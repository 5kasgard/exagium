# Task data

**Optical Recognition of Handwritten Digits**, E. Alpaydin and C. Kaynak (1998), UCI Machine Learning Repository, https://doi.org/10.24432/C50P49. Licensed CC BY 4.0. Downloaded 2026-11-06 from https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip and unchanged.

| File | Rows | sha256 |
|---|---|---|
| `optdigits.tra` | 3,823 (30 writers): training | `e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd` |
| `optdigits.tes` | 1,797 (13 other writers): test | `6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8` |
| `optdigits.names` | the dataset's own description | `3e82f7202d72a2b7dbdbc324c8c90fe8853164f5d6ab978a071357ad3de89f02` |

Each row is 64 pixel counts (an 8×8 grid of 4×4 blocks of a 32×32 bitmap, so 0..16) and the digit. The test writers are disjoint from the training writers. No test row equals a training row.

**The test set is public, and widely copied:** scikit-learn's `load_digits()` is exactly `optdigits.tes` (checked row for row). A candidate that has seen it, through a library, a download or its own source, can't be told apart by its score. That is why `loop.py reproduce` retrains every candidate with no network, from the training file alone, and why a verifier reads `train.py`.
