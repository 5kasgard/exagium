```hypothesis
We propose a Factorization Machine family ("label-fm") that models pairwise feature interactions via a low-rank factorization. The adult dataset mixes ordinal and categorical features; FMs excel at capturing interactions in such sparse tabular data (e.g., click‑through rate prediction). With a small factor dimension (k=8), inference cost is O(d·k) per item, far below label-mlp16, while accuracy should exceed label-linear (0.8524). If dev accuracy does not beat label-linear or inference cost exceeds label-mlp16 (~2300 Ir/item), the hypothesis fails.
```

```python
import json, math, struct, sys
import numpy as np

# Fixed hyperparameters (except k which we tune on the tuning split)
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
SEED = 1
K_CHOICES = [4, 8, 16]  # factor dimensions to try

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def build_encoders(X, types):
    """Fit encoders on X (n x d). Returns list of encoder dicts per column, and total feature dimension D."""
    n, d = X.shape
    encoders = []
    D = 0
    for j in range(d):
        col = X[:, j]
        t = types[j]
        if t == "ordinal":
            mu = col.mean()
            sd = col.std()
            if sd == 0:
                sd = 1.0
            encoders.append({"type": "ordinal", "mu": mu, "sd": sd, "feat_idx": D})
            D += 1
        else:  # categorical
            vals = np.unique(col)
            mapping = {int(v): D + i for i, v in enumerate(vals)}
            encoders.append({"type": "categorical", "mapping": mapping, "feat_start": D, "num_feats": len(vals)})
            D += len(vals)
    return encoders, D

def apply_encoders(X, encoders, D):
    """Convert raw X (n x d) to feature matrix