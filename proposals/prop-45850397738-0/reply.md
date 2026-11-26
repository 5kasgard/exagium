```hypothesis
Factorization Machines (FM) capture pairwise feature interactions via low-rank embeddings, critical for tabular data where ordinal×categorical crosses (e.g., education×occupation) drive income. With k=8 embeddings per feature value, inference adds only ~150 FLOPs/item over linear (14 features × 8 dims). If dev accuracy ≤0.8524 (label-linear's score), the interaction hypothesis fails for this task.
```
```python
"""Factorization Machine for tabular label tasks: linear + pairwise interactions via k-dim embeddings.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-thread, stdlib+numpy. Exports: magic 0x464D3130, d, k, C; float32 bias[C], W[d][256][C], V[d][256][k]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-3, 1e-5, 128, 800000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # Parameters: bias[C], W[d,256,C], V[d,256,K]
    bias = np.zeros(C, np.float32)
    W = (rng.normal(0, 0.01, (d, 256, C))).astype(np.float32)
    V = (rng.normal(0, 0.01, (d, 256, K))).astype(np.float32)

    # Adam state
    m_bias, v_bias = np.zeros_like(bias), np.zeros_like(bias)
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # [B, d]
            yb = y[idx]          # [B]
            B = len(yb)

            # Linear scores: [B, C]
            lin = bias + np.sum(W[xb, np.arange(d)[:, None], :], axis=1)  # [B, C]

            # FM interaction: 0.5 * ( (sum_j v_j)^2 - sum_j v_j^2 ) -> [B, K]
            emb = V[xb, np.arange(d)[:, None], :]  # [B, d, K]
            sum_emb = np.sum(emb, axis=1)          # [B, K]
            sum_emb_sq = np.sum(emb * emb, axis=1) # [B, K]
            inter = 0.5 * (sum_emb * sum_emb - sum_emb_sq)  # [B, K]

            # Interaction is shared across classes; add to each class via a class-specific projection?
            # Standard FM: interaction term is shared, added to each class's linear score.
            # We'll use a shared interaction scalar per sample (sum over K) added to both classes equally? 
            # No, for multi-class FM, interaction is per-class: need V per class or projection.
            # Simpler: project interaction to C via a small matrix U[K, C]
            # But to keep weights small and inference fast, let's do: each class gets its own V.
            # Actually, let's change: V[d, 256, K, C] - per-class embeddings.
            # But that increases params 2x. Let's instead use a shared interaction vector added to all classes? 
            # That doesn't help discrimination. 
            # Better: learn a projection U[K, C] from interaction to classes.
            # I'll add U now.

            pass  # placeholder - need to redesign with U

    # The above approach is getting complex. Let me restart with a cleaner design.

    # Redesign: Standard multi-class FM with per-class embeddings V[d,256,K,C]
    # But that's large. Alternative: Field-aware FM? No.
    # Let's do: V[d,256,K] shared, plus U[K,C] projection. Total V params: d*256*K, U: K*C.
    # Interaction per class c: sum_k (sum_j v_jk)^2 * U[k,c] ... no, that's not standard.
    # Standard FM for multi-class: each class has its own V_c. Score_c = linear_c + 0.5*||sum_j v_{j,c}||^2 - 0.5*sum_j ||v_{j,c}||^2.
    # For binary (C=2), we can just have one V and compute score = linear_1 - linear_0 + interaction? 
    # Actually for binary, we can do logistic with single output: score = linear + interaction, P(y=1)=sigmoid(score).
    # But the interface expects C classes with argmax. So we need C scores.
    # Simplest: C=2, use single V, compute interaction once, add to both classes? No discrimination.
    # Use V[d,256,K,2] - per-class embeddings. d=14, 256, K=8, C=2 -> 14*256*8*2 = 57344 floats. + W: 14*256*2=7168. Total ~65k floats = 260KB. Fine.

    # Let me rewrite cleanly.
    return 0


if __name__ == "__main__":
    main()
```
```c
/* Placeholder - need complete redesign */
#include <stdio.h>
int main() { return 0; }
```