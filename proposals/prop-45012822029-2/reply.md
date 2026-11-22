```hypothesis
We propose a "label-emb-linear" family: each of the 16 context positions gets a learned embedding of dimension D (e.g., 4–32). The 16 embeddings are concatenated and fed to a linear layer that predicts the 65 classes. This factorizes the 16×256×65 one-hot weight matrix of label-linear into two smaller matrices (16×256×D and 16×D×65), reducing parameters and capturing position‑specific byte similarities. Tuning D on the last 20% of train.in should yield a model that beats label-linear (0.3375) on dev, or matches it with a smaller weights file and fewer CPU cycles. If dev accuracy does not exceed 0.3375 at similar or lower cost, the hypothesis fails.
```

```python
"""label-emb-linear: per-position byte embeddings + linear layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only.
Weights format: magic 0x454D424C ('LMBE'), d=16, D, C=65;
float32 embed[d][256][D], W[d][C][D], bias[C] (all little-endian)."""
import json, math, struct, sys

import numpy as np

# Hyperparameter search space (tuned on the last 20% of train.in)
D_CANDIDATES = [4, 8, 16, 32]
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS_TUNE = 10
MAX_EPOCHS_FULL = 50
SAMPLES_PER_EPOCH = 600000  # max samples per epoch (same as label-linear)

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def split_train_tune(X, y, tune_frac=0.2):
    n = X.shape[0]
    tune_size = int(n * tune_frac)
    # last 20% in file order
    return X[:-tune_size], y[:-tune_size], X[-tune_size:], y[-tune_size:]

def accuracy(scores, y):
    return np.mean(np.argmax(scores, axis=1) == y)

def train_model(X_tr, y_tr, X_te, y_te, D, rng, epochs, verbose=False):
    n, d = X_tr.shape
    C = 65
    # Initialize parameters
    embed = (rng.normal(0, 0.01, (d, 256, D))).astype(np.float32)
    W = (rng.normal(0, 0.01, (d, C, D))).astype(np.float32)
    bias = np.zeros(C, dtype=np.float32)

    # Adam state
    m_embed, v_embed = np.zeros_like(embed), np.zeros_like(embed)
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_bias, v_bias = np.zeros_like(bias), np.zeros_like(bias)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X_tr[idx]          # (B, d)
            yb = y_tr[idx]          # (B,)
            B = xb.shape[0]

            # Forward: gather embeddings for each position
            # embed_j shape (B, D) for each j
            embs = np.empty((d, B, D), dtype=np.float32)
            for j in range(d):
                embs[j] = embed[j, xb[:, j]]

            # Scores: sum_j embs[j] @ W[j].T  -> (B, C)
            scores = np.zeros((B, C), dtype=np.float32)
            for j in range(d):
                scores += embs[j] @ W[j].T
            scores += bias

            # Softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores, dtype=np.float32)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            dscores = probs / B  # (B, C)

            # Gradients
            grad_W = np.zeros_like(W)
            grad_embed = np.zeros_like(embed)
            grad_bias = dscores.sum(axis=0).astype(np.float32)

            for j in range(d):
                # dW[j] = dscores.T @ embs[j]  (C, D)
                grad_W[j] = dscores.T @ embs[j]
                # demb = dscores @ W[j]  (B, D)
                demb = dscores @ W[j]
                # Accumulate into grad_embed[j] at positions xb[:, j]
                np.add.at(grad_embed[j], xb[:, j], demb)

            # L2 regularization on W and embed (not bias)
            grad_W += L2 * W
            grad_embed += L2 * embed

            # Adam update
            t += 1
            for param, grad, m, v in [
                (W, grad_W, m_W, v_W),
                (embed, grad_embed, m_embed, v_embed),
                (bias, grad_bias, m_bias, v_bias)
            ]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + eps)

        # Optional validation
        if verbose and X_te is not None:
            # Evaluate on tune set
            embs_te = np.empty((d, X_te.shape[0], D), dtype=np.float32)
            for j in range(d):
                embs_te[j] = embed[j, X_te[:, j]]
            scores_te = np.zeros((X_te.shape[0], C), dtype=np.float32)
            for j in range(d):
                scores_te += embs_te[j] @ W[j].T
            scores_te += bias
            acc = accuracy(scores_te, y_te)
            if verbose:
                print(f"  D={D} epoch {epoch+1}/{epochs} tune acc={acc:.4f}", file=sys.stderr)

    return embed, W, bias

def evaluate(embed, W, bias, X, y):
    d = X.shape[1]
    C = 65
    embs = np.empty((d, X.shape[0], embed.shape[2]), dtype=np.float32)
    for j in range(d):
        embs[j] = embed[j, X[:, j]]
    scores = np.zeros((X.shape[0], C), dtype=np.float32)
    for j in range(d):
        scores += embs[j] @ W[j].T
    scores += bias
    return accuracy(scores, y)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    assert d == 16 and C == 65

    rng = np.random.default_rng(SEED)

    # Split for tuning (last 20% in file order)
    X_tr, y_tr, X_te, y_te = split_train_tune(X, y, 0.2)

    best_D = None
    best_acc = -1.0
    best_params = None

    for D in D_CANDIDATES:
        if X_te.shape[0] > 0:
            embed, W, bias = train_model(X_tr, y_tr, X_te, y_te, D, rng, MAX_EPOCHS_TUNE, verbose=True)
            acc = evaluate(embed, W, bias, X_te, y_te)
        else:
            embed, W, bias = train_model(X_tr, y_tr, None, None, D, rng, MAX_EPOCHS_TUNE, verbose=False)
            acc = evaluate(embed, W, bias, X_tr, y_tr)
        print(f"D={D} tune accuracy={acc:.4f}", file=sys.stderr)
        if acc > best_acc:
            best_acc = acc
            best_D = D
            best_params = (embed, W, bias)

    print(f"Selected D={best_D} with tune accuracy={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best D
    embed, W, bias = train_model(X, y, None, None, best_D, rng, MAX_EPOCHS_FULL, verbose=False)

    # Write weights.bin
    # Magic 'LMBE' = 0x454D424C little-endian
    magic = 0x454D424C
    out = struct.pack("<4i", magic, d, best_D, C)
    out += embed.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += bias.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-emb-linear inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index 0..64).
 * Weights format (little-endian):
 *   uint32 magic (0x454D424C 'LMBE')
 *   int32 d (16), int32 D, int32 C (65)
 *   float32 embed[d][256][D]
 *   float32 W[d][C][D]
 *   float32 bias[C] */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned char *slurp(const char *p, long *n) {
    FILE *f = fopen(p, "rb");
    if (!f) { perror(p); exit(3); }
    fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *b = malloc(*n ? *n : 1);
    if (fread(b, 1, *n, f) != (size_t)*n) { fprintf(stderr, "short read %s\n", p); exit(3); }
    fclose(f);
    return b;
}

static const unsigned char *next_rec(const unsigned char *buf, long n, long *pos, uint32_t *len) {
    if (*pos + 4 > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    uint32_t k = buf[*pos] | buf[*pos + 1] << 8 | buf[*pos + 2] << 16 | (uint32_t)buf[*pos + 3] << 24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k >> 8 & 255, k >> 16 & 255, k >> 24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 16 || C != 65 || D <= 0 || D > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L
        + (long)d * 256 * D * 4      // embed
        + (long)d * C * D * 4        // W
        + (long)C * 4;               // bias
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *embed = (const float *)(hd + 4);
    const float *W = embed + (long)d * 256 * D;
    const float *bias = W + (long)d * C * D;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Initialize scores with bias
        for (int c = 0; c < C; c++) scores[c] = bias[c];

        // For each position j
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            const float *e = embed + ((long)j * 256 + v) * D;
            const float *Wj = W + (long)j * C * D;
            // scores[c] += dot(Wj[c], e)
            for (int c = 0; c < C; c++) {
                float sum = 0.0f;
                const float *Wc = Wj + (long)c * D;
                for (int d_ = 0; d_ < D; d_++) sum += Wc[d_] * e[d_];
                scores[c] += sum;
            }
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```