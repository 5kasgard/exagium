```hypothesis
Family: label-fm (Field-aware Factorization Machine). Why: Adult has 14 fields with rich pairwise interactions (e.g., education×occupation, marital-status×relationship) that linear models miss. A low-rank FM (K=8) adds only ~260 KB weights and ~1500 FMA/item, capturing all field×field interactions via shared per-value embeddings. This should exceed label-linear's 0.8524 dev accuracy at comparable latency. Failure signal: dev accuracy ≤ 0.8524 or Ir/item > 2000.
```

```python
"""label-fm: Field-aware Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-threaded, stdlib + numpy only.
Trains on 80%/20% split to pick K ∈ {4,8,12}, then refits on all data.
Exports: magic 0x464D3132, d, K, C; float32 linear[C][d][256], emb[C][d][256][K]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
K_CANDIDATES = (4, 8, 12)  # tuned on last 20% of train.in


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def softmax(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits, dtype=np.float32)
    return exp / exp.sum(axis=1, keepdims=True)


def train_fm(X, y, C, d, K, rng, n_epochs):
    """Returns (linear, emb) trained on (X, y). Shapes: linear[C,d,256], emb[C,d,256,K]."""
    n = X.shape[0]
    linear = rng.uniform(-0.01, 0.01, (C, d, 256)).astype(np.float32)
    emb = rng.uniform(-0.01, 0.01, (C, d, 256, K)).astype(np.float32)
    m_lin = np.zeros_like(linear)
    v_lin = np.zeros_like(linear)
    m_emb = np.zeros_like(emb)
    v_emb = np.zeros_like(emb)
    t = 0

    for _ in range(n_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d) uint8
            yb = y[idx]          # (B,) int64
            B = xb.shape[0]

            # Forward: compute scores for each class
            scores = np.zeros((B, C), np.float32)
            emb_cache = np.zeros((C, B, d, K), np.float32)  # for backward
            for c in range(C):
                # Linear part
                lin = np.zeros(B, np.float32)
                for j in range(d):
                    lin += linear[c, j, xb[:, j]]
                # FM part via Gram matrix
                for j in range(d):
                    emb_cache[c, :, j, :] = emb[c, j, xb[:, j], :]
                G = emb_cache[c] @ emb_cache[c].transpose(0, 2, 1)  # (B, d, d)
                fm = np.sum(np.triu(G, k=1), axis=(1, 2))
                scores[:, c] = lin + fm

            # Softmax + gradient
            probs = softmax(scores)
            probs[np.arange(B), yb] -= 1.0
            grad = probs / B  # (B, C)

            # Backward: accumulate gradients
            d_linear = np.zeros_like(linear)
            d_emb = np.zeros_like(emb)
            for c in range(C):
                gc = grad[:, c]                 # (B,)
                # Linear gradients
                for j in range(d):
                    np.add.at(d_linear[c, j], xb[:, j], gc)
                # Embedding gradients
                sum_emb = emb_cache[c].sum(axis=1)  # (B, K)
                for j in range(d):
                    vals = xb[:, j]
                    sum_except = sum_emb - emb_cache[c, :, j, :]  # (B, K)
                    gemb = gc[:, None] * sum_except              # (B, K)
                    np.add.at(d_emb[c, j], vals, gemb)

            # L2 regularization
            d_linear += L2 * linear
            d_emb += L2 * emb

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            # Linear
            m_lin = beta1 * m_lin + (1 - beta1) * d_linear
            v_lin = beta2 * v_lin + (1 - beta2) * d_linear * d_linear
            m_hat = m_lin / (1 - beta1 ** t)
            v_hat = v_lin / (1 - beta2 ** t)
            linear -= LR * m_hat / (np.sqrt(v_hat) + eps)
            # Embeddings
            m_emb = beta1 * m_emb + (1 - beta1) * d_emb
            v_emb = beta2 * v_emb + (1 - beta2) * d_emb * d_emb
            m_hat = m_emb / (1 - beta1 ** t)
            v_hat = v_emb / (1 - beta2 ** t)
            emb -= LR * m_hat / (np.sqrt(v_hat) + eps)

    return linear, emb


def evaluate(X, y, linear, emb, C, d, K):
    n = X.shape[0]
    correct = 0
    for i in range(n):
        xi = X[i]
        best_c, best_score = 0, -1e30
        for c in range(C):
            score = 0.0
            for j in range(d):
                score += linear[c, j, xi[j]]
            # FM
            vecs = [emb[c, j, xi[j], :] for j in range(d)]
            for j in range(d):
                vj = vecs[j]
                for k in range(j + 1, d):
                    score += np.dot(vj, vecs[k])
            if score > best_score:
                best_score, best_c = score, c
        if best_c == y[i]:
            correct += 1
    return correct / n


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    # Tuning split: last 20% in file order
    split = int(0.8 * n)
    X_tr, X_va = X[:split], X[split:]
    y_tr, y_va = y[:split], y[split:]

    rng = np.random.default_rng(SEED)
    n_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / max(1, split)))

    best_K, best_acc = 8, -1.0
    for K in K_CANDIDATES:
        lin, em = train_fm(X_tr, y_tr, C, d, K, np.random.default_rng(SEED), n_epochs)
        acc = evaluate(X_va, y_va, lin, em, C, d, K)
        if acc > best_acc:
            best_acc, best_K = acc, K

    # Retrain on all data with best K
    rng = np.random.default_rng(SEED)
    n_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    linear, emb = train_fm(X, y, C, d, best_K, rng, n_epochs)

    # Serialize: magic, d, K, C; linear[C,d,256]; emb[C,d,256,K]
    out = struct.pack("<4i", 0x464D3132, d, best_K, C)
    out += linear.astype("<f4").tobytes()
    out += emb.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Field-aware Factorization Machine with per-class embeddings.
 * Input record: d bytes (one per field). Output: 1 byte class index (argmax). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
    if (nw < 16 || hd[0] != 0x464D3132) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d != 14 || C != 2 || K <= 0 || K > 32) { fprintf(stderr, "bad header dims\n"); return 3; }

    size_t linear_elems = (size_t)C * d * 256;
    size_t emb_elems = (size_t)C * d * 256 * K;
    size_t expected = 16 + 4 * (linear_elems + emb_elems);
    if ((size_t)nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %zu\n", nw, expected); return 3; }

    const float *linear = (const float *)(hd + 4);
    const float *emb = linear + linear_elems;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float best_score = -1e30f;
        unsigned char best_c = 0;
        for (int c = 0; c < C; c++) {
            float score = 0.0f;
            const float *lin_c = linear + (size_t)c * d * 256;
            const float *emb_c = emb + (size_t)c * d * 256 * K;

            // Linear part
            for (int j = 0; j < d; j++)
                score += lin_c[(size_t)j * 256 + xi[j]];

            // FM part: sum_{j<k} dot(emb_c[j][xi[j]], emb_c[k][xi[k]])
            for (int j = 0; j < d; j++) {
                const float *vj = emb_c + ((size_t)j * 256 + xi[j]) * K;
                for (int k2 = j + 1; k2 < d; k2++) {
                    const float *vk = emb_c + ((size_t)k2 * 256 + xi[k2]) * K;
                    for (int f = 0; f < K; f++)
                        score += vj[f] * vk[f];
                }
            }

            if (score > best_score) {
                best_score = score;
                best_c = (unsigned char)c;
            }
        }
        put_rec(stdout, &best_c, 1);
    }
    return 0;
}
```