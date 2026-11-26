```hypothesis
Family: label-fm (Factorization Machine). Adult has 14 mixed-type features where pairwise interactions (e.g., education × occupation, capital-gain × hours) are known to matter. A 2nd-order FM learns per-field-per-value embeddings (k=8) and computes interactions in O(d·k) time, adding only ~130 KB weights and ~300 FLOPs/item over linear. It should beat label-linear (0.8524) on accuracy while staying near its inference cost. Failure: dev accuracy ≤ 0.853 or Ir/item > 1500.
```

```python
"""label-fm: Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single-thread, stdlib+numpy only.
Encodes each of the d=14 bytes as a field with 256 values.
Learns: bias w0, linear weights w[j][v], interaction embeddings V[j][v][f] (f=0..k-1).
Adam lr=1e-2, L2=1e-5, batch=256, epochs=min(50, ceil(200000/n)).
k=8 chosen on tuning split (last 20% of train.in).
Weights file: magic 0x464D3100, d, k, C; float32 w0, w[d][256], V[d][256][k]."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-5, 256, 200000, 50


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X_bytes.shape[0]

    # Tuning split: last 20%
    split = int(n * 0.8)
    X_tr, X_va = X_bytes[:split], X_bytes[split:]
    y_tr, y_va = y[:split], y[split:]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    w0 = np.float32(0.0)
    w = (rng.normal(0, 0.01, (d, 256)) * 0.01).astype(np.float32)
    V = (rng.normal(0, 0.01, (d, 256, K)) * 0.01).astype(np.float32)

    params = [w0, w, V]
    # Adam state
    m_w0, v_w0 = np.float32(0.0), np.float32(0.0)
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)

    def predict_scores(Xb):
        """Return raw scores for batch Xb (n_batch, d)"""
        n_batch = Xb.shape[0]
        scores = np.full(n_batch, w0, dtype=np.float32)
        # Linear part
        for j in range(d):
            scores += w[j, Xb[:, j]]
        # Interaction part: 0.5 * sum_f [(sum_j V[j,x_j,f])^2 - sum_j V[j,x_j,f]^2]
        sum_V = np.zeros((n_batch, K), dtype=np.float32)
        sum_V2 = np.zeros((n_batch, K), dtype=np.float32)
        for j in range(d):
            Vj = V[j, Xb[:, j], :]  # (n_batch, K)
            sum_V += Vj
            sum_V2 += Vj * Vj
        inter = 0.5 * np.sum(sum_V * sum_V - sum_V2, axis=1)
        scores += inter
        return scores

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def evaluate(Xb, yb):
        s = predict_scores(Xb)
        p = sigmoid(s)
        pred = (p > 0.5).astype(np.int64)
        return np.mean(pred == yb)

    best_acc = -1.0
    best_params = None
    t = 0

    for epoch in range(min(MAX_EPOCHS, math.ceil(SAMPLES / max(1, split)))):
        order = rng.permutation(split)
        for s in range(0, split, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_tr[idx], y_tr[idx]
            n_batch = len(yb)

            # Forward
            scores = np.full(n_batch, w0, dtype=np.float32)
            for j in range(d):
                scores += w[j, Xb[:, j]]
            sum_V = np.zeros((n_batch, K), dtype=np.float32)
            sum_V2 = np.zeros((n_batch, K), dtype=np.float32)
            for j in range(d):
                Vj = V[j, Xb[:, j], :]
                sum_V += Vj
                sum_V2 += Vj * Vj
            inter = 0.5 * np.sum(sum_V * sum_V - sum_V2, axis=1)
            scores += inter

            # Logistic loss gradient
            p = sigmoid(scores)
            grad = (p - yb).astype(np.float32)  # (n_batch,)

            # Backward
            # w0
            g_w0 = np.mean(grad)
            # w
            g_w = np.zeros_like(w)
            for j in range(d):
                for v in range(256):
                    mask = (Xb[:, j] == v)
                    if np.any(mask):
                        g_w[j, v] = np.mean(grad[mask]) + L2 * w[j, v]
            # V
            g_V = np.zeros_like(V)
            # d(inter)/dV[j,v,f] = grad_i * (sum_V[i,f] - V[j,v,f]) for each sample i with Xb[i,j]=v
            for j in range(d):
                for v in range(256):
                    mask = (Xb[:, j] == v)
                    if np.any(mask):
                        # sum_V[mask] is (m, K), V[j,v] is (K,)
                        diff = sum_V[mask] - V[j, v, :]  # (m, K)
                        # grad[mask] is (m,), broadcast to (m, K)
                        g_V[j, v, :] = np.mean(grad[mask, None] * diff, axis=0) + L2 * V[j, v, :]

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8

            # w0
            m_w0 = beta1 * m_w0 + (1 - beta1) * g_w0
            v_w0 = beta2 * v_w0 + (1 - beta2) * g_w0 * g_w0
            m_hat = m_w0 / (1 - beta1 ** t)
            v_hat = v_w0 / (1 - beta2 ** t)
            w0 -= LR * m_hat / (np.sqrt(v_hat) + eps)

            # w
            m_w = beta1 * m_w + (1 - beta1) * g_w
            v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
            m_hat = m_w / (1 - beta1 ** t)
            v_hat = v_w / (1 - beta2 ** t)
            w -= LR * m_hat / (np.sqrt(v_hat) + eps)

            # V
            m_V = beta1 * m_V + (1 - beta1) * g_V
            v_V = beta2 * v_V + (1 - beta2) * g_V * g_V
            m_hat = m_V / (1 - beta1 ** t)
            v_hat = v_V / (1 - beta2 ** t)
            V -= LR * m_hat / (np.sqrt(v_hat) + eps)

        # Validation check
        va_acc = evaluate(X_va, y_va)
        if va_acc > best_acc:
            best_acc = va_acc
            best_params = (w0.copy(), w.copy(), V.copy())

    # Retrain on all data with best init (or just use best_params)
    w0, w, V = best_params
    # One more pass on full data for final polish (few epochs)
    for epoch in range(3):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_bytes[idx], y[idx]
            n_batch = len(yb)

            scores = np.full(n_batch, w0, dtype=np.float32)
            for j in range(d):
                scores += w[j, Xb[:, j]]
            sum_V = np.zeros((n_batch, K), dtype=np.float32)
            sum_V2 = np.zeros((n_batch, K), dtype=np.float32)
            for j in range(d):
                Vj = V[j, Xb[:, j], :]
                sum_V += Vj
                sum_V2 += Vj * Vj
            inter = 0.5 * np.sum(sum_V * sum_V - sum_V2, axis=1)
            scores += inter

            p = sigmoid(scores)
            grad = (p - yb).astype(np.float32)

            g_w0 = np.mean(grad)
            g_w = np.zeros_like(w)
            for j in range(d):
                for v in range(256):
                    mask = (Xb[:, j] == v)
                    if np.any(mask):
                        g_w[j, v] = np.mean(grad[mask]) + L2 * w[j, v]
            g_V = np.zeros_like(V)
            for j in range(d):
                for v in range(256):
                    mask = (Xb[:, j] == v)
                    if np.any(mask):
                        diff = sum_V[mask] - V[j, v, :]
                        g_V[j, v, :] = np.mean(grad[mask, None] * diff, axis=0) + L2 * V[j, v, :]

            # Simple SGD for final polish (no Adam state)
            w0 -= LR * 0.1 * g_w0
            w -= LR * 0.1 * g_w
            V -= LR * 0.1 * g_V

    # Write weights
    # Magic: 0x464D3100 ('FM1\0')
    # d, k, C as int32
    # w0 as float32
    # w[d][256] as float32 (row-major)
    # V[d][256][k] as float32
    out = struct.pack("<4i", 0x464D3100, d, K, C)
    out += struct.pack("<f", w0)
    out += w.astype("<f4").tobytes()
    out += V.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Factorization Machine with k=8. Input: d bytes per record. Output: 1 byte class (0/1).
 * Weights format: magic 0x464D3100, int32 d, k, C; float32 w0; float32 w[d][256]; float32 V[d][256][k].
 * Score = w0 + sum_j w[j][x_j] + 0.5 * sum_f [(sum_j V[j][x_j][f])^2 - sum_j V[j][x_j][f]^2].
 * Class = 1 if score > 0 else 0. */
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
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wb;
    if (nw < 16 || hd[0] != 0x464D3100) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (d != 14 || K != 8 || C != 2) { fprintf(stderr, "unexpected dims\n"); return 3; }

    size_t expected = 16 + 4 + 4L * d * 256 + 4L * d * 256 * K;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size %ld != %zu\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 4);
    float w0 = p[0];
    const float *w = p + 1;
    const float *V = w + (long)d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_V = malloc(sizeof(float) * K);
    float *sum_V2 = malloc(sizeof(float) * K);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float score = w0;
        for (int f = 0; f < K; f++) { sum_V[f] = 0.0f; sum_V2[f] = 0.0f; }

        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            score += w[(long)j * 256 + v];
            const float *Vjv = V + ((long)j * 256 + v) * K;
            for (int f = 0; f < K; f++) {
                float val = Vjv[f];
                sum_V[f] += val;
                sum_V2[f] += val * val;
            }
        }

        float inter = 0.0f;
        for (int f = 0; f < K; f++) {
            inter += 0.5f * (sum_V[f] * sum_V[f] - sum_V2[f]);
        }
        score += inter;

        unsigned char pred = (score > 0.0f) ? 1 : 0;
        put_rec(stdout, &pred, 1);
    }
    return 0;
}
```