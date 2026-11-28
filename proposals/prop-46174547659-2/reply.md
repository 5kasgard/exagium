```hypothesis
Family: label-emb-linear (embedding + linear). Each of the 16 context bytes is mapped to a learned dense vector (embedding size E), concatenated, then fed to a linear classifier. This reduces the 4096 one-hot dimensions to 16*E, allowing parameter sharing across byte values and positions, which should improve generalization on the small training set. The model is still extremely fast at inference (only 16*E + 16*E*C multiply-adds). We expect it to beat label-linear (0.3375) on dev accuracy. If it does not, it would show that one-hot linear is already optimal for this task or that the embedding size is insufficient.
```

```python
"""Embedding + linear family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Tunes embedding size on the last 20% of train.in (file order), then trains on all data.
Weights format: magic 0x454D424C, int32 d, E, C; float32 emb[d][256][E], W_out[d*E][C], b_out[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
CANDIDATE_E = [16, 32, 64, 128]  # embedding sizes to try on tuning split


def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X_bytes.shape[1]
    n = X_bytes.shape[0]

    # Tuning split: last 20% in file order
    split = int(0.8 * n)
    X_train, y_train = X_bytes[:split], y[:split]
    X_tune, y_tune = X_bytes[split:], y[split:]
    n_train = X_train.shape[0]

    rng = np.random.default_rng(SEED)

    best_E = None
    best_acc = -1.0
    best_emb = best_W_out = best_b_out = None

    for E in CANDIDATE_E:
        # Initialize embeddings: (d, 256, E)
        scale_emb = math.sqrt(6.0 / (256 + E))
        emb = rng.uniform(-scale_emb, scale_emb, (d, 256, E)).astype(np.float32)
        # Output layer: (d*E, C)
        scale_out = math.sqrt(6.0 / (d * E + C))
        W_out = rng.uniform(-scale_out, scale_out, (d * E, C)).astype(np.float32)
        b_out = np.zeros(C, np.float32)

        params = [emb, W_out, b_out]
        state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
        t = 0

        # Short training for tuning (5 epochs)
        for _ in range(5):
            order = rng.permutation(n_train)
            for s in range(0, n_train, BATCH):
                idx = order[s:s + BATCH]
                xb = X_train[idx]
                yy = y_train[idx]
                B = xb.shape[0]

                # Forward: concatenate embeddings
                h = np.zeros((B, d * E), dtype=np.float32)
                for j in range(d):
                    h[:, j * E:(j + 1) * E] = emb[j, xb[:, j], :]

                logits = h @ W_out + b_out
                logits -= logits.max(axis=1, keepdims=True)
                exp_logits = np.exp(logits)
                probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                probs[np.arange(B), yy] -= 1.0
                grad_logits = probs / B

                # Gradients
                grad_W_out = h.T @ grad_logits + L2 * W_out
                grad_b_out = grad_logits.sum(axis=0)
                grad_h = grad_logits @ W_out.T  # (B, d*E)

                grad_emb = np.zeros_like(emb)
                for i in range(B):
                    for j in range(d):
                        v = xb[i, j]
                        grad_emb[j, v, :] += grad_h[i, j * E:(j + 1) * E]
                grad_emb += L2 * emb

                grads = [grad_emb, grad_W_out, grad_b_out]
                t += 1
                for p, gr, (m, v) in zip(params, grads, state):
                    m *= 0.9; m += 0.1 * gr
                    v *= 0.999; v += 0.001 * gr * gr
                    m_hat = m / (1 - 0.9 ** t)
                    v_hat = v / (1 - 0.999 ** t)
                    p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

        # Evaluate on tuning split
        B = X_tune.shape[0]
        h_tune = np.zeros((B, d * E), dtype=np.float32)
        for j in range(d):
            h_tune[:, j * E:(j + 1) * E] = emb[j, X_tune[:, j], :]
        logits_tune = h_tune @ W_out + b_out
        preds = logits_tune.argmax(axis=1)
        acc = (preds == y_tune).mean()
        if acc > best_acc:
            best_acc = acc
            best_E = E
            best_emb = emb.copy()
            best_W_out = W_out.copy()
            best_b_out = b_out.copy()

    # Retrain on full data with best_E
    E = best_E
    emb, W_out, b_out = best_emb, best_W_out, best_b_out
    params = [emb, W_out, b_out]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n_full = n
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_full))

    for _ in range(epochs):
        order = rng.permutation(n_full)
        for s in range(0, n_full, BATCH):
            idx = order[s:s + BATCH]
            xb = X_bytes[idx]
            yy = y[idx]
            B = xb.shape[0]

            h = np.zeros((B, d * E), dtype=np.float32)
            for j in range(d):
                h[:, j * E:(j + 1) * E] = emb[j, xb[:, j], :]

            logits = h @ W_out + b_out
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yy] -= 1.0
            grad_logits = probs / B

            grad_W_out = h.T @ grad_logits + L2 * W_out
            grad_b_out = grad_logits.sum(axis=0)
            grad_h = grad_logits @ W_out.T

            grad_emb = np.zeros_like(emb)
            for i in range(B):
                for j in range(d):
                    v = xb[i, j]
                    grad_emb[j, v, :] += grad_h[i, j * E:(j + 1) * E]
            grad_emb += L2 * emb

            grads = [grad_emb, grad_W_out, grad_b_out]
            t += 1
            for p, gr, (m, v) in zip(params, grads, state):
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    magic = 0x454D424C  # "LMBE" in little-endian
    out = struct.pack("<4i", magic, d, E, C)
    out += emb.astype("<f4").tobytes()
    out += W_out.astype("<f4").tobytes()
    out += b_out.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Embedding + linear inference for label tasks: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output record: 1 byte (class index).
 * Weights format: magic 0x454D424C, int32 d, E, C; float32 emb[d][256][E], W_out[d*E][C], b_out[C]. */
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
    int d = hd[1], E = hd[2], C = hd[3];
    if (d != 16 || C < 1 || C > 256 || E < 1) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * E + (long)d * E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W_out = emb + (long)d * 256 * E;
    const float *b_out = W_out + (long)d * E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = (float *)malloc(sizeof(float) * d * E);
    float *scores = (float *)malloc(sizeof(float) * C);
    if (!h || !scores) { fprintf(stderr, "malloc failed\n"); return 3; }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memset(h, 0, sizeof(float) * d * E);
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            for (int k = 0; k < E; k++) h[j * E + k] += e[k];
        }

        for (int c = 0; c < C; c++) scores[c] = b_out[c];
        for (int k = 0; k < d * E; k++) {
            float hk = h[k];
            if (hk != 0.0f) {
                const float *w = W_out + (long)k * C;
                for (int c = 0; c < C; c++) scores[c] += hk * w[c];
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```