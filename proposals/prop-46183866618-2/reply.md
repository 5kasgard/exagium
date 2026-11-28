```hypothesis
Family: label-embed (positional embeddings + linear classifier).
Why: The task predicts next character from 16 previous bytes. One-hot linear (label-linear) ignores similarities between characters; embeddings capture semantic similarity (e.g., letters vs punctuation) and reduce dimensionality, improving generalization. A small embedding dimension (8-32) keeps inference fast (dot products) and model size tiny.
Failure signal: If dev accuracy ≤ 0.3375 (label-linear) or inference cost exceeds label-linear by >2x, the family does not meet the goal.
```

```python
"""label-embed: positional embeddings + linear classifier for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Hyperparameter (embedding dim D) chosen on last 20% of train.in (tuning split),
then retrained on all train.in. Exports: magic, d, D, C, embed[d][256][D], W[C][d*D], b[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 1e-2
L2 = 1e-5
BATCH = 256
MAX_EPOCHS = 100
SAMPLES = 600000
D_CANDIDATES = [4, 8, 16, 32, 64]  # tried on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def split_tune(X, y, tune_frac=0.2):
    n = X.shape[0]
    tune_n = int(n * tune_frac)
    return X[:-tune_n], y[:-tune_n], X[-tune_n:], y[-tune_n:]


def train_model(X_tr, y_tr, X_te, y_te, d, C, D, rng):
    # embeddings: (d, 256, D)
    scale = math.sqrt(6.0 / (256 + D))
    emb = (rng.uniform(-1, 1, (d, 256, D)) * scale).astype(np.float32)
    # linear: (C, d*D)
    W = (rng.uniform(-1, 1, (C, d * D)) * math.sqrt(6.0 / (d * D))).astype(np.float32)
    b = np.zeros(C, np.float32)

    # Adam state
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    m_W, v_W = np.zeros_like(W), np.zeros_like(W)
    m_b, v_b = np.zeros_like(b), np.zeros_like(b)
    t = 0

    n = X_tr.shape[0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X_tr[idx], y_tr[idx]
            bs = xb.shape[0]

            # forward
            # embeddings: (bs, d, D)
            e = emb[np.arange(d), xb.T].transpose(1, 0, 2)  # (bs, d, D)
            flat = e.reshape(bs, d * D)  # (bs, d*D)
            scores = flat @ W.T + b  # (bs, C)

            # softmax + cross-entropy gradient
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            probs /= bs

            # gradients
            grad_W = probs.T @ flat + L2 * W  # (C, d*D)
            grad_b = probs.sum(axis=0)
            grad_flat = probs @ W  # (bs, d*D)
            grad_e = grad_flat.reshape(bs, d, D)  # (bs, d, D)

            # gradient w.r.t embeddings: scatter-add
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_e[:, j])

            # Adam update
            t += 1
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8

            for param, grad, m, v in [
                (emb, grad_emb, m_emb, v_emb),
                (W, grad_W, m_W, v_W),
                (b, grad_b, m_b, v_b),
            ]:
                m *= beta1
                m += (1 - beta1) * grad
                v *= beta2
                v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + eps)

    # evaluate on tune
    e_te = emb[np.arange(d), X_te.T].transpose(1, 0, 2)
    flat_te = e_te.reshape(X_te.shape[0], d * D)
    scores_te = flat_te @ W.T + b
    preds = scores_te.argmax(axis=1)
    acc = (preds == y_te).mean()
    return acc, emb, W, b


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]

    rng = np.random.default_rng(SEED)
    X_tr, y_tr, X_te, y_te = split_tune(X, y)

    best_D = D_CANDIDATES[0]
    best_acc = -1.0
    for D in D_CANDIDATES:
        acc, _, _, _ = train_model(X_tr, y_tr, X_te, y_te, d, C, D, rng)
        if acc > best_acc:
            best_acc = acc
            best_D = D

    # retrain on all data
    rng = np.random.default_rng(SEED)
    _, emb, W, b = train_model(X, y, X[:1], y[:1], d, C, best_D, rng)  # dummy tune

    # write weights
    # magic 0x454D4244 ("EMBD"), d, D, C
    header = struct.pack("<4i", 0x454D4244, d, best_D, C)
    # embeddings: (d, 256, D) row-major
    emb_bytes = emb.astype("<f4").tobytes()
    # W: (C, d*D)
    W_bytes = W.astype("<f4").tobytes()
    # b: (C,)
    b_bytes = b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + emb_bytes + W_bytes + b_bytes)


if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: one byte class index.
 * Weights format: magic(0x454D4244), d, D, C (int32 each), then
 *   float32 emb[d][256][D], W[C][d*D], b[C] (all little-endian). */
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
    if (nw < 16 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 16 || C < 1 || C > 256 || D < 1) { fprintf(stderr, "bad header values\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * D + (long)C * d * D + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * D;
    const float *b = W + (long)C * d * D;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = malloc(sizeof(float) * d * D);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            memcpy(flat + j * D, e, D * sizeof(float));
        }

        // linear layer
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            const float *w = W + (long)c * d * D;
            for (int k = 0; k < d * D; k++) sum += flat[k] * w[k];
            scores[c] = sum;
        }

        // argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```