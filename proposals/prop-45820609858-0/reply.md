```hypothesis
Family: label-emb-linear. Each of the 14 byte columns receives a learned embedding vector of dimension E (tuned on the last 20% of train). Embeddings are summed across columns to form a dense E‑dimensional representation, followed by a linear classifier. This captures categorical similarities better than one‑hot linear while keeping inference extremely cheap (d*E adds + E*C multiply‑adds per item). Failure: dev accuracy ≤ 0.8524 (label-linear) or instructions/item not lower than label-linear.
```

```python
"""label-emb-linear: embedding + linear classifier for tabular byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Weights format (little-endian):
  int32 magic=0x454D424C, d, E, C
  float32 emb[d][256][E]
  float32 W_out[E][C]
  float32 b_out[C]"""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
E_CANDIDATES = [8, 16, 32]  # tuned on last 20% of train.in


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def evaluate(X, y, emb, W_out, b_out):
    n = X.shape[0]
    h = np.zeros((n, emb.shape[2]), dtype=np.float32)
    for j in range(X.shape[1]):
        h += emb[j, X[:, j]]
    scores = h @ W_out + b_out
    pred = scores.argmax(axis=1)
    return (pred == y).mean()


def train_model(X, y, X_val, y_val, E, steps, rng):
    d = X.shape[1]
    C = 2
    # init
    emb = rng.normal(0, 0.01, size=(d, 256, E)).astype(np.float32)
    W_out = rng.normal(0, math.sqrt(1.0 / E), size=(E, C)).astype(np.float32)
    b_out = np.zeros(C, np.float32)
    # Adam state
    m_emb, v_emb = np.zeros_like(emb), np.zeros_like(emb)
    m_W, v_W = np.zeros_like(W_out), np.zeros_like(W_out)
    m_b, v_b = np.zeros_like(b_out), np.zeros_like(b_out)
    t = 0
    n = X.shape[0]
    for _ in range(steps):
        idx = rng.permutation(n)[:BATCH]
        xb, yb = X[idx], y[idx]
        # forward
        h = np.zeros((BATCH, E), dtype=np.float32)
        for j in range(d):
            h += emb[j, xb[:, j]]
        scores = h @ W_out + b_out
        # softmax gradient
        scores -= scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        probs[np.arange(BATCH), yb] -= 1.0
        probs /= BATCH
        # gradients
        dW_out = h.T @ probs + L2 * W_out
        db_out = probs.sum(axis=0)
        dh = probs @ W_out.T
        demb = np.zeros_like(emb)
        for j in range(d):
            np.add.at(demb[j], xb[:, j], dh)
        demb += L2 * emb
        # Adam update
        t += 1
        beta1, beta2 = 0.9, 0.999
        for param, grad, m, v in [
            (emb, demb, m_emb, v_emb),
            (W_out, dW_out, m_W, v_W),
            (b_out, db_out, m_b, v_b),
        ]:
            m *= beta1; m += (1 - beta1) * grad
            v *= beta2; v += (1 - beta2) * grad * grad
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            param -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return emb, W_out, b_out


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    n = X.shape[0]
    val_size = n // 5
    X_train, y_train = X[:-val_size], y[:-val_size]
    X_val, y_val = X[-val_size:], y[-val_size:]

    rng = np.random.default_rng(SEED)
    steps_per_candidate = 2000  # quick tuning
    best_E, best_acc = E_CANDIDATES[0], -1.0
    for E in E_CANDIDATES:
        # use a derived seed for reproducibility across runs
        cand_rng = np.random.default_rng(SEED + E * 1000)
        emb, W_out, b_out = train_model(X_train, y_train, X_val, y_val, E, steps_per_candidate, cand_rng)
        acc = evaluate(X_val, y_val, emb, W_out, b_out)
        if acc > best_acc:
            best_acc, best_E = acc, E

    # final training on all data
    final_steps = min(MAX_EPOCHS, math.ceil(SAMPLES / n)) * (n // BATCH + 1)
    final_rng = np.random.default_rng(SEED + 9999)
    emb, W_out, b_out = train_model(X, y, None, None, best_E, final_steps, final_rng)

    # write weights
    magic = 0x454D424C
    out = struct.pack("<4i", magic, d, best_E, C)
    out += emb.astype("<f4").tobytes()
    out += W_out.astype("<f4").tobytes()
    out += b_out.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-emb-linear inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d from weights). Output: one byte class index. */
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
    if (nw < 16 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * E + (long)E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *emb = (const float *)(hd + 4);
    const float *W_out = emb + (long)d * 256 * E;
    const float *b_out = W_out + (long)E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = calloc(E, sizeof(float));
    float *scores = malloc(C * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(h, 0, E * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            for (int u = 0; u < E; u++) h[u] += e[u];
        }
        for (int c = 0; c < C; c++) {
            float s = b_out[c];
            for (int u = 0; u < E; u++) s += h[u] * W_out[(long)u * C + c];
            scores[c] = s;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```