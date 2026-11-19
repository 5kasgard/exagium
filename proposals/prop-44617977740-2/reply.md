```hypothesis
The label-emb-mlp family uses per-position byte embeddings (size E) summed into a hidden vector, followed by ReLU and a linear output layer. This factorizes the input layer, reducing parameters and inference cost compared to label-mlp16 while adding non-linearity over label-linear. Tuning E on the validation split (last 20% of train) should yield higher accuracy than label-linear at similar or lower CPU cost. If dev accuracy does not exceed 0.34 or inference cost exceeds label-linear by >2x, the approach fails.
```

```python
"""label-emb-mlp: per-position embeddings + ReLU + linear output.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
E_CANDIDATES = [16, 32, 64, 128]  # embedding/hidden sizes to try on validation split

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_model(X, y, E, C, rng, steps):
    d = X.shape[1]  # 16
    # embeddings: list of d arrays (256, E)
    emb = [(rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / 256)).astype(np.float32) for _ in range(d)]
    b1 = np.zeros(E, np.float32)
    W2 = (rng.uniform(-1, 1, (E, C)) * math.sqrt(6.0 / E)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = emb + [b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = X.shape[0]
    for _ in range(steps):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X[idx], y[idx]
            # forward
            hidden = np.zeros((len(xb), E), np.float32)
            for j in range(d):
                hidden += emb[j][xb[:, j]]
            hidden += b1
            np.maximum(hidden, 0, out=hidden)  # ReLU
            logits = hidden @ W2 + b2
            # softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            probs /= len(yb)  # dlogits
            # backward
            dW2 = hidden.T @ probs + L2 * W2
            db2 = probs.sum(axis=0)
            dhidden = probs @ W2.T
            dhidden[hidden == 0] = 0  # ReLU gradient
            db1 = dhidden.sum(axis=0)
            demb = [np.zeros_like(e) for e in emb]
            for j in range(d):
                np.add.at(demb[j], xb[:, j], dhidden)
                demb[j] = demb[j] / len(yb) + L2 * emb[j]
            grads = demb + [db1, dW2, db2]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return emb, b1, W2, b2

def evaluate(X, y, emb, b1, W2, b2):
    d = X.shape[1]
    hidden = np.zeros((len(X), b1.shape[0]), np.float32)
    for j in range(d):
        hidden += emb[j][X[:, j]]
    hidden += b1
    np.maximum(hidden, 0, out=hidden)
    logits = hidden @ W2 + b2
    preds = logits.argmax(axis=1)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    rng = np.random.default_rng(SEED)
    n = X.shape[0]
    val_size = n // 5
    tr_X, tr_y = X[:-val_size], y[:-val_size]
    val_X, val_y = X[-val_size:], y[-val_size:]
    # choose E on validation split
    best_E, best_acc = E_CANDIDATES[0], -1.0
    steps = min(MAX_EPOCHS, math.ceil(SAMPLES / tr_X.shape[0]))
    for E in E_CANDIDATES:
        emb, b1, W2, b2 = train_model(tr_X, tr_y, E, C, rng, steps)
        acc = evaluate(val_X, val_y, emb, b1, W2, b2)
        if acc > best_acc:
            best_acc, best_E = acc, E
    # retrain on full data with best_E
    steps_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    emb, b1, W2, b2 = train_model(X, y, best_E, C, rng, steps_full)
    # write weights.bin
    # magic 0x454D424C ("LBME"), d, E, C
    out = struct.pack("<4i", 0x454D424C, d, best_E, C)
    for j in range(d):
        out += emb[j].astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: one byte class index. */
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
    if (d != 16 || C < 1 || C > 256) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * E + E + (long)E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }
    const float *emb = (const float *)(hd + 4);
    const float *b1 = emb + (long)d * 256 * E;
    const float *W2 = b1 + E;
    const float *b2 = W2 + (long)E * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = malloc(sizeof(float) * E);
    float *logits = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int e = 0; e < E; e++) hidden[e] = b1[e];
        for (int j = 0; j < d; j++) {
            const float *evec = emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) hidden[e] += evec[e];
        }
        for (int e = 0; e < E; e++) if (hidden[e] < 0.0f) hidden[e] = 0.0f;
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int e = 0; e < E; e++) sum += hidden[e] * W2[(long)e * C + c];
            logits[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```