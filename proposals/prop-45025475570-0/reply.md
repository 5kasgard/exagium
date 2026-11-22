```hypothesis
Family: label-emb (embedding-based linear model). Each of the 14 input bytes gets a learned dense embedding (size E), replacing sparse one-hot encodings. This captures category similarities, reduces dimensionality, and should improve accuracy over label-linear (0.8524) while keeping inference fast (small E). Hyperparameter E is chosen on the tuning split (last 20% of train). If dev accuracy ≤ 0.855, the family does not justify its complexity.
```

```python
"""label-emb: embedding-based linear model for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Weights format (little-endian):
  magic(0x454D424C), d, E, C; float32 emb[d][256][E]; float32 W[d*E][C]; float32 b[C]."""
import json, math, struct, sys

import numpy as np

# hyperparameter candidates for embedding size E
E_CANDIDATES = [4, 8, 16]
# fixed seeds for each candidate (deterministic tuning)
CAND_SEEDS = [1, 2, 3]
# final training seed
FINAL_SEED = 999
# training hyperparameters (same as label-linear)
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x454D424C  # "EMBL" in little-endian


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_model(X, y, E, seed, epochs):
    """Train embeddings + linear layer on (X, y) for given epochs. Returns (emb, W, b)."""
    n, d = X.shape
    C = 2
    rng = np.random.default_rng(seed)
    # embeddings: (d, 256, E)
    emb = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
    # linear weights: (d*E, C)
    W = (rng.normal(0, 0.01, (d * E, C))).astype(np.float32)
    b = np.zeros(C, np.float32)

    params = [emb, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(X_batch):
        # X_batch: (m, d) uint8
        m = X_batch.shape[0]
        # gather embeddings: (m, d, E)
        e = emb[X_batch]  # fancy indexing works because emb is (d,256,E) and X_batch is (m,d)
        # flatten to (m, d*E)
        feat = e.reshape(m, -1)
        logits = feat @ W + b
        return logits, feat

    def backward(feat, g, m):
        # g: (m, C) gradient of loss wrt logits
        # grad for W: feat.T @ g + L2 * W
        # grad for b: g.sum(0)
        # grad for emb: need to scatter g @ W.T back to embedding indices
        grad_W = feat.T @ g + L2 * W
        grad_b = g.sum(axis=0)
        # grad_feat = g @ W.T  (m, d*E)
        grad_feat = (g @ W.T).astype(np.float32)
        grad_emb = np.zeros_like(emb)
        grad_feat_3d = grad_feat.reshape(m, d, E)
        # scatter add: for each sample i, feature j, value X[i,j], add grad_feat_3d[i,j]
        np.add.at(grad_emb, (np.arange(d)[:, None], X_batch.T), grad_feat_3d.transpose(1, 0, 2))
        grad_emb += L2 * emb
        return [grad_emb, grad_W, grad_b]

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            X_batch, y_batch = X[idx], y[idx]
            m = len(y_batch)
            logits, feat = forward(X_batch)
            # softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(m), y_batch] -= 1.0
            g = (probs / m).astype(np.float32)
            grads = backward(feat, g, m)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m_, v_ = state[i]
                m_ *= 0.9; m_ += 0.1 * gr
                v_ *= 0.999; v_ += 0.001 * gr * gr
                p -= LR * (m_ / (1 - 0.9 ** t)) / (np.sqrt(v_ / (1 - 0.999 ** t)) + 1e-8)
    return params


def evaluate(X, y, emb, W, b):
    logits, _ = forward(X, emb, W, b)
    pred = np.argmax(logits, axis=1)
    return (pred == y).mean()


def forward(X, emb, W, b):
    e = emb[X]
    feat = e.reshape(X.shape[0], -1)
    logits = feat @ W + b
    return logits, feat


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = X.shape[1]
    C = card["classes"]
    n = len(y)
    # tuning split: last 20% in file order
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    # compute epochs for tuning (based on training split size)
    tune_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))

    best_acc, best_E, best_params = -1, None, None
    for E, seed in zip(E_CANDIDATES, CAND_SEEDS):
        emb, W, b = train_model(X_train, y_train, E, seed, tune_epochs)
        acc = evaluate(X_tune, y_tune, emb, W, b)
        if acc > best_acc:
            best_acc, best_E, best_params = acc, E, (emb, W, b)

    # final training on all data
    final_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    emb, W, b = train_model(X, y, best_E, FINAL_SEED, final_epochs)

    # write weights
    out = struct.pack("<4i", MAGIC, d, best_E, C)
    out += emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class index.
 * Weights format: magic(0x454D424C), d, E, C; float32 emb[d][256][E]; float32 W[d*E][C]; float32 b[C]. */
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
    if (C != 2 || d != 14 || E <= 0 || E > 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * E + (long)d * E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * E;
    const float *b = W + (long)d * E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *feat = malloc(sizeof(float) * d * E);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // gather embeddings into feat
        for (int j = 0; j < d; j++) {
            const float *src = emb + ((long)j * 256 + xi[j]) * E;
            float *dst = feat + j * E;
            memcpy(dst, src, E * sizeof(float));
        }
        // linear layer
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int f = 0; f < d * E; f++) sum += feat[f] * W[(long)f * C + c];
            logits[c] = sum;
        }
        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```