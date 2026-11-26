```hypothesis
A two-hidden-layer MLP (32,16) with per-byte z-score encoding and ReLU activations, trained with Adam, will capture more non-linearities than label-linear and label-mlp16, achieving >97% accuracy on dev while keeping inference cost low (around 2x label-linear). If it fails to beat label-mlp16 (97.7%), it shows that two layers are not beneficial for this small dataset.
```
```python
"""Two-hidden-layer MLP table family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Encoding: per-byte z-score for ordinal features. Two ReLU layers (H1, H2) chosen on tuning split (last 20%).
Adam lr=3e-3, L2=1e-4, minibatch=64, epochs=min(100, ceil(600000/n)). Deterministic seed=1.
Exported as: magic 0x54424C32, d, H1, H2, C; float32 T1[d][256][H1], b1[H1], W2[H1][H2], b2[H2], W3[H2][C], b3[C]."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
# Candidate (H1, H2) pairs to try on tuning split
CANDIDATES = [(16, 8), (32, 16), (32, 8), (64, 32)]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def encode_features(X, types):
    """Return list of encoding matrices E_j (256 x dim_j) and total dim."""
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)))
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def build_table(enc, W1):
    """Build per-byte lookup table T1[d][256][H1] from encodings and first-layer weights."""
    d = len(enc)
    H1 = W1.shape[1]
    T1 = np.zeros((d, 256, H1), dtype=np.float32)
    off = 0
    for j, E in enumerate(enc):
        dim = E.shape[1]
        T1[j] = (E.astype(np.float32) @ W1[off:off + dim]).astype(np.float32)
        off += dim
    return T1


def train_model(Z, y, C, H1, H2, rng):
    """Train two-hidden-layer MLP on encoded features Z (n x k). Returns params."""
    n, k = Z.shape
    # Initialize weights
    W1 = (rng.uniform(-1, 1, (k, H1)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(H1, np.float32)
    W2 = (rng.uniform(-1, 1, (H1, H2)) * math.sqrt(6.0 / H1)).astype(np.float32)
    b2 = np.zeros(H2, np.float32)
    W3 = (rng.uniform(-1, 1, (H2, C)) * math.sqrt(6.0 / H2)).astype(np.float32)
    b3 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2, W3, b3]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z[idx], y[idx]
            # Forward
            a1 = z @ W1 + b1
            h1 = np.maximum(a1, 0)
            a2 = h1 @ W2 + b2
            h2 = np.maximum(a2, 0)
            logits = h2 @ W3 + b3
            g3 = softmax_grad(logits, yy)
            # Backward
            grad_W3 = h2.T @ g3 + L2 * W3
            grad_b3 = g3.sum(0)
            gh2 = (g3 @ W3.T) * (a2 > 0)
            grad_W2 = h1.T @ gh2 + L2 * W2
            grad_b2 = gh2.sum(0)
            gh1 = (gh2 @ W2.T) * (a1 > 0)
            grad_W1 = z.T @ gh1 + L2 * W1
            grad_b1 = gh1.sum(0)
            grads = [grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return params


def evaluate(T1, b1, W2, b2, W3, b3, X, y, enc):
    """Compute accuracy on given data."""
    n = X.shape[0]
    H1 = b1.shape[0]
    H2 = b2.shape[0]
    C = b3.shape[0]
    correct = 0
    for i in range(n):
        acc1 = b1.copy()
        for j in range(X.shape[1]):
            acc1 += T1[j, X[i, j]]
        h1 = np.maximum(acc1, 0)
        acc2 = b2 + h1 @ W2
        h2 = np.maximum(acc2, 0)
        logits = b3 + h2 @ W3
        pred = np.argmax(logits)
        if pred == y[i]:
            correct += 1
    return correct / n


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    n = X.shape[0]

    # Split: last 20% for tuning
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    # Encode training features
    enc = encode_features(X_train, types)
    Z_train = np.concatenate([E[X_train[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    Z_tune = np.concatenate([E[X_tune[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    best_H1, best_H2 = CANDIDATES[0]

    for H1, H2 in CANDIDATES:
        params = train_model(Z_train, y_train, C, H1, H2, rng)
        W1, b1, W2, b2, W3, b3 = params
        T1 = build_table(enc, W1)
        acc = evaluate(T1, b1, W2, b2, W3, b3, X_tune, y_tune, enc)
        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_H1, best_H2 = H1, H2

    # Retrain on full training set with best architecture
    enc_full = encode_features(X, types)
    Z_full = np.concatenate([E[X[:, j]] for j, E in enumerate(enc_full)], axis=1).astype(np.float32)
    rng = np.random.default_rng(SEED)  # reset seed for reproducibility
    final_params = train_model(Z_full, y, C, best_H1, best_H2, rng)
    W1, b1, W2, b2, W3, b3 = final_params
    T1 = build_table(enc_full, W1)

    # Pack weights
    d = X.shape[1]
    magic = 0x54424C32
    header = struct.pack("<5i", magic, d, best_H1, best_H2, C)
    out = header
    out += T1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    out += W3.astype("<f4").tobytes()
    out += b3.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Two-hidden-layer MLP table family for label tasks: ./infer weights.bin in.rec N
 * Input: d bytes per record. Output: one byte class index (argmax). */
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
    if (nw < 20 || hd[0] != 0x54424C32) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H1 = hd[2], H2 = hd[3], C = hd[4];
    if (C < 1 || C > 256) { fprintf(stderr, "bad class count\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * H1 + H1 + (long)H1 * H2 + H2 + (long)H2 * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *T1 = (const float *)(hd + 5);
    const float *b1 = T1 + (long)d * 256 * H1;
    const float *W2 = b1 + H1;
    const float *b2 = W2 + (long)H1 * H2;
    const float *W3 = b2 + H2;
    const float *b3 = W3 + (long)H2 * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc1 = malloc(sizeof(float) * H1);
    float *acc2 = malloc(sizeof(float) * H2);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memcpy(acc1, b1, H1 * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *t = T1 + ((long)j * 256 + xi[j]) * H1;
            for (int h = 0; h < H1; h++) acc1[h] += t[h];
        }
        for (int h = 0; h < H1; h++) if (acc1[h] < 0.0f) acc1[h] = 0.0f;

        memcpy(acc2, b2, H2 * sizeof(float));
        for (int h = 0; h < H1; h++) {
            float ah = acc1[h];
            if (ah == 0.0f) continue;
            const float *w = W2 + (long)h * H2;
            for (int u = 0; u < H2; u++) acc2[u] += ah * w[u];
        }
        for (int u = 0; u < H2; u++) if (acc2[u] < 0.0f) acc2[u] = 0.0f;

        memcpy(scores, b3, C * sizeof(float));
        for (int u = 0; u < H2; u++) {
            float au = acc2[u];
            if (au == 0.0f) continue;
            const float *w = W3 + (long)u * C;
            for (int c = 0; c < C; c++) scores[c] += au * w[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```