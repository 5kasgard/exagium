```hypothesis
Family: label-mlp-tuned (one-hidden-layer MLP with hidden size chosen on the tuning split).
Why: Adult has non-linear feature interactions; a small MLP captures them while staying fast. Tuning hidden size on the validation split adapts capacity without overfitting.
Failure: If dev accuracy ≤ label-linear (0.8524) or inference cost exceeds label-mlp16 (2325 Ir/item).
```

```python
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
CANDIDATE_H = [8, 16, 32, 64]


def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def encode_features(X, types, fit=True, encoders=None):
    n, d = X.shape
    if fit:
        encoders = []
    Z_parts = []
    for j, t in enumerate(types):
        v = X[:, j]
        if t == "ordinal":
            if fit:
                mu = v.mean()
                sd = v.std()
                if sd == 0:
                    sd = 1.0
                enc = (np.arange(256, dtype=np.float64) - mu) / sd
                encoders.append(enc.astype(np.float32)[:, None])
            else:
                enc = encoders[j]
            Z_parts.append(enc[v])
        else:
            if fit:
                vals = np.unique(v)
                E = np.zeros((256, len(vals)), dtype=np.float32)
                E[vals, np.arange(len(vals))] = 1.0
                encoders.append(E)
            else:
                E = encoders[j]
            Z_parts.append(E[v])
    Z = np.concatenate(Z_parts, axis=1).astype(np.float32)
    return Z, encoders


def train_mlp(Z, y, H, C, seed, epochs, lr, l2, batch):
    n, k = Z.shape
    rng = np.random.default_rng(seed)
    if H:
        W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [W1, b1, W2, b2]
    else:
        W1 = np.zeros((k, C), np.float32)
        b1 = np.zeros(C, np.float32)
        params = [W1, b1]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            z, yy = Z[idx], y[idx]
            if H:
                a = z @ W1 + b1
                h = np.maximum(a, 0)
                logits = h @ W2 + b2
            else:
                logits = z @ W1 + b1
            logits = logits - logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yy)), yy] -= 1.0
            g = (probs / len(yy)).astype(np.float32)
            if H:
                gh = g @ W2.T * (a > 0)
                grads = [
                    z.T @ gh + l2 * W1,
                    gh.sum(0),
                    h.T @ g + l2 * W2,
                    g.sum(0)
                ]
            else:
                grads = [
                    z.T @ g + l2 * W1,
                    g.sum(0)
                ]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9
                m += 0.1 * gr
                v *= 0.999
                v += 0.001 * gr * gr
                p -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return params


def evaluate(Z, y, params, H, C):
    W1, b1 = params[0], params[1]
    if H:
        W2, b2 = params[2], params[3]
        a = Z @ W1 + b1
        h = np.maximum(a, 0)
        logits = h @ W2 + b2
    else:
        logits = Z @ W1 + b1
    pred = logits.argmax(axis=1)
    return (pred == y).mean()


def pack_weights(card, encoders, params, H, C):
    d = card["input"]["bytes"]
    W = H if H else C
    T_list = []
    off = 0
    for E in encoders:
        W1 = params[0]
        T = E @ W1[off:off + E.shape[1], :W]
        T_list.append(T.astype(np.float32))
        off += E.shape[1]
    T_stack = np.stack(T_list)
    b1 = params[1].astype(np.float32)
    out = struct.pack("<4i", 0x54424C31, d, H, C)
    out += T_stack.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if H:
        out += params[2].astype("<f4").tobytes()
        out += params[3].astype("<f4").tobytes()
    return out


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    n = X.shape[0]
    n_train = int(0.8 * n)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    best_H = 0
    best_val_acc = -1.0
    best_encoders = None

    for H in CANDIDATE_H:
        Z_train, encoders = encode_features(X_train, types, fit=True)
        Z_val, _ = encode_features(X_val, types, fit=False, encoders=encoders)
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_train))
        params = train_mlp(Z_train, y_train, H, C, SEED, epochs, LR, L2, BATCH)
        val_acc = evaluate(Z_val, y_val, params, H, C)
        print(f"H={H}, val_acc={val_acc:.4f}", file=sys.stderr)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_H = H
            best_encoders = encoders

    print(f"Selected H={best_H} with val_acc={best_val_acc:.4f}", file=sys.stderr)

    Z_full, encoders_full = encode_features(X, types, fit=True)
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    final_params = train_mlp(Z_full, y, best_H, C, SEED, epochs_full, LR, L2, BATCH)
    out_bytes = pack_weights(card, encoders_full, final_params, best_H, C)
    open(sys.argv[4], "wb").write(out_bytes)


if __name__ == "__main__":
    main()
```

```c
/* Table family (linear, or one ReLU layer) for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins). Format: see train.py. */
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

/* next record of a record file: 4-byte little-endian length, then the bytes. */
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
    if (nw < 16 || hd[0] != 0x54424C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], W = H ? H : C;
    if (C < 1 || C > 256 || nw != 16 + 4L * ((long)d * 256 * W + W + (H ? (long)H * C + C : 0))) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *T = (const float *)(hd + 4), *b1 = T + (long)d * 256 * W, *W2 = b1 + W, *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W), *s = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int w = 0; w < W; w++) acc[w] = b1[w];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }
        const float *sc = acc;
        if (H) {
            for (int c = 0; c < C; c++) s[c] = b2[c];
            for (int u = 0; u < H; u++) {
                float h = acc[u] > 0.0f ? acc[u] : 0.0f;
                for (int c = 0; c < C; c++) s[c] += h * W2[(long)u * C + c];
            }
            sc = s;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```