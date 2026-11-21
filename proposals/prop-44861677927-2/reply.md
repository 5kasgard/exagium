```hypothesis
Family: label-embedding-mlp. A single-hidden-layer MLP with learned byte embeddings per position, trained with Adam. The hidden size H is tuned on the last 20% of train.in (candidates: 0, 8, 16, 32, 64). This captures positional n-gram interactions that linear models miss. If the best H is 0, it matches label-linear; if H>0 wins, we expect higher accuracy than label-linear (0.3375) with only O(H*d) extra inference cost.
```

```python
"""label-embedding-mlp: Table family with one ReLU hidden layer and tuned hidden size.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters (same as label-linear)
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
# Hidden sizes to try on the tuning split (0 = linear)
H_CANDIDATES = [0, 8, 16, 32, 64]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def build_encoders(X, types):
    """Return list of encoding matrices E_j (256, dim_j) for each position."""
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


def encode(X, enc):
    """Concatenate one-hot/ordinal encodings for all positions."""
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)


def init_params(k, H, C, rng):
    """Initialize parameters for given hidden size H."""
    if H:
        W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        return [W1, b1, W2, b2]
    else:
        W1 = np.zeros((k, C), np.float32)
        b1 = np.zeros(C, np.float32)
        return [W1, b1]


def softmax_grad(logits, y):
    """Return gradient of cross-entropy loss w.r.t logits (already averaged over batch)."""
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return (probs / len(y)).astype(np.float32)


def forward_linear(z, W1, b1):
    return z @ W1 + b1


def forward_mlp(z, W1, b1, W2, b2):
    a = z @ W1 + b1
    h = np.maximum(a, 0)
    return h @ W2 + b2, a, h


def train_on_split(Z, y, idx, H, C, rng_seed):
    """Train on given indices, return parameters."""
    n = len(idx)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    k = Z.shape[1]
    rng = np.random.default_rng(rng_seed)
    params = init_params(k, H, C, rng)
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            batch_idx = order[s:s + BATCH]
            z, yy = Z[batch_idx], y[batch_idx]
            if H:
                W1, b1, W2, b2 = params
                a = z @ W1 + b1
                h = np.maximum(a, 0)
                logits = h @ W2 + b2
                g = softmax_grad(logits, yy)
                gh = (g @ W2.T) * (a > 0)
                grads = [
                    z.T @ gh + L2 * W1,
                    gh.sum(0),
                    h.T @ g + L2 * W2,
                    g.sum(0)
                ]
            else:
                W1, b1 = params
                logits = z @ W1 + b1
                g = softmax_grad(logits, yy)
                grads = [
                    z.T @ g + L2 * W1,
                    g.sum(0)
                ]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return params


def eval_accuracy(Z, y, idx, params, H):
    """Compute accuracy on given indices."""
    correct = 0
    for i in idx:
        z = Z[i:i+1]
        if H:
            W1, b1, W2, b2 = params
            logits = np.maximum(z @ W1 + b1, 0) @ W2 + b2
        else:
            W1, b1 = params
            logits = z @ W1 + b1
        pred = logits.argmax(axis=1)[0]
        if pred == y[i]:
            correct += 1
    return correct / len(idx)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    types = card["input"]["types"]

    # Build encoders on full training data (no leakage: only vocabulary)
    enc = build_encoders(X, types)
    Z = encode(X, enc)

    # Split: last 20% for tuning
    n = len(X)
    split = int(0.8 * n)
    train_idx = np.arange(split)
    tune_idx = np.arange(split, n)

    best_H = 0
    best_acc = -1.0
    for H in H_CANDIDATES:
        # Use a deterministic seed per H
        params = train_on_split(Z, y, train_idx, H, C, SEED + H * 1000)
        acc = eval_accuracy(Z, y, tune_idx, params, H)
        if acc > best_acc:
            best_acc = acc
            best_H = H

    # Retrain on all data with best_H
    final_params = train_on_split(Z, y, np.arange(n), best_H, C, SEED)

    # Prepare lookup tables T[j][byte][w] as in label-linear
    off = 0
    W = best_H if best_H else C
    T = []
    for E in enc:
        if best_H:
            W1 = final_params[0]
            T.append((E.astype(np.float32) @ W1[off:off + E.shape[1]]).astype(np.float32))
        else:
            W1 = final_params[0]
            T.append((E.astype(np.float32) @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]

    # Write weights.bin
    b1 = final_params[1].astype("<f4")
    out = struct.pack("<4i", 0x54424C31, d, best_H, C)
    out += np.stack(T).astype("<f4").tobytes()
    out += b1.tobytes()
    if best_H:
        W2 = final_params[2].astype("<f4")
        b2 = final_params[3].astype("<f4")
        out += W2.tobytes()
        out += b2.tobytes()
    open(sys.argv[4], "wb").write(out)


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