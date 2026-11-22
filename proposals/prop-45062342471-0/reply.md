```hypothesis
Family: label-emb-mlp. Per-feature embeddings (all 14 columns treated as 256-category) fed into a small ReLU MLP (one hidden layer). This captures non-linear feature interactions missed by label-linear while staying far cheaper than GBDT/large MLP. If dev accuracy ≤ label-linear (0.8524) or inference cost > 2× label-linear, the hypothesis fails.
```
```python
import json, math, struct, sys

import numpy as np

# Hyperparameter grid (tuned on last 20% of train.in)
E_GRID = [4, 8, 16]
H_GRID = [8, 16, 32]
GRID_EPOCHS = 10
FINAL_EPOCHS = 50
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def split_train_tune(X, y):
    n = X.shape[0]
    tune_n = n // 5
    train_n = n - tune_n
    return X[:train_n], y[:train_n], X[train_n:], y[train_n:]


def train_model(X_tr, y_tr, X_va, y_va, d, E, H, C, epochs, seed):
    rng = np.random.default_rng(seed)
    # Embedding table: (d, 256, E)
    Emb = (rng.normal(0, 1, (d, 256, E)) / math.sqrt(E)).astype(np.float32)
    W1 = (rng.normal(0, 1, (d * E, H)) * math.sqrt(2.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, 1, (H, C)) * math.sqrt(2.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [Emb, W1, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    def forward(Xb):
        # Xb: (batch, d) uint8
        emb = Emb[Xb]  # (batch, d, E)
        flat = emb.reshape(len(Xb), d * E)
        h = flat @ W1 + b1
        h_relu = np.maximum(h, 0)
        logits = h_relu @ W2 + b2
        return flat, h, h_relu, logits

    def loss_grad(logits, yb):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(len(yb)), yb] -= 1.0
        return probs.astype(np.float32) / len(yb)

    t = 0
    n_tr = X_tr.shape[0]
    for _ in range(min(epochs, math.ceil(SAMPLES / n_tr))):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X_tr[idx], y_tr[idx]
            flat, h, h_relu, logits = forward(xb)
            g = loss_grad(logits, yb)  # (batch, C)

            # Backprop
            gW2 = h_relu.T @ g + L2 * W2
            gb2 = g.sum(0)
            gh_relu = g @ W2.T
            gh = gh_relu * (h > 0)
            gW1 = flat.T @ gh + L2 * W1
            gb1 = gh.sum(0)
            gflat = gh @ W1.T  # (batch, d*E)
            gEmb = np.zeros_like(Emb)
            gEmb_reshaped = gflat.reshape(len(xb), d, E)
            for j in range(d):
                np.add.at(gEmb[j], xb[:, j], gEmb_reshaped[:, j])
            gEmb += L2 * Emb

            grads = [gEmb, gW1, gb1, gW2, gb2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Validation accuracy
    flat, h, h_relu, logits = forward(X_va)
    pred = logits.argmax(1)
    acc = (pred == y_va).mean()
    return acc, params


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]

    X_tr, y_tr, X_va, y_va = split_train_tune(X, y)

    best_acc, best_params, best_E, best_H = -1, None, None, None
    for E in E_GRID:
        for H in H_GRID:
            acc, params = train_model(X_tr, y_tr, X_va, y_va, d, E, H, C, GRID_EPOCHS, SEED)
            if acc > best_acc:
                best_acc, best_params, best_E, best_H = acc, params, E, H

    # Retrain on full data
    final_params = train_model(X, y, X[:1], y[:1], d, best_E, best_H, C, FINAL_EPOCHS, SEED)[1]
    Emb, W1, b1, W2, b2 = final_params

    # Write weights: magic, d, E, H, C; then Emb[d,256,E], W1[d*E,H], b1[H], W2[H,C], b2[C]
    magic = 0x454D424C  # "EMBL"
    out = struct.pack("<5i", magic, d, best_E, best_H, C)
    out += Emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: 1 byte class index. */
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }
    long emb_sz = (long)d * 256 * E;
    long w1_sz = (long)d * E * H;
    long w2_sz = (long)H * C;
    long expected = 20 + 4L * (emb_sz + w1_sz + H + w2_sz + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *Emb = (const float *)(hd + 5);
    const float *W1 = Emb + emb_sz;
    const float *b1 = W1 + w1_sz;
    const float *W2 = b1 + H;
    const float *b2 = W2 + w2_sz;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* Concatenate embeddings */
        for (int j = 0; j < d; j++) {
            const float *src = Emb + ((long)j * 256 + xi[j]) * E;
            float *dst = flat + j * E;
            for (int e = 0; e < E; e++) dst[e] = src[e];
        }

        /* Hidden layer: ReLU(flat * W1 + b1) */
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int fe = 0; fe < d * E; fe++) sum += flat[fe] * W1[(long)fe * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        /* Output layer */
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```