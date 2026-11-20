```hypothesis
Family: label-emb-mlp. Learned embeddings for categorical features (9 columns) plus normalized ordinal features (5 columns) fed into a one-hidden-layer ReLU MLP. Embeddings capture richer category relationships than one-hot; the hidden layer models interactions. Precomputed per-byte lookup tables keep inference fast. If dev accuracy ≤ 0.8524 (label-linear) or cost per item exceeds label-linear, the hypothesis fails.
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
EMB_GRID = [4, 8]
H_GRID = [16, 32, 64]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n, d = X.shape
    C = card["classes"]
    types = card["input"]["types"]
    cat_idx = [j for j, t in enumerate(types) if t == "categorical"]
    ord_idx = [j for j, t in enumerate(types) if t == "ordinal"]
    rng = np.random.default_rng(SEED)

    # split: first 80% train, last 20% tune (file order)
    split = int(0.8 * n)
    idx_train = np.arange(split)
    idx_tune = np.arange(split, n)

    def preprocess(X_sub):
        mu, sd = {}, {}
        for j in ord_idx:
            v = X_sub[:, j].astype(np.float64)
            mu[j], sd[j] = v.mean(), v.std() or 1.0
        cat_maps = {}
        cat_sizes = {}
        for j in cat_idx:
            vals = np.unique(X_sub[:, j])
            mapping = np.full(256, len(vals), dtype=np.int32)  # unseen -> last index
            mapping[vals] = np.arange(len(vals))
            cat_maps[j] = mapping
            cat_sizes[j] = len(vals) + 1
        return mu, sd, cat_maps, cat_sizes

    def build_model(E, H, cat_sizes):
        W = H if H else C
        D = sum(E for _ in cat_idx) + len(ord_idx)
        params = []
        # embeddings
        embs = {}
        for j in cat_idx:
            K = cat_sizes[j]
            emb = (rng.uniform(-1, 1, (K, E)) * math.sqrt(6.0 / (K + E))).astype(np.float32)
            embs[j] = emb
            params.append(emb)
        # W1, b1
        W1 = (rng.uniform(-1, 1, (D, W)) * math.sqrt(6.0 / D)).astype(np.float32)
        b1 = np.zeros(W, np.float32)
        params.extend([W1, b1])
        if H:
            W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
            b2 = np.zeros(C, np.float32)
            params.extend([W2, b2])
        return params, embs, W1, b1, (W2, b2) if H else None

    def forward(Xb, mu, sd, cat_maps, embs, W1, b1, W2_b2, H):
        B = Xb.shape[0]
        Z_parts = []
        for j in ord_idx:
            z = (Xb[:, j].astype(np.float32) - mu[j]) / sd[j]
            Z_parts.append(z[:, None])
        for j in cat_idx:
            emb_idx = cat_maps[j][Xb[:, j]]
            Z_parts.append(embs[j][emb_idx])
        Z = np.concatenate(Z_parts, axis=1)
        a = Z @ W1 + b1
        if H:
            h = np.maximum(a, 0)
            W2, b2 = W2_b2
            logits = h @ W2 + b2
            return logits, Z, a, h
        else:
            return a, Z, a, None

    def loss_grad(logits, yb):
        B = logits.shape[0]
        max_logits = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(B), yb] -= 1.0
        return probs / B

    def accuracy(Xs, ys, mu, sd, cat_maps, embs, W1, b1, W2_b2, H):
        correct = 0
        for i in range(0, len(Xs), 1024):
            Xb = Xs[i:i+1024]
            yb = ys[i:i+1024]
            logits, _, _, _ = forward(Xb, mu, sd, cat_maps, embs, W1, b1, W2_b2, H)
            pred = logits.argmax(axis=1)
            correct += (pred == yb).sum()
        return correct / len(Xs)

    best_acc, best_cfg, best_state = -1, None, None
    for E in EMB_GRID:
        for H in H_GRID:
            mu, sd, cat_maps, cat_sizes = preprocess(X[idx_train])
            params, embs, W1, b1, W2_b2 = build_model(E, H, cat_sizes)
            state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
            t = 0
            n_train = len(idx_train)
            epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_train))
            for _ in range(epochs):
                order = rng.permutation(n_train)
                for s in range(0, n_train, BATCH):
                    idx = idx_train[order[s:s+BATCH]]
                    Xb, yb = X[idx], y[idx]
                    logits, Z, a, h = forward(Xb, mu, sd, cat_maps, embs, W1, b1, W2_b2, H)
                    g = loss_grad(logits, yb)
                    if H:
                        W2, b2 = W2_b2
                        dW2 = h.T @ g + L2 * W2
                        db2 = g.sum(0)
                        dh = g @ W2.T
                        da = dh * (a > 0)
                    else:
                        da = g
                    dW1 = Z.T @ da + L2 * W1
                    db1 = da.sum(0)
                    # embedding gradients
                    dZ = da @ W1.T
                    off = 0
                    grads = []
                    for j in ord_idx:
                        grads.append(None)  # no params
                        off += 1
                    for j in cat_idx:
                        dEmb = np.zeros_like(embs[j])
                        dZ_j = dZ[:, off:off+E]
                        emb_idx = cat_maps[j][Xb[:, j]]
                        np.add.at(dEmb, emb_idx, dZ_j)
                        grads.append(dEmb)
                        off += E
                    grads.extend([dW1, db1])
                    if H:
                        grads.extend([dW2, db2])
                    t += 1
                    for i, (p, gr) in enumerate(zip(params, grads)):
                        m, v = state[i]
                        m *= 0.9; m += 0.1 * gr
                        v *= 0.999; v += 0.001 * gr * gr
                        p -= LR * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
            acc = accuracy(X[idx_tune], y[idx_tune], mu, sd, cat_maps, embs, W1, b1, W2_b2, H)
            if acc > best_acc:
                best_acc, best_cfg = acc, (E, H)
                best_state = (mu, sd, cat_maps, cat_sizes, params, embs, W1, b1, W2_b2, H)

    # retrain on all data with best config
    E, H = best_cfg
    mu, sd, cat_maps, cat_sizes = preprocess(X)
    params, embs, W1, b1, W2_b2 = build_model(E, H, cat_sizes)
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X[idx], y[idx]
            logits, Z, a, h = forward(Xb, mu, sd, cat_maps, embs, W1, b1, W2_b2, H)
            g = loss_grad(logits, yb)
            if H:
                W2, b2 = W2_b2
                dW2 = h.T @ g + L2 * W2
                db2 = g.sum(0)
                dh = g @ W2.T
                da = dh * (a > 0)
            else:
                da = g
            dW1 = Z.T @ da + L2 * W1
            db1 = da.sum(0)
            dZ = da @ W1.T
            off = 0
            grads = []
            for j in ord_idx:
                grads.append(None)
                off += 1
            for j in cat_idx:
                dEmb = np.zeros_like(embs[j])
                dZ_j = dZ[:, off:off+E]
                emb_idx = cat_maps[j][Xb[:, j]]
                np.add.at(dEmb, emb_idx, dZ_j)
                grads.append(dEmb)
                off += E
            grads.extend([dW1, db1])
            if H:
                grads.extend([dW2, db2])
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)

    # build lookup tables T[j][v] for each feature
    W = H if H else C
    T = np.zeros((d, 256, W), dtype=np.float32)
    off = 0
    for j in range(d):
        if j in ord_idx:
            w = W1[off:off+1]  # (1, W)
            for v in range(256):
                z = (v - mu[j]) / sd[j]
                T[j, v] = z * w
            off += 1
        else:
            w = W1[off:off+E]  # (E, W)
            for v in range(256):
                emb_idx = cat_maps[j][v]
                T[j, v] = embs[j][emb_idx] @ w
            off += E

    # write weights: magic, d, H, C, T, b1, (W2, b2)
    magic = 0x454D4231  # "EMB1"
    out = struct.pack("<4i", magic, d, H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if H:
        W2, b2 = W2_b2
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
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
    if (nw < 16 || hd[0] != 0x454D4231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], W = H ? H : C;
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * W + W + (H ? (long)H * C + C : 0));
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *T = (const float *)(hd + 4);
    const float *b1 = T + (long)d * 256 * W;
    const float *W2 = b1 + W;
    const float *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W);
    float *sc = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int w = 0; w < W; w++) acc[w] = b1[w];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }
        const float *logits = acc;
        if (H) {
            for (int c = 0; c < C; c++) sc[c] = b2[c];
            for (int u = 0; u < H; u++) {
                float h = acc[u] > 0.0f ? acc[u] : 0.0f;
                for (int c = 0; c < C; c++) sc[c] += h * W2[(long)u * C + c];
            }
            logits = sc;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```