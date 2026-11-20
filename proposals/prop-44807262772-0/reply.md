```hypothesis
Family: label-emb (embedding + shallow MLP). Ordinal features are z-scored; each categorical feature gets a learned embedding (dim E). Concatenated embeddings + ordinal scalars feed a one-hidden-layer ReLU MLP (width H) to 2-class softmax. Tuning (E, H, lr, L2) on the last 20% of train.in should beat label-linear's 0.8524 by capturing categorical interactions; if dev accuracy ≤ 0.8524 the hypothesis fails.
```
```python
"""label-emb: embedding + 1-hidden-layer MLP for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format (little-endian):
  magic(0x454D4244) d E H C
  for each feature j:
    if ordinal: mean(f32) std(f32)
    if categorical: 256*E f32 (embedding table)
  W1[D][H] f32, b1[H] f32, W2[H][C] f32, b2[C] f32
where D = #ordinal + #categorical*E.
"""
import json, math, struct, sys
import numpy as np

SEED = 1
SAMPLES = 600000
MAX_EPOCHS = 100
BATCH = 64
# Grid for tuning split (last 20% of train.in)
GRID = {
    'E': [2, 4, 8],
    'H': [16, 32, 64],
    'lr': [1e-3, 3e-3],
    'L2': [1e-4, 1e-3],
}

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    d = card["input"]["bytes"]
    n = X_bytes.shape[0]

    # Split: last 20% for tuning
    split = int(0.8 * n)
    idx = np.arange(n)
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    train_idx, tune_idx = idx[:split], idx[split:]

    # Ordinal stats on train split
    ord_stats = []
    for j, t in enumerate(types):
        if t == "ordinal":
            v = X_bytes[train_idx, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            ord_stats.append((mu, sd))
        else:
            ord_stats.append(None)

    best_acc, best_cfg, best_params = -1, None, None

    for E in GRID['E']:
        for H in GRID['H']:
            for lr in GRID['lr']:
                for L2 in GRID['L2']:
                    # Build model
                    cat_dims = [256 if t == "categorical" else 0 for t in types]
                    D = sum(1 for t in types if t == "ordinal") + sum(E for c in cat_dims if c)
                    rng_cfg = np.random.default_rng(SEED + hash((E, H, lr, L2)) % 2**32)
                    # Embeddings for categorical
                    emb = [rng_cfg.uniform(-1, 1, (256, E)).astype(np.float32) * math.sqrt(6.0 / (256 + E)) if c else None for c in cat_dims]
                    # MLP
                    W1 = (rng_cfg.uniform(-1, 1, (D, H)) * math.sqrt(6.0 / D)).astype(np.float32)
                    b1 = np.zeros(H, np.float32)
                    W2 = (rng_cfg.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
                    b2 = np.zeros(C, np.float32)
                    params = [W1, b1, W2, b2] + [e for e in emb if e is not None]
                    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

                    def forward(batch_X):
                        # batch_X: (B, d) uint8
                        B = batch_X.shape[0]
                        feats = np.zeros((B, D), np.float32)
                        col = 0
                        for j, t in enumerate(types):
                            if t == "ordinal":
                                mu, sd = ord_stats[j]
                                feats[:, col] = (batch_X[:, j].astype(np.float32) - mu) / sd
                                col += 1
                            else:
                                e = emb[cat_dims.index(256) if False else [i for i,c in enumerate(cat_dims) if c==256].index(j)]  # messy; fix below
                        # Re-implement cleanly:
                        return feats

                    # Re-write forward with proper embedding indexing
                    cat_indices = [i for i, c in enumerate(cat_dims) if c]
                    def forward(batch_X):
                        B = batch_X.shape[0]
                        feats = np.zeros((B, D), np.float32)
                        col = 0
                        for j, t in enumerate(types):
                            if t == "ordinal":
                                mu, sd = ord_stats[j]
                                feats[:, col] = (batch_X[:, j].astype(np.float32) - mu) / sd
                                col += 1
                            else:
                                # embedding lookup
                                e = emb[cat_indices.index(j)]
                                feats[:, col:col+E] = e[batch_X[:, j]]
                                col += E
                        a = feats @ W1 + b1
                        h = np.maximum(a, 0)
                        logits = h @ W2 + b2
                        return logits, feats, h

                    def softmax_cross_entropy_grad(logits, yy):
                        P = np.exp(logits - logits.max(axis=1, keepdims=True))
                        P /= P.sum(axis=1, keepdims=True)
                        P[np.arange(len(yy)), yy] -= 1.0
                        return (P / len(yy)).astype(np.float32)

                    t_step = 0
                    for epoch in range(min(MAX_EPOCHS, math.ceil(SAMPLES / split))):
                        order = rng.permutation(split)
                        for s in range(0, split, BATCH):
                            idx_b = train_idx[order[s:s+BATCH]]
                            xb, yb = X_bytes[idx_b], y[idx_b]
                            logits, feats, h = forward(xb)
                            g = softmax_cross_entropy_grad(logits, yb)
                            # Backprop
                            gW2 = h.T @ g + L2 * W2
                            gb2 = g.sum(0)
                            gh = g @ W2.T * (h > 0)
                            gW1 = feats.T @ gh + L2 * W1
                            gb1 = gh.sum(0)
                            # Embedding grads
                            g_emb = [np.zeros_like(e) for e in emb]
                            col = sum(1 for t in types if t == "ordinal")
                            for j in cat_indices:
                                e = emb[cat_indices.index(j)]
                                idxs = xb[:, j]
                                np.add.at(g_emb[cat_indices.index(j)], idxs, gh[:, col:col+E])
                                col += E
                            grads = [gW1, gb1, gW2, gb2] + g_emb
                            t_step += 1
                            for i, (p, gr) in enumerate(zip(params, grads)):
                                m, v = state[i]
                                m *= 0.9; m += 0.1 * gr
                                v *= 0.999; v += 0.001 * gr * gr
                                p -= lr * (m / (1 - 0.9 ** t_step)) / (np.sqrt(v / (1 - 0.999 ** t_step)) + 1e-8)

                    # Evaluate on tune split
                    correct = 0
                    for s in range(0, len(tune_idx), BATCH):
                        idx_b = tune_idx[s:s+BATCH]
                        xb, yb = X_bytes[idx_b], y[idx_b]
                        logits, _, _ = forward(xb)
                        pred = logits.argmax(axis=1)
                        correct += (pred == yb).sum()
                    acc = correct / len(tune_idx)
                    if acc > best_acc:
                        best_acc, best_cfg = acc, (E, H, lr, L2)
                        best_params = (params, emb, ord_stats)

    # Retrain on full train with best config
    E, H, lr, L2 = best_cfg
    cat_dims = [256 if t == "categorical" else 0 for t in types]
    D = sum(1 for t in types if t == "ordinal") + sum(E for c in cat_dims if c)
    rng = np.random.default_rng(SEED)
    emb = [rng.uniform(-1, 1, (256, E)).astype(np.float32) * math.sqrt(6.0 / (256 + E)) if c else None for c in cat_dims]
    W1 = (rng.uniform(-1, 1, (D, H)) * math.sqrt(6.0 / D)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [W1, b1, W2, b2] + [e for e in emb if e is not None]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    cat_indices = [i for i, c in enumerate(cat_dims) if c]
    def forward(batch_X):
        B = batch_X.shape[0]
        feats = np.zeros((B, D), np.float32)
        col = 0
        for j, t in enumerate(types):
            if t == "ordinal":
                mu, sd = ord_stats[j]
                feats[:, col] = (batch_X[:, j].astype(np.float32) - mu) / sd
                col += 1
            else:
                e = emb[cat_indices.index(j)]
                feats[:, col:col+E] = e[batch_X[:, j]]
                col += E
        a = feats @ W1 + b1
        h = np.maximum(a, 0)
        logits = h @ W2 + b2
        return logits, feats, h

    def softmax_cross_entropy_grad(logits, yy):
        P = np.exp(logits - logits.max(axis=1, keepdims=True))
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)

    t_step = 0
    for epoch in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx_b = order[s:s+BATCH]
            xb, yb = X_bytes[idx_b], y[idx_b]
            logits, feats, h = forward(xb)
            g = softmax_cross_entropy_grad(logits, yb)
            gW2 = h.T @ g + L2 * W2
            gb2 = g.sum(0)
            gh = g @ W2.T * (h > 0)
            gW1 = feats.T @ gh + L2 * W1
            gb1 = gh.sum(0)
            g_emb = [np.zeros_like(e) for e in emb]
            col = sum(1 for t in types if t == "ordinal")
            for j in cat_indices:
                idxs = xb[:, j]
                np.add.at(g_emb[cat_indices.index(j)], idxs, gh[:, col:col+E])
                col += E
            grads = [gW1, gb1, gW2, gb2] + g_emb
            t_step += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= lr * (m / (1 - 0.9 ** t_step)) / (np.sqrt(v / (1 - 0.999 ** t_step)) + 1e-8)

    # Pack weights
    # magic, d, E, H, C
    out = struct.pack("<5i", 0x454D4244, d, E, H, C)
    # Per feature
    for j, t in enumerate(types):
        if t == "ordinal":
            mu, sd = ord_stats[j]
            out += struct.pack("<2f", mu, sd)
        else:
            e = emb[cat_indices.index(j)]
            out += e.astype("<f4").tobytes()
    # MLP
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d from header). Output: 1 byte class index.
 * Weights layout (little-endian):
 *   int32 magic=0x454D4244, d, E, H, C
 *   for each feature j=0..d-1:
 *     if ordinal: float32 mean, std
 *     if categorical: 256*E float32 (embedding table)
 *   float32 W1[D][H], b1[H], W2[H][C], b2[C]
 *   where D = #ordinal + #categorical*E
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h,1,4,f); fwrite(r,1,k,f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    int32_t *hdr = (int32_t*)wb;
    if (nw < 20 || hdr[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hdr[1], E = hdr[2], H = hdr[3], C = hdr[4];
    if (C != 2 || d < 1 || E < 1 || H < 1) { fprintf(stderr, "bad header dims\n"); return 3; }

    // Count ordinal/categorical to compute D and expected weight size
    // We don't have types in weights; we must infer from weight layout.
    // But we know d=14 from card; we can hardcode the types order as per card.
    // Since the card is fixed for this task, we embed the type list.
    const char *types[] = {
        "ordinal","categorical","ordinal","categorical","ordinal",
        "categorical","categorical","categorical","categorical",
        "categorical","ordinal","ordinal","ordinal","categorical"
    };
    int n_ord = 0, n_cat = 0;
    for (int j=0;j<d;j++) if (strcmp(types[j],"ordinal")==0) n_ord++; else n_cat++;
    int D = n_ord + n_cat * E;
    long expected = 20 + (long)n_ord * 8 + (long)n_cat * 256 * E * 4 +
                    (long)D * H * 4 + H * 4 + (long)H * C * 4 + C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    // Pointers into weight buffer
    unsigned char *p = wb + 20;
    float *ord_mean = (float*)p; p += n_ord * 4;
    float *ord_std  = (float*)p; p += n_ord * 4;
    float *emb = (float*)p; p += n_cat * 256 * E * 4;
    float *W1 = (float*)p; p += (long)D * H * 4;
    float *b1 = (float*)p; p += H * 4;
    float *W2 = (float*)p; p += (long)H * C * 4;
    float *b2 = (float*)p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *feat = malloc(sizeof(float) * D);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Build feature vector
        int col = 0, oi = 0, ci = 0;
        for (int j = 0; j < d; j++) {
            if (strcmp(types[j], "ordinal") == 0) {
                feat[col++] = (xi[j] - ord_mean[oi]) / ord_std[oi];
                oi++;
            } else {
                const float *e = emb + (long)ci * 256 * E + xi[j] * E;
                for (int k = 0; k < E; k++) feat[col++] = e[k];
                ci++;
            }
        }

        // Hidden layer: ReLU(feat * W1 + b1)
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int d_ = 0; d_ < D; d_++) sum += feat[d_] * W1[(long)d_ * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }
        // Output layer
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            logits[c] = sum;
        }
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```