```hypothesis
label-emb: Logistic regression with learned embeddings for categorical features (8 of 14 columns).
Categorical columns get dense embeddings (dim 4) instead of one-hot, capturing category similarities;
ordinal columns remain z-scored. Reduces dimensionality vs one-hot while modeling interactions via
embedding space. Should beat label-linear (0.8524) at similar inference cost (~800 Ir/item).
Failure: if categories have no shared structure, accuracy <= label-linear.
```
```python
"""label-emb: Logistic regression with embeddings for categorical features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Categorical columns: learned embeddings (dim E).
Ordinal columns: z-scored. Linear layer on concatenated features. Adam, L2.
Hyperparameter E chosen on tuning split (last 20%), then trained on all data."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 64, 600000, 100
CANDIDATE_E = [2, 4, 8, 16]  # embedding dimensions to try


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
    C = card["classes"]
    types = card["input"]["types"]
    d = len(types)

    # Identify categorical and ordinal columns
    cat_idx = [j for j, t in enumerate(types) if t == "categorical"]
    ord_idx = [j for j, t in enumerate(types) if t == "ordinal"]
    n_cat, n_ord = len(cat_idx), len(ord_idx)

    # Split tuning (last 20% in file order)
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_tu, y_tu = X[split:], y[split:]

    # Ordinal stats on full training data (for final model); on train split for tuning
    def compute_ord_stats(X_sub):
        means = X_sub[:, ord_idx].mean(axis=0).astype(np.float32)
        stds = X_sub[:, ord_idx].std(axis=0).astype(np.float32)
        stds[stds == 0] = 1.0
        return means, stds

    # For each candidate E, train on train split, evaluate on tuning
    best_E, best_acc = CANDIDATE_E[0], -1.0
    rng = np.random.default_rng(SEED)

    for E in CANDIDATE_E:
        # Initialize embeddings: [n_cat, 256, E]
        emb = (rng.uniform(-0.1, 0.1, (n_cat, 256, E)) * math.sqrt(6.0 / (256 + E))).astype(np.float32)
        # Linear weights: [n_ord + n_cat*E, C]
        total_dim = n_ord + n_cat * E
        W = (rng.uniform(-0.1, 0.1, (total_dim, C)) * math.sqrt(6.0 / total_dim)).astype(np.float32)
        b = np.zeros(C, np.float32)

        # Ordinal stats on train split
        ord_means, ord_stds = compute_ord_stats(X_tr)

        # Adam state
        params = [emb, W, b]
        state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
        t = 0

        n_tr = X_tr.shape[0]
        epochs = min(MAX_EPOCHS, max(1, math.ceil(SAMPLES / n_tr)))

        for _ in range(epochs):
            order = rng.permutation(n_tr)
            for s in range(0, n_tr, BATCH):
                idx = order[s:s + BATCH]
                xb, yb = X_tr[idx], y_tr[idx]
                bs = len(yb)

                # Forward
                # Ordinal part
                z_ord = ((xb[:, ord_idx].astype(np.float32) - ord_means) / ord_stds)  # [bs, n_ord]
                # Categorical embeddings
                z_cat = np.zeros((bs, n_cat * E), np.float32)
                for k, col in enumerate(cat_idx):
                    z_cat[:, k*E:(k+1)*E] = emb[k, xb[:, col]]
                # Concatenate
                z = np.concatenate([z_ord, z_cat], axis=1)  # [bs, total_dim]
                # Linear
                logits = z @ W + b  # [bs, C]
                # Softmax + cross-entropy gradient
                logits -= logits.max(axis=1, keepdims=True)
                exp_logits = np.exp(logits)
                probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                probs[np.arange(bs), yb] -= 1.0
                grad_out = probs / bs  # [bs, C]

                # Gradients
                grad_W = z.T @ grad_out + L2 * W
                grad_b = grad_out.sum(axis=0)
                grad_z = grad_out @ W.T  # [bs, total_dim]

                # Split grad_z
                grad_z_ord = grad_z[:, :n_ord]
                grad_z_cat = grad_z[:, n_ord:]

                # Embedding gradients
                grad_emb = np.zeros_like(emb)
                for k, col in enumerate(cat_idx):
                    idxs = xb[:, col]
                    np.add.at(grad_emb[k], idxs, grad_z_cat[:, k*E:(k+1)*E])

                grads = [grad_emb, grad_W, grad_b]

                # Adam update
                t += 1
                for i, (p, gr) in enumerate(zip(params, grads)):
                    m, v = state[i]
                    m *= 0.9; m += 0.1 * gr
                    v *= 0.999; v += 0.001 * gr * gr
                    m_hat = m / (1 - 0.9 ** t)
                    v_hat = v / (1 - 0.999 ** t)
                    p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

        # Evaluate on tuning split
        ord_means_tu, ord_stds_tu = compute_ord_stats(X_tr)  # use train stats for consistency
        z_ord_tu = ((X_tu[:, ord_idx].astype(np.float32) - ord_means_tu) / ord_stds_tu)
        z_cat_tu = np.zeros((X_tu.shape[0], n_cat * E), np.float32)
        for k, col in enumerate(cat_idx):
            z_cat_tu[:, k*E:(k+1)*E] = emb[k, X_tu[:, col]]
        z_tu = np.concatenate([z_ord_tu, z_cat_tu], axis=1)
        logits_tu = z_tu @ W + b
        pred_tu = logits_tu.argmax(axis=1)
        acc = (pred_tu == y_tu).mean()
        if acc > best_acc:
            best_acc, best_E = acc, E

    # Retrain on full data with best_E
    E = best_E
    total_dim = n_ord + n_cat * E
    emb = (rng.uniform(-0.1, 0.1, (n_cat, 256, E)) * math.sqrt(6.0 / (256 + E))).astype(np.float32)
    W = (rng.uniform(-0.1, 0.1, (total_dim, C)) * math.sqrt(6.0 / total_dim)).astype(np.float32)
    b = np.zeros(C, np.float32)
    ord_means, ord_stds = compute_ord_stats(X)

    params = [emb, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, max(1, math.ceil(SAMPLES / n)))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            z_ord = ((xb[:, ord_idx].astype(np.float32) - ord_means) / ord_stds)
            z_cat = np.zeros((bs, n_cat * E), np.float32)
            for k, col in enumerate(cat_idx):
                z_cat[:, k*E:(k+1)*E] = emb[k, xb[:, col]]
            z = np.concatenate([z_ord, z_cat], axis=1)

            logits = z @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            grad_out = probs / bs

            grad_W = z.T @ grad_out + L2 * W
            grad_b = grad_out.sum(axis=0)
            grad_z = grad_out @ W.T
            grad_z_ord = grad_z[:, :n_ord]
            grad_z_cat = grad_z[:, n_ord:]

            grad_emb = np.zeros_like(emb)
            for k, col in enumerate(cat_idx):
                idxs = xb[:, col]
                np.add.at(grad_emb[k], idxs, grad_z_cat[:, k*E:(k+1)*E])

            grads = [grad_emb, grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Save weights
    # Magic: 0x454D424C ('LMBE' little-endian for EMbedding Linear)
    # int32: magic, d, n_cat, n_ord, E, C
    # float32: emb [n_cat, 256, E]
    # float32: ord_means [n_ord], ord_stds [n_ord]
    # float32: W [total_dim, C], b [C]
    magic = 0x454D424C
    header = struct.pack("<6i", magic, d, n_cat, n_ord, E, C)
    emb_bytes = emb.astype("<f4").tobytes()
    ord_means_bytes = ord_means.astype("<f4").tobytes()
    ord_stds_bytes = ord_stds.astype("<f4").tobytes()
    W_bytes = W.astype("<f4").tobytes()
    b_bytes = b.astype("<f4").tobytes()
    out = header + emb_bytes + ord_means_bytes + ord_stds_bytes + W_bytes + b_bytes
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-emb: Logistic regression with embeddings for categorical features.
 * ./infer weights.bin in.rec N
 * Input record: d bytes (14 for adult). Output: 1 byte class index. */
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
    if (nw < 24 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], n_cat = hd[2], n_ord = hd[3], E = hd[4], C = hd[5];
    if (C < 1 || C > 256 || d != n_cat + n_ord) { fprintf(stderr, "bad weights dims\n"); return 3; }
    int total_dim = n_ord + n_cat * E;
    long expected = 24L
        + 4L * n_cat * 256 * E
        + 4L * n_ord * 2
        + 4L * total_dim * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 6);
    const float *ord_means = emb + (long)n_cat * 256 * E;
    const float *ord_stds = ord_means + n_ord;
    const float *W = ord_stds + n_ord;
    const float *b = W + (long)total_dim * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * total_dim);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Ordinal part
        int o = 0;
        for (int j = 0; j < d; j++) {
            // We need to know which columns are ordinal/categorical.
            // Hardcode for adult based on card types order:
            // 0:ord, 1:cat, 2:ord, 3:cat, 4:ord, 5:cat, 6:cat, 7:cat, 8:cat, 9:cat, 10:ord, 11:ord, 12:ord, 13:cat
            // But we don't have types in weights. We must encode the mapping.
            // Simpler: store the ordinal column indices in weights? For now, assume fixed adult schema.
            // Since this is a task-specific candidate, we can hardcode.
        }
        // The above approach requires knowing the column types. Since the weights don't store them,
        // we must either store them or hardcode. For a general family, we'd store the masks.
        // But the contract says "for this task". We'll hardcode the adult schema.
        // Let's redo: store ordinal indices in weights? We have space. Add n_ord ints after header.
        // But weights format is fixed. Let's change train.py to include ordinal indices.
        // However, the infer.c must match. Since we're writing both, we can modify the format.
        // Let's add ordinal indices to weights: after header, store n_ord int32 indices.
        // But then expected size changes. Let's do it properly.

        // Actually, the train.py I wrote doesn't save ordinal indices. I need to fix that.
        // But I already wrote train.py. Let me adjust: in train.py, save ord_idx array.
        // Since I can't change train.py now, I'll assume the infer.c knows the schema.
        // For the purpose of this answer, I'll hardcode the adult schema in infer.c.
        // The checker runs on adult task, so it's fine.
    }
    return 0;
}
```

Wait, I need to fix the infer.c to properly handle the column types. Since the train.py doesn't save the ordinal/categorical column indices, I need to either:
1. Modify train.py to save them
2. Hardcode in infer.c for adult

Since this is a task-specific candidate (for adult), hardcoding is acceptable. But the train.py I wrote uses the card's types to determine cat_idx and ord_idx. The infer.c must use the same mapping. Since the card is fixed for adult, I can hardcode the indices in infer.c.

Let me complete the infer.c with hardcoded indices for adult.

Also, I notice the train.py saves ord_means and ord_stds in order of ord_idx. So in infer.c, I need to apply them to the correct columns.

Let me write the complete infer.c with the hardcoded mapping.```hypothesis
label-emb: Logistic regression with embeddings for categorical features (8 of 14 columns).
Categorical columns get dense embeddings (dim 4) instead of one-hot, capturing category similarities;
ordinal columns remain z-scored. Reduces dimensionality vs one-hot while modeling interactions via
embedding space. Should beat label-linear (0.8524) at similar inference cost (~800 Ir/item).
Failure: if categories have no shared structure, accuracy <= label-linear.
```
```python
"""label-emb: Logistic regression with embeddings for categorical features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread. Categorical columns: learned embeddings (dim E).
Ordinal columns: z-scored. Linear layer on concatenated features. Adam, L2.
Hyperparameter E chosen on tuning split (last 20%), then trained on all data."""
import json, math, struct, sys

import numpy as np

SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 0.01, 1e-4, 64, 600000, 100
CANDIDATE_E = [2, 4, 8, 16]  # embedding dimensions to try


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
    C = card["classes"]
    types = card["input"]["types"]
    d = len(types)

    # Identify categorical and ordinal columns
    cat_idx = [j for j, t in enumerate(types) if t == "categorical"]
    ord_idx = [j for j, t in enumerate(types) if t == "ordinal"]
    n_cat, n_ord = len(cat_idx), len(ord_idx)

    # Split tuning (last 20% in file order)
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_tu, y_tu = X[split:], y[split:]

    # Ordinal stats
    def compute_ord_stats(X_sub):
        means = X_sub[:, ord_idx].mean(axis=0).astype(np.float32)
        stds = X_sub[:, ord_idx].std(axis=0).astype(np.float32)
        stds[stds == 0] = 1.0
        return means, stds

    rng = np.random.default_rng(SEED)
    best_E, best_acc = CANDIDATE_E[0], -1.0

    for E in CANDIDATE_E:
        emb = (rng.uniform(-0.1, 0.1, (n_cat, 256, E)) * math.sqrt(6.0 / (256 + E))).astype(np.float32)
        total_dim = n_ord + n_cat * E
        W = (rng.uniform(-0.1, 0.1, (total_dim, C)) * math.sqrt(6.0 / total_dim)).astype(np.float32)
        b = np.zeros(C, np.float32)

        ord_means, ord_stds = compute_ord_stats(X_tr)

        params = [emb, W, b]
        state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
        t = 0

        n_tr = X_tr.shape[0]
        epochs = min(MAX_EPOCHS, max(1, math.ceil(SAMPLES / n_tr)))

        for _ in range(epochs):
            order = rng.permutation(n_tr)
            for s in range(0, n_tr, BATCH):
                idx = order[s:s + BATCH]
                xb, yb = X_tr[idx], y_tr[idx]
                bs = len(yb)

                z_ord = ((xb[:, ord_idx].astype(np.float32) - ord_means) / ord_stds)
                z_cat = np.zeros((bs, n_cat * E), np.float32)
                for k, col in enumerate(cat_idx):
                    z_cat[:, k*E:(k+1)*E] = emb[k, xb[:, col]]
                z = np.concatenate([z_ord, z_cat], axis=1)

                logits = z @ W + b
                logits -= logits.max(axis=1, keepdims=True)
                exp_logits = np.exp(logits)
                probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                probs[np.arange(bs), yb] -= 1.0
                grad_out = probs / bs

                grad_W = z.T @ grad_out + L2 * W
                grad_b = grad_out.sum(axis=0)
                grad_z = grad_out @ W.T
                grad_z_cat = grad_z[:, n_ord:]

                grad_emb = np.zeros_like(emb)
                for k, col in enumerate(cat_idx):
                    idxs = xb[:, col]
                    np.add.at(grad_emb[k], idxs, grad_z_cat[:, k*E:(k+1)*E])

                grads = [grad_emb, grad_W, grad_b]
                t += 1
                for i, (p, gr) in enumerate(zip(params, grads)):
                    m, v = state[i]
                    m *= 0.9; m += 0.1 * gr
                    v *= 0.999; v += 0.001 * gr * gr
                    m_hat = m / (1 - 0.9 ** t)
                    v_hat = v / (1 - 0.999 ** t)
                    p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

        # Evaluate on tuning split
        ord_means_tu, ord_stds_tu = compute_ord_stats(X_tr)
        z_ord_tu = ((X_tu[:, ord_idx].astype(np.float32) - ord_means_tu) / ord_stds_tu)
        z_cat_tu = np.zeros((X_tu.shape[0], n_cat * E), np.float32)
        for k, col in enumerate(cat_idx):
            z_cat_tu[:, k*E:(k+1)*E] = emb[k, X_tu[:, col]]
        z_tu = np.concatenate([z_ord_tu, z_cat_tu], axis=1)
        logits_tu = z_tu @ W + b
        pred_tu = logits_tu.argmax(axis=1)
        acc = (pred_tu == y_tu).mean()
        if acc > best_acc:
            best_acc, best_E = acc, E

    # Retrain on full data with best_E
    E = best_E
    total_dim = n_ord + n_cat * E
    emb = (rng.uniform(-0.1, 0.1, (n_cat, 256, E)) * math.sqrt(6.0 / (256 + E))).astype(np.float32)
    W = (rng.uniform(-0.1, 0.1, (total_dim, C)) * math.sqrt(6.0 / total_dim)).astype(np.float32)
    b = np.zeros(C, np.float32)
    ord_means, ord_stds = compute_ord_stats(X)

    params = [emb, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, max(1, math.ceil(SAMPLES / n)))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)

            z_ord = ((xb[:, ord_idx].astype(np.float32) - ord_means) / ord_stds)
            z_cat = np.zeros((bs, n_cat * E), np.float32)
            for k, col in enumerate(cat_idx):
                z_cat[:, k*E:(k+1)*E] = emb[k, xb[:, col]]
            z = np.concatenate([z_ord, z_cat], axis=1)

            logits = z @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            grad_out = probs / bs

            grad_W = z.T @ grad_out + L2 * W
            grad_b = grad_out.sum(axis=0)
            grad_z = grad_out @ W.T
            grad_z_cat = grad_z[:, n_ord:]

            grad_emb = np.zeros_like(emb)
            for k, col in enumerate(cat_idx):
                idxs = xb[:, col]
                np.add.at(grad_emb[k], idxs, grad_z_cat[:, k*E:(k+1)*E])

            grads = [grad_emb, grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Save weights
    # Magic: 0x454D424C ('LMBE')
    # int32: magic, d, n_cat, n_ord, E, C
    # int32: ord_idx[n_ord]  (indices of ordinal columns)
    # float32: emb [n_cat, 256, E]
    # float32: ord_means [n_ord], ord_stds [n_ord]
    # float32: W [total_dim, C], b [C]
    magic = 0x454D424C
    header = struct.pack("<6i", magic, d, n_cat, n_ord, E, C)
    ord_idx_bytes = np.array(ord_idx, dtype=np.int32).tobytes()
    emb_bytes = emb.astype("<f4").tobytes()
    ord_means_bytes = ord_means.astype("<f4").tobytes()
    ord_stds_bytes = ord_stds.astype("<f4").tobytes()
    W_bytes = W.astype("<f4").tobytes()
    b_bytes = b.astype("<f4").tobytes()
    out = header + ord_idx_bytes + emb_bytes + ord_means_bytes + ord_stds_bytes + W_bytes + b_bytes
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* label-emb: Logistic regression with embeddings for categorical features.
 * ./infer weights.bin in.rec N
 * Input record: d bytes (14 for adult). Output: 1 byte class index. */
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
    if (nw < 24 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], n_cat = hd[2], n_ord = hd[3], E = hd[4], C = hd[5];
    if (C < 1 || C > 256 || d != n_cat + n_ord) { fprintf(stderr, "bad weights dims\n"); return 3; }
    int total_dim = n_ord + n_cat * E;

    // Read ordinal indices
    const int32_t *ord_idx = (const int32_t *)(hd + 6);
    // Validate we have enough space for ord_idx
    long header_size = 24L + 4L * n_ord;
    if (nw < header_size) { fprintf(stderr, "weights too small for ord_idx\n"); return 3; }

    const float *emb = (const float *)((const char *)(hd + 6) + 4L * n_ord);
    const float *ord_means = emb + (long)n_cat * 256 * E;
    const float *ord_stds = ord_means + n_ord;
    const float *W = ord_stds + n_ord;
    const float *b = W + (long)total_dim * C;

    long expected = header_size
        + 4L * n_cat * 256 * E
        + 4L * n_ord * 2
        + 4L * total_dim * C
        + 4L * C;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * total_dim);
    float *scores = malloc(sizeof(float) * C);
    if (!z || !scores) { fprintf(stderr, "oom\n"); exit(3); }

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Ordinal features
        for (int o = 0; o < n_ord; o++) {
            int col = ord_idx[o];
            float val = (float)xi[col];
            z[o] = (val - ord_means[o]) / ord_stds[o];
        }
        // Categorical embeddings
        int cat_ptr = 0;
        for (int j = 0; j < d; j++) {
            // Check if column j is ordinal
            int is_ord = 0;
            for (int o = 0; o < n_ord; o++) if (ord_idx[o] == j) { is_ord = 1; break; }
            if (is_ord) continue;
            // Categorical column
            const float *e = emb + (long)cat_ptr * 256 * E + (long)xi[j] * E;
            for (int e_i = 0; e_i < E; e_i++) z[n_ord + cat_ptr * E + e_i] = e[e_i];
            cat_ptr++;
        }

        // Linear layer
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int f = 0; f < total_dim; f++) {
            float zf = z[f];
            if (zf == 0.0f) continue;
            const float *w = W + (long)f * C;
            for (int c = 0; c < C; c++) scores[c] += zf * w[c];
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```