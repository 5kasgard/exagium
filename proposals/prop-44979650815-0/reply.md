```hypothesis
Family: label-emb. A linear model with learned embeddings for categorical features (ordinal features are z-scored). This reduces parameters vs one-hot, regularizes high-cardinality categoricals, and captures category similarities. It should beat label-linear on accuracy because one-hot overfits on sparse categorical columns; embeddings generalize better. If it fails to beat label-linear, it indicates the categorical features have no exploitable similarity structure or the dataset is too small to learn useful embeddings.
```
```python
"""label-emb: linear model with learned embeddings for categorical columns, z-scored ordinals.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Weights format: magic 0x454D424C, d, C, E, n_ord, n_cat, d bytes column types (0=ord,1=cat),
then per column: ordinal -> mean,std (float32); categorical -> n_cats (uint32), then (n_cats+1)*E float32 embeddings,
then linear weights W[D][C] (float32), bias b[C] (float32). D = n_ord + n_cat*E.
Hyperparameters (E, lr) chosen on last 20% tuning split, then retrained on all data."""
import json, math, struct, sys

import numpy as np

SEED = 1
TUNING_FRAC = 0.2
E_CANDIDATES = [4, 8, 16]
LR_CANDIDATES = [0.01, 0.03, 0.1]
L2 = 1e-4
BATCH = 256
MAX_EPOCHS_TUNE = 20
MAX_EPOCHS_FULL = 100
SAMPLES = 600000

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def preprocess(card, X, col_types, col_stats, fit=False):
    """Convert raw byte matrix X (n,d) to feature matrix Z (n, D).
    col_stats: for ordinal -> (mean, std); for categorical -> (n_cats, cat_to_idx).
    If fit=True, compute stats from X and return them."""
    n, d = X.shape
    if fit:
        col_types = []
        col_stats = []
    Z_parts = []
    for j in range(d):
        t = card["input"]["types"][j]
        v = X[:, j]
        if t == "ordinal":
            if fit:
                mu = v.mean()
                sd = v.std()
                if sd == 0:
                    sd = 1.0
                col_types.append(0)
                col_stats.append((mu, sd))
            else:
                mu, sd = col_stats[j]
            Z_parts.append(((v - mu) / sd).reshape(-1, 1).astype(np.float32))
        else:
            if fit:
                vals = np.unique(v)
                n_cats = len(vals)
                cat_to_idx = {val: i for i, val in enumerate(vals)}
                col_types.append(1)
                col_stats.append((n_cats, cat_to_idx))
            else:
                n_cats, cat_to_idx = col_stats[j]
            idx = np.array([cat_to_idx.get(int(x), n_cats) for x in v], dtype=np.int32)
            Z_parts.append(idx.reshape(-1, 1))
    return Z_parts, col_types, col_stats

def build_feature_matrix(Z_parts, col_types, embeddings, E):
    """Z_parts: list of arrays per column. For ordinal: (n,1) float32; for cat: (n,1) int32.
    embeddings: list of (n_cats+1, E) float32 for each cat column."""
    n = Z_parts[0].shape[0]
    feat = []
    cat_idx = 0
    for j, t in enumerate(col_types):
        if t == 0:
            feat.append(Z_parts[j])
        else:
            emb = embeddings[cat_idx]
            idx = Z_parts[j].flatten()
            feat.append(emb[idx])
            cat_idx += 1
    return np.concatenate(feat, axis=1).astype(np.float32)

def softmax_cross_entropy_grad(logits, y):
    """logits (n,C), y (n,) int. Returns gradient of loss w.r.t logits (n,C)."""
    n = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    return probs / n

def accuracy(logits, y):
    return np.mean(np.argmax(logits, axis=1) == y)

def train_model(X_feat, y, D, C, E, lr, epochs, rng, L2=1e-4, batch=256):
    """Train linear layer on top of fixed features X_feat (n,D). Returns W,b."""
    n = X_feat.shape[0]
    W = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / D)).astype(np.float32)
    b = np.zeros(C, np.float32)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s+batch]
            xb = X_feat[idx]
            yb = y[idx]
            logits = xb @ W + b
            g = softmax_cross_entropy_grad(logits, yb)  # (batch, C)
            grad_W = xb.T @ g + L2 * W
            grad_b = g.sum(axis=0)
            t += 1
            beta1, beta2 = 0.9, 0.999
            mW = beta1 * mW + (1 - beta1) * grad_W
            vW = beta2 * vW + (1 - beta2) * grad_W * grad_W
            mb = beta1 * mb + (1 - beta1) * grad_b
            vb = beta2 * vb + (1 - beta2) * grad_b * grad_b
            mW_hat = mW / (1 - beta1 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            mb_hat = mb / (1 - beta1 ** t)
            vb_hat = vb / (1 - beta2 ** t)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + 1e-8)
            b -= lr * mb_hat / (np.sqrt(vb_hat) + 1e-8)
    return W, b

def train_embeddings(X_parts, y, col_types, embeddings, W, b, E, lr, epochs, rng, L2=1e-4, batch=256):
    """Jointly fine-tune embeddings and linear layer. X_parts: list per column (ordinal: float32, cat: int32).
    embeddings: list of (n_cats+1, E) float32. Returns updated embeddings, W, b."""
    n = X_parts[0].shape[0]
    cat_indices = [j for j, t in enumerate(col_types) if t == 1]
    # Initialize Adam states for embeddings
    m_emb = [np.zeros_like(emb) for emb in embeddings]
    v_emb = [np.zeros_like(emb) for emb in embeddings]
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s+batch]
            # Build feature matrix for this batch
            feat_parts = []
            for j, tcol in enumerate(col_types):
                if tcol == 0:
                    feat_parts.append(X_parts[j][idx])
                else:
                    cat_idx = cat_indices.index(j)
                    emb = embeddings[cat_idx]
                    cat_idx_batch = X_parts[j][idx].flatten()
                    feat_parts.append(emb[cat_idx_batch])
            xb = np.concatenate(feat_parts, axis=1).astype(np.float32)
            yb = y[idx]
            logits = xb @ W + b
            g = softmax_cross_entropy_grad(logits, yb)  # (batch, C)
            # Gradient for W, b
            grad_W = xb.T @ g + L2 * W
            grad_b = g.sum(axis=0)
            # Gradient for embeddings: for each cat column, accumulate grad_emb[cat_idx] += g * W_cat
            # W_cat is the slice of W corresponding to that embedding dimension.
            # We need to know the column offsets in the feature vector.
            offset = 0
            grad_emb = [np.zeros_like(emb) for emb in embeddings]
            for j, tcol in enumerate(col_types):
                if tcol == 0:
                    offset += 1
                else:
                    cat_idx = cat_indices.index(j)
                    W_cat = W[offset:offset+E]  # (E, C)
                    # g is (batch, C), W_cat is (E, C). grad for each embedding vector: sum over batch of g * (embedding grad?)
                    # Actually, feature for cat column is embedding lookup: f = emb[cat_idx] (E,).
                    # logits = sum_k f_k * W_cat[k] + ... So d logits / d emb[cat_idx, k] = W_cat[k].
                    # So grad_emb[cat_idx, k] = sum_batch g * W_cat[k]? Wait: grad of loss w.r.t f is g @ W_cat.T (batch, E).
                    # Then for each example, the gradient for the embedding row is that row's gradient.
                    # We can accumulate by scattering.
                    cat_idx_batch = X_parts[j][idx].flatten()
                    # g shape (batch, C), W_cat shape (E, C) -> g @ W_cat.T = (batch, E)
                    grad_f = g @ W_cat.T  # (batch, E)
                    # Now we need to add grad_f to grad_emb[cat_idx] at rows cat_idx_batch.
                    np.add.at(grad_emb[cat_idx], cat_idx_batch, grad_f)
                    # Add L2 regularization on embeddings
                    grad_emb[cat_idx] += L2 * emb
                    offset += E
            # Adam updates
            t += 1
            beta1, beta2 = 0.9, 0.999
            # embeddings
            for i in range(len(embeddings)):
                m_emb[i] = beta1 * m_emb[i] + (1 - beta1) * grad_emb[i]
                v_emb[i] = beta2 * v_emb[i] + (1 - beta2) * grad_emb[i] * grad_emb[i]
                m_hat = m_emb[i] / (1 - beta1 ** t)
                v_hat = v_emb[i] / (1 - beta2 ** t)
                embeddings[i] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            # W, b
            mW = beta1 * mW + (1 - beta1) * grad_W
            vW = beta2 * vW + (1 - beta2) * grad_W * grad_W
            mb = beta1 * mb + (1 - beta1) * grad_b
            vb = beta2 * vb + (1 - beta2) * grad_b * grad_b
            mW_hat = mW / (1 - beta1 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            mb_hat = mb / (1 - beta1 ** t)
            vb_hat = vb / (1 - beta2 ** t)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + 1e-8)
            b -= lr * mb_hat / (np.sqrt(vb_hat) + 1e-8)
    return embeddings, W, b

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # Load all data
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_all = X_all.shape[0]
    # Split tuning (last 20%)
    n_tune = int(n_all * TUNING_FRAC)
    n_train = n_all - n_tune
    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_tune, y_tune = X_all[n_train:], y_all[n_train:]
    
    rng = np.random.default_rng(SEED)
    
    # Preprocess training data to get column stats
    Z_parts_train, col_types, col_stats = preprocess(card, X_train, None, None, fit=True)
    n_ord = sum(1 for t in col_types if t == 0)
    n_cat = sum(1 for t in col_types if t == 1)
    cat_indices = [j for j, t in enumerate(col_types) if t == 1]
    cat_ncats = [col_stats[j][0] for j in cat_indices]
    
    # Preprocess tuning data using same stats
    Z_parts_tune, _, _ = preprocess(card, X_tune, col_types, col_stats, fit=False)
    
    best_acc = -1
    best_E = E_CANDIDATES[0]
    best_lr = LR_CANDIDATES[0]
    best_embeddings = None
    best_W = None
    best_b = None
    
    for E in E_CANDIDATES:
        for lr in LR_CANDIDATES:
            # Initialize embeddings
            embeddings = []
            for n_cats in cat_ncats:
                emb = (rng.uniform(-1, 1, (n_cats + 1, E)) * math.sqrt(6.0 / (n_cats + 1))).astype(np.float32)
                embeddings.append(emb)
            D = n_ord + n_cat * E
            # Train linear layer on frozen embeddings first (few epochs)
            X_feat_train = build_feature_matrix(Z_parts_train, col_types, embeddings, E)
            W, b = train_model(X_feat_train, y_train, D, C, E, lr, epochs=5, rng=rng, L2=L2, batch=BATCH)
            # Joint fine-tuning
            embeddings, W, b = train_embeddings(Z_parts_train, y_train, col_types, embeddings, W, b, E, lr, epochs=MAX_EPOCHS_TUNE, rng=rng, L2=L2, batch=BATCH)
            # Evaluate on tuning
            X_feat_tune = build_feature_matrix(Z_parts_tune, col_types, embeddings, E)
            logits_tune = X_feat_tune @ W + b
            acc = accuracy(logits_tune, y_tune)
            if acc > best_acc:
                best_acc = acc
                best_E = E
                best_lr = lr
                best_embeddings = [emb.copy() for emb in embeddings]
                best_W = W.copy()
                best_b = b.copy()
            print(f"E={E}, lr={lr}: tune acc={acc:.4f}", file=sys.stderr)
    
    print(f"Best: E={best_E}, lr={best_lr}, acc={best_acc:.4f}", file=sys.stderr)
    
    # Retrain on full data with best hyperparameters
    Z_parts_full, col_types_full, col_stats_full = preprocess(card, X_all, None, None, fit=True)
    n_ord_full = sum(1 for t in col_types_full if t == 0)
    n_cat_full = sum(1 for t in col_types_full if t == 1)
    cat_indices_full = [j for j, t in enumerate(col_types_full) if t == 1]
    cat_ncats_full = [col_stats_full[j][0] for j in cat_indices_full]
    
    embeddings = []
    for n_cats in cat_ncats_full:
        emb = (rng.uniform(-1, 1, (n_cats + 1, best_E)) * math.sqrt(6.0 / (n_cats + 1))).astype(np.float32)
        embeddings.append(emb)
    D_full = n_ord_full + n_cat_full * best_E
    X_feat_full = build_feature_matrix(Z_parts_full, col_types_full, embeddings, best_E)
    W, b = train_model(X_feat_full, y_all, D_full, C, best_E, best_lr, epochs=5, rng=rng, L2=L2, batch=BATCH)
    embeddings, W, b = train_embeddings(Z_parts_full, y_all, col_types_full, embeddings, W, b, best_E, best_lr, epochs=MAX_EPOCHS_FULL, rng=rng, L2=L2, batch=BATCH)
    
    # Write weights file
    # Compute total size for validation
    # Header: 6 int32 = 24 bytes
    # Column types: d bytes
    # Per column: ordinal -> 2 float32 (8 bytes); categorical -> 1 uint32 + (n_cats+1)*E float32
    # Linear: D_full * C float32 + C float32
    total_size = 24 + d
    for j in range(d):
        if col_types_full[j] == 0:
            total_size += 8
        else:
            n_cats = col_stats_full[j][0]
            total_size += 4 + (n_cats + 1) * best_E * 4
    total_size += D_full * C * 4 + C * 4
    
    # Pack
    out = bytearray()
    out += struct.pack("<6I", 0x454D424C, d, C, best_E, n_ord_full, n_cat_full)
    out += bytes(col_types_full)  # d bytes
    for j in range(d):
        if col_types_full[j] == 0:
            mu, sd = col_stats_full[j]
            out += struct.pack("<ff", mu, sd)
        else:
            n_cats, _ = col_stats_full[j]
            out += struct.pack("<I", n_cats)
            emb = embeddings[cat_indices_full.index(j)]
            out += emb.astype("<f4").tobytes()
    out += best_W.astype("<f4").tobytes()
    out += best_b.astype("<f4").tobytes()
    
    assert len(out) == total_size, f"Size mismatch: {len(out)} vs {total_size}"
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d from weights). Output: one byte class index.
 * Weights format: magic 0x454D424C, d, C, E, n_ord, n_cat, d bytes col_types,
 * then per column: ordinal -> mean,std (float32); categorical -> n_cats (uint32), then (n_cats+1)*E float32 embeddings,
 * then W[D][C] (float32), b[C] (float32). D = n_ord + n_cat*E. */
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 24) { fprintf(stderr, "weights too small\n"); return 3; }
    uint32_t *hdr = (uint32_t *)wb;
    if (hdr[0] != 0x454D424C) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hdr[1], C = hdr[2], E = hdr[3], n_ord = hdr[4], n_cat = hdr[5];
    if (d <= 0 || C <= 0 || C > 256 || E <= 0) { fprintf(stderr, "bad header values\n"); return 3; }
    unsigned char *col_types = wb + 24;
    if (24 + d > nw) { fprintf(stderr, "truncated col_types\n"); return 3; }
    // Parse columns to compute expected size and offsets
    const unsigned char *ptr = col_types + d;
    int D = 0;
    int cat_col_idx = 0;
    int *cat_ncats = malloc(n_cat * sizeof(int));
    int *cat_emb_offset = malloc(n_cat * sizeof(int)); // offset in embeddings buffer
    int *ord_mean_std_offset = malloc(n_ord * sizeof(int)); // offset in ordinal buffer
    float *ord_buf = NULL;
    float *emb_buf = NULL;
    // We'll do two passes: first compute sizes and check bounds, then set pointers.
    // But we can just compute total expected size and validate nw.
    long expected = 24 + d;
    for (int j = 0; j < d; j++) {
        if (col_types[j] == 0) { // ordinal
            expected += 8; // two float32
        } else { // categorical
            if (ptr + 4 > wb + nw) { fprintf(stderr, "truncated n_cats\n"); return 3; }
            uint32_t n_cats = *(uint32_t *)ptr;
            ptr += 4;
            cat_ncats[cat_col_idx] = n_cats;
            expected += (long)(n_cats + 1) * E * 4;
            cat_col_idx++;
        }
    }
    // Now ptr points to start of embeddings/ordinal data? Actually ordinal data is interleaved.
    // Let's re-parse properly with pointers.
    // We'll allocate arrays for ordinal means/stds and embedding pointers.
    float *ord_means = malloc(n_ord * sizeof(float));
    float *ord_stds = malloc(n_ord * sizeof(float));
    float **embeddings = malloc(n_cat * sizeof(float *));
    ptr = col_types + d;
    int ord_idx = 0, cat_idx = 0;
    for (int j = 0; j < d; j++) {
        if (col_types[j] == 0) {
            if (ptr + 8 > wb + nw) { fprintf(stderr, "truncated ordinal stats\n"); return 3; }
            ord_means[ord_idx] = *(float *)ptr;
            ord_stds[ord_idx] = *(float *)(ptr + 4);
            ptr += 8;
            ord_idx++;
        } else {
            if (ptr + 4 > wb + nw) { fprintf(stderr, "truncated n_cats\n"); return 3; }
            uint32_t n_cats = *(uint32_t *)ptr;
            ptr += 4;
            if (ptr + (long)(n_cats + 1) * E * 4 > wb + nw) { fprintf(stderr, "truncated embeddings\n"); return 3; }
            embeddings[cat_idx] = (float *)ptr;
            ptr += (long)(n_cats + 1) * E * 4;
            cat_idx++;
        }
    }
    // Now ptr points to W matrix
    D = n_ord + n_cat * E;
    if (ptr + (long)D * C * 4 > wb + nw) { fprintf(stderr, "truncated W\n"); return 3; }
    float *W = (float *)ptr;
    ptr += (long)D * C * 4;
    if (ptr + (long)C * 4 > wb + nw) { fprintf(stderr, "truncated b\n"); return 3; }
    float *b = (float *)ptr;
    ptr += (long)C * 4;
    if (ptr != wb + nw) { fprintf(stderr, "weights size mismatch: expected %ld, got %ld\n", ptr - wb, nw); return 3; }
    
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *feat = malloc(D * sizeof(float));
    float *scores = malloc(C * sizeof(float));
    
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Build feature vector
        int fidx = 0;
        ord_idx = 0; cat_idx = 0;
        for (int j = 0; j < d; j++) {
            if (col_types[j] == 0) {
                float val = (xi[j] - ord_means[ord_idx]) / ord_stds[ord_idx];
                feat[fidx++] = val;
                ord_idx++;
            } else {
                int cat_val = xi[j];
                int n_cats = cat_ncats[cat_idx];
                float *emb = embeddings[cat_idx];
                if (cat_val >= n_cats) cat_val = n_cats; // unseen
                float *src = emb + (long)cat_val * E;
                for (int e = 0; e < E; e++) feat[fidx++] = src[e];
                cat_idx++;
            }
        }
        // Compute scores = feat * W + b
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int f = 0; f < D; f++) {
            float fv = feat[f];
            if (fv != 0.0f) {
                float *wf = W + (long)f * C;
                for (int c = 0; c < C; c++) scores[c] += fv * wf[c];
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```