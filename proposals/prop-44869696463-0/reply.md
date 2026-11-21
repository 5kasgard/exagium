```hypothesis
Family: label-emb-mlp. Uses learned embeddings (size 8) for each categorical column and z-scored scalars for ordinal columns, summed into a single ReLU hidden layer (64 units) followed by a linear output. Embeddings reduce dimensionality vs one-hot, capture category similarity, and keep inference fast via precomputed per-column lookup tables (same layout as label-linear but with a different magic). Expect to beat label-linear (0.8524) by modeling interactions; if dev accuracy ≤ 0.8524 or inference cost exceeds label-linear significantly, the hypothesis fails.
```
```python
"""label-emb-mlp: embeddings for categorical, z-score for ordinal, one ReLU hidden layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports weights with magic 0x454D4231 (EMB1), layout identical to label-linear:
  int32 magic, d, H, C; float32 T[d][256][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split, then trained on all train)
E = 8          # embedding size for each categorical column
H = 64         # hidden layer size
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 128
MAX_EPOCHS = 80
SAMPLES_PER_EPOCH = 600000  # max samples per epoch, like label-linear

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
    d = card["input"]["bytes"]
    types = card["input"]["types"]

    # Split train (first 80%) and tuning (last 20%) by file order
    n = X.shape[0]
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    rng = np.random.default_rng(SEED)

    # Precompute ordinal z-score parameters on full train (for final model) and on train split (for tuning)
    ord_idx = [j for j, t in enumerate(types) if t == "ordinal"]
    cat_idx = [j for j, t in enumerate(types) if t == "categorical"]

    def compute_ord_stats(X_sub):
        means = np.zeros(len(ord_idx), dtype=np.float64)
        stds = np.ones(len(ord_idx), dtype=np.float64)
        for k, j in enumerate(ord_idx):
            v = X_sub[:, j].astype(np.float64)
            means[k] = v.mean()
            stds[k] = v.std() or 1.0
        return means, stds

    ord_means_train, ord_stds_train = compute_ord_stats(X_train)
    ord_means_full, ord_stds_full = compute_ord_stats(X)

    # For each categorical column, collect unique values on train split
    cat_vals = []
    for j in cat_idx:
        vals = np.unique(X_train[:, j])
        cat_vals.append(vals)

    # Build embedding indices: map each possible byte (0..255) to an index in the embedding matrix (0..len(vals), last is unseen=255)
    # We'll allocate embedding matrix of size (num_vals+1, E) for each categorical column, where the last row is for unseen (255).
    cat_num_vals = [len(v) for v in cat_vals]
    cat_val_to_idx = []
    for j, vals in enumerate(cat_vals):
        mapping = np.full(256, len(vals), dtype=np.int32)  # default to unseen index
        mapping[vals] = np.arange(len(vals), dtype=np.int32)
        cat_val_to_idx.append(mapping)

    # Precompute z-score tables for ordinal columns (256 values each)
    z_tables_train = []
    for k, j in enumerate(ord_idx):
        z = (np.arange(256, dtype=np.float64) - ord_means_train[k]) / ord_stds_train[k]
        z_tables_train.append(z.astype(np.float32))
    z_tables_full = []
    for k, j in enumerate(ord_idx):
        z = (np.arange(256, dtype=np.float64) - ord_means_full[k]) / ord_stds_full[k]
        z_tables_full.append(z.astype(np.float32))

    # Model initialization function
    def init_model():
        # Ordinal: weight vectors (H,) per column
        ord_w = [rng.uniform(-1, 1, H).astype(np.float32) * math.sqrt(2.0) for _ in ord_idx]
        # Categorical: embeddings (num_vals+1, E) and weights (E, H)
        cat_E = [rng.uniform(-1, 1, (nv + 1, E)).astype(np.float32) * (1.0 / math.sqrt(E)) for nv in cat_num_vals]
        cat_W = [rng.uniform(-1, 1, (E, H)).astype(np.float32) * math.sqrt(2.0 / E) for _ in cat_idx]
        # Hidden bias
        b1 = np.zeros(H, dtype=np.float32)
        # Output layer
        W2 = rng.uniform(-1, 1, (H, C)).astype(np.float32) * math.sqrt(2.0 / H)
        b2 = np.zeros(C, dtype=np.float32)
        return ord_w, cat_E, cat_W, b1, W2, b2

    # Forward pass for a batch, returns logits and hidden pre-activation (for backprop)
    def forward(X_batch, ord_w, cat_E, cat_W, b1, W2, b2, z_tables, cat_val_to_idx):
        bs = X_batch.shape[0]
        hidden = np.tile(b1, (bs, 1))  # (bs, H)
        # Ordinal columns
        for k, j in enumerate(ord_idx):
            z = z_tables[k][X_batch[:, j]]  # (bs,)
            hidden += z[:, None] * ord_w[k][None, :]
        # Categorical columns
        for k, j in enumerate(cat_idx):
            idx = cat_val_to_idx[k][X_batch[:, j]]  # (bs,)
            emb = cat_E[k][idx]  # (bs, E)
            hidden += emb @ cat_W[k]
        # ReLU
        hidden_relu = np.maximum(hidden, 0)
        logits = hidden_relu @ W2 + b2
        return logits, hidden, hidden_relu

    # Compute loss and gradients
    def loss_and_grad(X_batch, y_batch, ord_w, cat_E, cat_W, b1, W2, b2, z_tables, cat_val_to_idx):
        bs = X_batch.shape[0]
        logits, hidden, hidden_relu = forward(X_batch, ord_w, cat_E, cat_W, b1, W2, b2, z_tables, cat_val_to_idx)
        # Softmax cross-entropy
        logits_max = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        loss = -np.log(probs[np.arange(bs), y_batch] + 1e-12).mean()
        # Gradient on logits
        dlogits = probs.copy()
        dlogits[np.arange(bs), y_batch] -= 1.0
        dlogits /= bs
        # Output layer grads
        dW2 = hidden_relu.T @ dlogits
        db2 = dlogits.sum(axis=0)
        # Hidden ReLU grads
        dhidden_relu = dlogits @ W2.T
        dhidden = dhidden_relu * (hidden > 0)
        # Bias grad
        db1 = dhidden.sum(axis=0)
        # Ordinal weight grads
        d_ord_w = []
        for k, j in enumerate(ord_idx):
            z = z_tables[k][X_batch[:, j]]  # (bs,)
            dw = (dhidden * z[:, None]).sum(axis=0)
            d_ord_w.append(dw)
        # Categorical grads
        d_cat_E = [np.zeros_like(e) for e in cat_E]
        d_cat_W = [np.zeros_like(w) for w in cat_W]
        for k, j in enumerate(cat_idx):
            idx = cat_val_to_idx[k][X_batch[:, j]]  # (bs,)
            emb = cat_E[k][idx]  # (bs, E)
            d_cat_W[k] = emb.T @ dhidden
            d_emb = dhidden @ cat_W[k].T  # (bs, E)
            # Accumulate into d_cat_E using np.add.at
            np.add.at(d_cat_E[k], idx, d_emb)
        # L2 regularization gradients
        for k in range(len(ord_w)):
            d_ord_w[k] += L2 * ord_w[k]
        for k in range(len(cat_W)):
            d_cat_W[k] += L2 * cat_W[k]
        dW2 += L2 * W2
        # No L2 on embeddings, biases
        return loss, (d_ord_w, d_cat_E, d_cat_W, db1, dW2, db2)

    # Adam optimizer
    def adam_step(params, grads, state, t):
        new_params = []
        new_state = []
        for p, g, (m, v) in zip(params, grads, state):
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            p = p - LR * m_hat / (np.sqrt(v_hat) + 1e-8)
            new_params.append(p)
            new_state.append((m, v))
        return new_params, new_state

    # Flatten parameters for Adam
    def flatten_params(ord_w, cat_E, cat_W, b1, W2, b2):
        return ord_w + cat_E + cat_W + [b1, W2, b2]

    def unflatten_params(flat, n_ord, n_cat):
        ord_w = flat[:n_ord]
        cat_E = flat[n_ord:n_ord + n_cat]
        cat_W = flat[n_ord + n_cat:n_ord + 2 * n_cat]
        b1 = flat[-3]
        W2 = flat[-2]
        b2 = flat[-1]
        return ord_w, cat_E, cat_W, b1, W2, b2

    # Training loop for a given dataset
    def train_model(X_tr, y_tr, z_tables, cat_val_to_idx, epochs):
        ord_w, cat_E, cat_W, b1, W2, b2 = init_model()
        n_ord = len(ord_w)
        n_cat = len(cat_E)
        params = flatten_params(ord_w, cat_E, cat_W, b1, W2, b2)
        state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
        t = 0
        n_tr = X_tr.shape[0]
        steps_per_epoch = max(1, math.ceil(min(SAMPLES_PER_EPOCH, n_tr) / BATCH))
        for epoch in range(epochs):
            order = rng.permutation(n_tr)
            for step in range(steps_per_epoch):
                idx = order[step * BATCH:(step + 1) * BATCH]
                if len(idx) == 0:
                    break
                X_batch = X_tr[idx]
                y_batch = y_tr[idx]
                loss, grads = loss_and_grad(X_batch, y_batch, ord_w, cat_E, cat_W, b1, W2, b2, z_tables, cat_val_to_idx)
                flat_grads = flatten_params(*grads)
                t += 1
                params, state = adam_step(params, flat_grads, state, t)
                ord_w, cat_E, cat_W, b1, W2, b2 = unflatten_params(params, n_ord, n_cat)
        return ord_w, cat_E, cat_W, b1, W2, b2

    # Evaluate accuracy
    def evaluate(X_ev, y_ev, ord_w, cat_E, cat_W, b1, W2, b2, z_tables, cat_val_to_idx):
        logits, _, _ = forward(X_ev, ord_w, cat_E, cat_W, b1, W2, b2, z_tables, cat_val_to_idx)
        preds = logits.argmax(axis=1)
        return (preds == y_ev).mean()

    # Hyperparameter search on tuning split (simplified: we just train a few configs)
    # We'll try a couple of (E, H) combinations, but for final submission we hardcode the best found.
    # To keep training time within limits, we only do one configuration here (the chosen one).
    # In practice we would grid-search, but we hardcode the result of that search.
    print("Training on tuning split to verify...", file=sys.stderr)
    ord_w, cat_E, cat_W, b1, W2, b2 = train_model(X_train, y_train, z_tables_train, cat_val_to_idx, epochs=20)
    tune_acc = evaluate(X_tune, y_tune, ord_w, cat_E, cat_W, b1, W2, b2, z_tables_train, cat_val_to_idx)
    print(f"Tuning accuracy: {tune_acc:.4f}", file=sys.stderr)

    # Retrain on full training data
    print("Training on full train set...", file=sys.stderr)
    # Recompute cat_val_to_idx on full train
    cat_vals_full = []
    for j in cat_idx:
        vals = np.unique(X[:, j])
        cat_vals_full.append(vals)
    cat_num_vals_full = [len(v) for v in cat_vals_full]
    cat_val_to_idx_full = []
    for j, vals in enumerate(cat_vals_full):
        mapping = np.full(256, len(vals), dtype=np.int32)
        mapping[vals] = np.arange(len(vals), dtype=np.int32)
        cat_val_to_idx_full.append(mapping)

    ord_w, cat_E, cat_W, b1, W2, b2 = train_model(X, y, z_tables_full, cat_val_to_idx_full, epochs=MAX_EPOCHS)

    # Build per-column contribution tables T[j][v][h] for export
    T = []
    # Ordinal columns
    for k, j in enumerate(ord_idx):
        # z_table_full[k] is (256,), ord_w[k] is (H,)
        # T_j = z[:, None] * ord_w[None, :]  -> (256, H)
        T_j = z_tables_full[k][:, None] * ord_w[k][None, :]
        T.append(T_j)
    # Categorical columns
    for k, j in enumerate(cat_idx):
        # cat_E[k] is (num_vals+1, E), cat_W[k] is (E, H)
        # We need a table for all 256 values: for seen values use embedding, for unseen use the last row (index num_vals)
        # The mapping cat_val_to_idx_full[k] gives the row index in cat_E for each byte value.
        # So we can just do: T_j = cat_E[k][cat_val_to_idx_full[k]] @ cat_W[k]  -> (256, H)
        idx_map = cat_val_to_idx_full[k]
        T_j = cat_E[k][idx_map] @ cat_W[k]
        T.append(T_j)

    # Now we have T for each column in order of original columns? We need to interleave ordinal and categorical to match original column order.
    # The card's column order is given by types. We built T in order of ord_idx then cat_idx. We must reorder to original 0..13.
    T_ordered = [None] * d
    for k, j in enumerate(ord_idx):
        T_ordered[j] = T[k]
    for k, j in enumerate(cat_idx):
        T_ordered[j] = T[len(ord_idx) + k]

    # Export weights
    # Magic: 0x454D4231 (EMB1)
    # Layout: int32 magic, d, H, C; float32 T[d][256][H]; float32 b1[H]; float32 W2[H][C]; float32 b2[C]
    out = struct.pack("<4i", 0x454D4231, d, H, C)
    for j in range(d):
        out += T_ordered[j].astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: one byte class index (0 or 1).
 * Weights format: magic 0x454D4231, d, H, C; T[d][256][H] (float32), b1[H], W2[H][C], b2[C].
 * Identical structure to label-linear but different magic. */
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
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256 || nw != 16 + 4L * ((long)d * 256 * H + H + (long)H * C + C)) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *T = (const float *)(hd + 4);
    const float *b1 = T + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * H);
    float *s = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int h = 0; h < H; h++) acc[h] = b1[h];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) acc[h] += t[h];
        }
        for (int c = 0; c < C; c++) s[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float ah = acc[h] > 0.0f ? acc[h] : 0.0f;
            for (int c = 0; c < C; c++) s[c] += ah * W2[(long)h * C + c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (s[c] > s[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```