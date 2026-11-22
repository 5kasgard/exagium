```hypothesis
Family: label-embedmlp (per-position byte embeddings + one-hidden-layer ReLU MLP)
Why: The 16-byte context has strong positional n-gram structure. Linear model (0.3375) ignores cross-position interactions; GBDT (0.3923) captures them but is 18× slower. Low-rank per-position embeddings (E=8) feed a small hidden layer (H=64), modeling interactions with ~40k parameters (vs 266k for linear) and far fewer ops than trees. Expected dev accuracy >0.35 at <15k Ir/item.
Failure: If validation accuracy ≤0.34 or inference cost >20k Ir/item, the family does not improve the Pareto frontier.
```

```python
"""label-embedmlp: per-position byte embeddings + 1-hidden-layer ReLU MLP.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Hyperparameters (E, H) chosen on last 20% of train.in (tuning split), then retrained on all train.
Weight format: magic 0x454D4244, d, H, C, T[d*256*H], b1[H], W2[H*C], b2[C] (all float32 LE)."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid for tuning split
GRID = [(4, 32), (4, 64), (8, 32), (8, 64)]
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def epochs_for(n):
    return min(MAX_EPOCHS, math.ceil(SAMPLES / n))

def train_eval(X_tr, y_tr, X_val, y_val, d, C, E, H, rng):
    n_tr = X_tr.shape[0]
    # Parameters
    Emb = [rng.uniform(-0.1, 0.1, (256, E)).astype(np.float32) for _ in range(d)]
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    params = [Emb, W1, b1, W2, b2]
    # Adam state
    state = []
    for p in [W1, b1, W2, b2]:
        state.append((np.zeros_like(p), np.zeros_like(p)))
    emb_state = [[(np.zeros_like(e), np.zeros_like(e)) for e in Emb]]

    def forward(X, training=True):
        bs = X.shape[0]
        emb_out = np.empty((bs, d, E), np.float32)
        for j in range(d):
            emb_out[:, j, :] = Emb[j][X[:, j]]
        flat = emb_out.reshape(bs, d * E)
        hidden_pre = flat @ W1 + b1
        hidden = np.maximum(hidden_pre, 0, out=hidden_pre)  # reuse array for hidden
        logits = hidden @ W2 + b2
        return logits, hidden, flat, emb_out

    def loss_grad(logits, y):
        bs = logits.shape[0]
        logits_max = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        dlogits = probs
        dlogits[np.arange(bs), y] -= 1.0
        dlogits /= bs
        return dlogits

    for _ in range(epochs_for(n_tr)):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_tr[idx], y_tr[idx]
            logits, hidden, flat, emb_out = forward(Xb)
            dlogits = loss_grad(logits, yb)
            # Output layer grads
            dW2 = hidden.T @ dlogits
            db2 = dlogits.sum(axis=0)
            # Hidden layer grads
            dhidden = dlogits @ W2.T
            dhidden_pre = dhidden * (hidden > 0)
            # First layer grads
            dflat = dhidden_pre @ W1.T
            dW1 = flat.T @ dhidden_pre
            db1 = dhidden_pre.sum(axis=0)
            # Embedding grads
            demb_out = dflat.reshape(Xb.shape[0], d, E)
            dEmb = [np.zeros_like(e) for e in Emb]
            for j in range(d):
                np.add.at(dEmb[j], Xb[:, j], demb_out[:, j, :])
            # L2 regularization
            dW1 += 2 * L2 * W1
            dW2 += 2 * L2 * W2
            for j in range(d):
                dEmb[j] += 2 * L2 * Emb[j]
            # Adam updates
            t = 0  # we'll use a global step counter; simpler: increment per batch
            # We'll manage step counter outside
            # For simplicity, use the same Adam code as example but per param group
            # We'll just update with a simple Adam implementation here
            grads = [dW1, db1, dW2, db2]
            for i, (p, gr) in enumerate(zip([W1, b1, W2, b2], grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                # bias correction (approximate, using step count per epoch)
                # We'll keep a global step
                pass
            # Actually, let's use a global step counter as in example
    # The above is incomplete; we need a clean Adam with step counter.
    # Given the complexity, we'll instead use the exact Adam from the example but adapted.
    # Let's rewrite the training loop with a global step.

def train_model(X, y, d, C, E, H, rng, epochs):
    n = X.shape[0]
    Emb = [rng.uniform(-0.1, 0.1, (256, E)).astype(np.float32) for _ in range(d)]
    W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    # Adam state
    state = [(np.zeros_like(W1), np.zeros_like(W1)),
             (np.zeros_like(b1), np.zeros_like(b1)),
             (np.zeros_like(W2), np.zeros_like(W2)),
             (np.zeros_like(b2), np.zeros_like(b2))]
    emb_state = [[(np.zeros_like(e), np.zeros_like(e)) for e in Emb]]
    step = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            bs = Xb.shape[0]
            # Forward
            emb_out = np.empty((bs, d, E), np.float32)
            for j in range(d):
                emb_out[:, j, :] = Emb[j][Xb[:, j]]
            flat = emb_out.reshape(bs, d * E)
            hidden_pre = flat @ W1 + b1
            hidden = np.maximum(hidden_pre, 0)
            logits = hidden @ W2 + b2
            # Loss gradient
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            dlogits = probs
            dlogits[np.arange(bs), yb] -= 1.0
            dlogits /= bs
            # Backward
            dW2 = hidden.T @ dlogits
            db2 = dlogits.sum(axis=0)
            dhidden = dlogits @ W2.T
            dhidden_pre = dhidden * (hidden > 0)
            dflat = dhidden_pre @ W1.T
            dW1 = flat.T @ dhidden_pre
            db1 = dhidden_pre.sum(axis=0)
            demb_out = dflat.reshape(bs, d, E)
            dEmb = [np.zeros_like(e) for e in Emb]
            for j in range(d):
                np.add.at(dEmb[j], Xb[:, j], demb_out[:, j, :])
            # L2
            dW1 += 2 * L2 * W1
            dW2 += 2 * L2 * W2
            for j in range(d):
                dEmb[j] += 2 * L2 * Emb[j]
            # Adam step
            step += 1
            grads = [dW1, db1, dW2, db2]
            for i, (p, gr) in enumerate(zip([W1, b1, W2, b2], grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** step)
                v_hat = v / (1 - 0.999 ** step)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
            for j in range(d):
                m, v = emb_state[0][j]
                gr = dEmb[j]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** step)
                v_hat = v / (1 - 0.999 ** step)
                Emb[j] -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return Emb, W1, b1, W2, b2

def accuracy(X, y, Emb, W1, b1, W2, b2, d, E, H, C):
    bs = X.shape[0]
    emb_out = np.empty((bs, d, E), np.float32)
    for j in range(d):
        emb_out[:, j, :] = Emb[j][X[:, j]]
    flat = emb_out.reshape(bs, d * E)
    hidden = np.maximum(flat @ W1 + b1, 0)
    logits = hidden @ W2 + b2
    preds = logits.argmax(axis=1)
    return (preds == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    # Tuning split: last 20%
    n_val = n // 5
    X_tr, y_tr = X[:-n_val], y[:-n_val]
    X_val, y_val = X[-n_val:], y[-n_val:]

    best_acc, best_cfg, best_params = -1, None, None
    base_rng = np.random.default_rng(SEED)
    for E, H in GRID:
        # Derive a seed for this config
        cfg_seed = int(base_rng.integers(2**32))
        rng = np.random.default_rng(cfg_seed)
        epochs = epochs_for(X_tr.shape[0])
        Emb, W1, b1, W2, b2 = train_model(X_tr, y_tr, d, C, E, H, rng, epochs)
        acc = accuracy(X_val, y_val, Emb, W1, b1, W2, b2, d, E, H, C)
        if acc > best_acc:
            best_acc, best_cfg = acc, (E, H)
            best_params = (Emb, W1, b1, W2, b2)
        print(f"E={E} H={H} val_acc={acc:.4f}", file=sys.stderr)

    E, H = best_cfg
    print(f"Best: E={E} H={H} val_acc={best_acc:.4f}", file=sys.stderr)
    # Retrain on full data
    rng = np.random.default_rng(int(base_rng.integers(2**32)))
    epochs = epochs_for(n)
    Emb, W1, b1, W2, b2 = train_model(X, y, d, C, E, H, rng, epochs)

    # Precompute T[j][b] = Emb[j][b] @ W1[j*E:(j+1)*E]
    T = np.empty((d, 256, H), np.float32)
    for j in range(d):
        W1_j = W1[j*E:(j+1)*E, :]
        T[j] = Emb[j] @ W1_j

    # Write weights
    magic = 0x454D4244  # "EMBD"
    out = struct.pack("<4i", magic, d, H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-embedmlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: 1 byte class index.
 * Weight format: magic 0x454D4244, int32 d, H, C; float32 T[d*256*H], b1[H], W2[H*C], b2[C]. */
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
    if (nw < 16 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    if (C < 1 || C > 256) { fprintf(stderr, "bad class count\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }
    const float *T = (const float *)(hd + 4);
    const float *b1 = T + (long)d * 256 * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * H);
    float *sc = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memcpy(acc, b1, H * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * H;
            for (int h = 0; h < H; h++) acc[h] += t[h];
        }
        for (int h = 0; h < H; h++) if (acc[h] < 0.0f) acc[h] = 0.0f;
        memcpy(sc, b2, C * sizeof(float));
        for (int h = 0; h < H; h++) {
            float ah = acc[h];
            const float *w = W2 + (long)h * C;
            for (int c = 0; c < C; c++) sc[c] += ah * w[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```