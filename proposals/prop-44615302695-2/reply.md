```hypothesis
We propose a residual MLP with learned per-position byte embeddings (family "label-resmlp"). The model concatenates embeddings (size E) for each of the 16 context bytes, passes them through a hidden layer (size H, ReLU), and adds a residual linear connection from the concatenated embeddings to the output logits. This captures position‑specific interactions while keeping the parameter count below the linear baseline. Hyperparameters (E, H, residual) are chosen on the tuning split (last 20% of train.in). Success: dev accuracy > 0.3375 (label‑linear) with inference cost ≤ 2× label‑linear (~15k Ir/item). Failure: accuracy ≤ 0.3375 or cost > 15k Ir/item.
```

```python
"""label-resmlp: residual MLP with per-position byte embeddings.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH = 3e-3, 1e-4, 64
SAMPLES, MAX_EPOCHS = 600000, 100

# Hyperparameter grid for tuning split
GRID = [
    (8, 64, 0), (8, 64, 1), (8, 128, 0), (8, 128, 1),
    (16, 64, 0), (16, 64, 1), (16, 128, 0), (16, 128, 1),
]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def softmax_cross_entropy_grad(logits, y):
    # logits: (B, C), y: (B,)
    B = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits, dtype=np.float32)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(B), y] -= 1.0
    return probs / B

def forward(X, emb, W1, b1, W2, b2, W_res=None, b_res=None, H=None):
    # X: (B, 16) uint8
    B = X.shape[0]
    d = 16
    E = emb.shape[2]  # (d, 256, E)
    # Concatenate embeddings: (B, d*E)
    cat = np.zeros((B, d * E), dtype=np.float32)
    for j in range(d):
        cat[:, j*E:(j+1)*E] = emb[j, X[:, j]]
    # Residual logits
    if W_res is not None:
        logits_res = cat @ W_res + b_res
    else:
        logits_res = 0.0
    # Hidden layer
    hidden = cat @ W1 + b1
    hidden = np.maximum(hidden, 0, dtype=np.float32)  # ReLU
    logits_hidden = hidden @ W2 + b2
    return logits_res + logits_hidden, cat, hidden

def train_model(X, y, E, H, use_res, rng, epochs, batch_size):
    n, d = X.shape
    C = 65
    # Initialize embeddings: (d, 256, E)
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / E)).astype(np.float32)
    # First layer: (d*E, H)
    W1 = (rng.uniform(-1, 1, (d*E, H)) * math.sqrt(6.0 / (d*E))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # Output layer: (H, C)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    # Residual
    if use_res:
        W_res = (rng.uniform(-1, 1, (d*E, C)) * math.sqrt(6.0 / (d*E))).astype(np.float32)
        b_res = np.zeros(C, np.float32)
    else:
        W_res = b_res = None

    params = [emb, W1, b1, W2, b2]
    if use_res:
        params += [W_res, b_res]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s+batch_size]
            xb, yb = X[idx], y[idx]
            logits, cat, hidden = forward(xb, emb, W1, b1, W2, b2, W_res, b_res, H)
            grad_logits = softmax_cross_entropy_grad(logits, yb)

            # Gradients
            # Output layer
            gW2 = hidden.T @ grad_logits + L2 * W2
            gb2 = grad_logits.sum(axis=0)
            # Hidden layer
            ghidden = grad_logits @ W2.T
            ghidden[hidden == 0] = 0.0  # ReLU derivative
            gW1 = cat.T @ ghidden + L2 * W1
            gb1 = ghidden.sum(axis=0)
            # Residual
            if use_res:
                gW_res = cat.T @ grad_logits + L2 * W_res
                gb_res = grad_logits.sum(axis=0)
            # Embeddings
            gcat = ghidden @ W1.T
            if use_res:
                gcat += grad_logits @ W_res.T
            # gcat shape (B, d*E) -> split per position
            gemb = np.zeros_like(emb)
            for j in range(d):
                gj = gcat[:, j*E:(j+1)*E]
                # scatter-add: for each sample, add gj[i] to gemb[j, xb[i, j]]
                np.add.at(gemb[j], xb[:, j], gj)

            grads = [gemb, gW1, gb1, gW2, gb2]
            if use_res:
                grads += [gW_res, gb_res]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    return emb, W1, b1, W2, b2, W_res, b_res

def evaluate(X, y, emb, W1, b1, W2, b2, W_res, b_res):
    logits, _, _ = forward(X, emb, W1, b1, W2, b2, W_res, b_res)
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_all.shape[0]
    C = card["classes"]
    assert C == 65
    rng = np.random.default_rng(SEED)

    # Split: last 20% for tuning
    split = int(0.8 * n)
    X_train, y_train = X_all[:split], y_all[:split]
    X_val, y_val = X_all[split:], y_all[split:]

    # Determine epochs from sample budget
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))

    best_acc = -1.0
    best_cfg = None
    best_params = None

    for E, H, use_res in GRID:
        print(f"Trying E={E}, H={H}, res={use_res}", file=sys.stderr)
        emb, W1, b1, W2, b2, W_res, b_res = train_model(
            X_train, y_train, E, H, use_res, rng, epochs, BATCH
        )
        acc = evaluate(X_val, y_val, emb, W1, b1, W2, b2, W_res, b_res)
        print(f"  Val acc: {acc:.4f}", file=sys.stderr)
        if acc > best_acc:
            best_acc = acc
            best_cfg = (E, H, use_res)
            best_params = (emb, W1, b1, W2, b2, W_res, b_res)

    print(f"Best config: E={best_cfg[0]}, H={best_cfg[1]}, res={best_cfg[2]}, val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best config
    E, H, use_res = best_cfg
    emb, W1, b1, W2, b2, W_res, b_res = train_model(
        X_all, y_all, E, H, use_res, rng, epochs, BATCH
    )

    # Pack weights
    d = 16
    magic = 0x5245534D  # "MSER" little-endian
    has_res = 1 if use_res else 0
    header = struct.pack("<5i", magic, d, E, H, C, has_res)
    # Embeddings: d * 256 * E floats
    emb_bytes = emb.astype("<f4").tobytes()
    # W1: (d*E, H)
    W1_bytes = W1.astype("<f4").tobytes()
    b1_bytes = b1.astype("<f4").tobytes()
    # W2: (H, C)
    W2_bytes = W2.astype("<f4").tobytes()
    b2_bytes = b2.astype("<f4").tobytes()
    out = header + emb_bytes + W1_bytes + b1_bytes + W2_bytes + b2_bytes
    if use_res:
        W_res_bytes = W_res.astype("<f4").tobytes()
        b_res_bytes = b_res.astype("<f4").tobytes()
        out += W_res_bytes + b_res_bytes

    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-resmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
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
    if (nw < 24 || hd[0] != 0x5245534D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4], has_res = hd[5];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad header dims\n"); return 3; }

    size_t emb_count = (size_t)d * 256 * E;
    size_t W1_count = (size_t)d * E * H;
    size_t b1_count = H;
    size_t W2_count = (size_t)H * C;
    size_t b2_count = C;
    size_t W_res_count = has_res ? (size_t)d * E * C : 0;
    size_t b_res_count = has_res ? C : 0;
    size_t total_floats = emb_count + W1_count + b1_count + W2_count + b2_count + W_res_count + b_res_count;
    if (nw != 24 + (long)total_floats * 4) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *p = (const float *)(hd + 6);
    const float *emb = p; p += emb_count;
    const float *W1 = p; p += W1_count;
    const float *b1 = p; p += b1_count;
    const float *W2 = p; p += W2_count;
    const float *b2 = p; p += b2_count;
    const float *W_res = has_res ? p : NULL; if (has_res) p += W_res_count;
    const float *b_res = has_res ? p : NULL;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *cat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((size_t)j * 256 + xi[j]) * E;
            memcpy(cat + j * E, e, E * sizeof(float));
        }

        // Residual logits
        if (has_res) {
            for (int c = 0; c < C; c++) logits[c] = b_res[c];
            for (int idx = 0; idx < d * E; idx++) {
                float v = cat[idx];
                if (v == 0.0f) continue;
                const float *w = W_res + (size_t)idx * C;
                for (int c = 0; c < C; c++) logits[c] += v * w[c];
            }
        } else {
            memset(logits, 0, C * sizeof(float));
        }

        // Hidden layer: cat (d*E) @ W1 (d*E x H) + b1
        for (int h = 0; h < H; h++) hidden[h] = b1[h];
        for (int idx = 0; idx < d * E; idx++) {
            float v = cat[idx];
            if (v == 0.0f) continue;
            const float *w = W1 + (size_t)idx * H;
            for (int h = 0; h < H; h++) hidden[h] += v * w[h];
        }
        // ReLU
        for (int h = 0; h < H; h++) if (hidden[h] < 0.0f) hidden[h] = 0.0f;

        // Output layer: hidden (H) @ W2 (H x C) + b2
        for (int c = 0; c < C; c++) logits[c] += b2[c];
        for (int h = 0; h < H; h++) {
            float v = hidden[h];
            if (v == 0.0f) continue;
            const float *w = W2 + (size_t)h * C;
            for (int c = 0; c < C; c++) logits[c] += v * w[c];
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(cat); free(hidden); free(logits);
    return 0;
}
```