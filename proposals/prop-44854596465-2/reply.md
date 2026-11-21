```hypothesis
Family: label-embmlp (per-position byte embeddings + small MLP).  
Why: The linear model uses one-hot vectors (4096 dims) but cannot capture interactions between context positions. Learned embeddings (e.g., 8 dims/position → 128 dims) compress the input and a hidden ReLU layer (e.g., 64 units) can model pairwise interactions efficiently. This should improve accuracy over label-linear while keeping inference fast (fewer multiply-adds than one-hot linear).  
Failure: If dev accuracy ≤ 0.3375 (label-linear) or CPU cost per item exceeds label-linear by >2×, the family does not meet its goal.
```

```python
"""label-embmlp: per-position byte embeddings + optional hidden ReLU layer.
train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Hyperparameters chosen on last 20% of train.in (tuning split), then retrained on all train.
Exported weights format (little-endian):
  magic (int32) = 0x454D424D ('EMBM')
  d (int32) = 16, E (int32) = embedding dim, H (int32) = hidden size (0 = linear), C (int32) = 65
  embeddings [d][256][E] (float32)
  W1 [d*E][H or C] (float32), b1 [H or C] (float32)
  if H>0: W2 [H][C] (float32), b2 [C] (float32)
"""
import json, math, struct, sys
import numpy as np

SEED = 1
# Search grid
EMB_DIMS = [4, 8]
HIDDEN_SIZES = [0, 32, 64]   # 0 = linear
LR = 3e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS_SEARCH = 15
MAX_EPOCHS_FINAL = 30
SAMPLES_PER_EPOCH = 600000  # not used directly, we use epochs

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def softmax_cross_entropy_grad(logits, y):
    """logits: (B, C), y: (B,) int. Returns gradient (B, C) and loss."""
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits, dtype=np.float32)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    B = logits.shape[0]
    probs[np.arange(B), y] -= 1.0
    grad = probs / B
    loss = -np.log(probs[np.arange(B), y] + 1e-12).mean()
    return grad, loss

def train_model(X_tr, y_tr, X_val, y_val, d, E, H, C, rng, epochs, lr, l2, batch):
    n_tr = X_tr.shape[0]
    # Initialize parameters
    emb = rng.normal(0, 0.01, size=(d, 256, E)).astype(np.float32)
    in_dim = d * E
    if H > 0:
        W1 = rng.normal(0, math.sqrt(2.0 / in_dim), size=(in_dim, H)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = rng.normal(0, math.sqrt(2.0 / H), size=(H, C)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [emb, W1, b1, W2, b2]
    else:
        W1 = rng.normal(0, math.sqrt(2.0 / in_dim), size=(in_dim, C)).astype(np.float32)
        b1 = np.zeros(C, np.float32)
        params = [emb, W1, b1]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    best_val_acc = -1.0
    best_params = None
    for epoch in range(epochs):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, batch):
            idx = order[s:s+batch]
            xb = X_tr[idx]
            yb = y_tr[idx]
            # Forward
            # Embeddings: (batch, d, E)
            emb_seq = emb[xb]  # (batch, d, E)
            flat = emb_seq.reshape(len(yb), -1)  # (batch, d*E)
            if H > 0:
                hidden = flat @ W1 + b1
                hidden_relu = np.maximum(hidden, 0, dtype=np.float32)
                logits = hidden_relu @ W2 + b2
            else:
                logits = flat @ W1 + b1
            # Loss & grad
            grad_logits, _ = softmax_cross_entropy_grad(logits, yb)
            # Backprop
            if H > 0:
                grad_W2 = hidden_relu.T @ grad_logits + l2 * W2
                grad_b2 = grad_logits.sum(axis=0)
                grad_hidden = grad_logits @ W2.T
                grad_hidden[hidden <= 0] = 0.0
                grad_W1 = flat.T @ grad_hidden + l2 * W1
                grad_b1 = grad_hidden.sum(axis=0)
                # Embedding gradients
                grad_flat = grad_hidden @ W1.T  # (batch, d*E)
                grad_emb_seq = grad_flat.reshape(len(yb), d, E)
                grad_emb = np.zeros_like(emb)
                np.add.at(grad_emb, (xb[:,:,None], np.arange(256)[None,None,:]), grad_emb_seq[:,:,None,:])  # wrong shape
                # Simpler: loop over positions (d=16 small)
                grad_emb = np.zeros_like(emb)
                for j in range(d):
                    for b in range(len(yb)):
                        byte_val = xb[b, j]
                        grad_emb[j, byte_val] += grad_emb_seq[b, j]
            else:
                grad_W1 = flat.T @ grad_logits + l2 * W1
                grad_b1 = grad_logits.sum(axis=0)
                grad_flat = grad_logits @ W1.T
                grad_emb_seq = grad_flat.reshape(len(yb), d, E)
                grad_emb = np.zeros_like(emb)
                for j in range(d):
                    for b in range(len(yb)):
                        byte_val = xb[b, j]
                        grad_emb[j, byte_val] += grad_emb_seq[b, j]
            grads = [grad_emb, grad_W1, grad_b1]
            if H > 0:
                grads += [grad_W2, grad_b2]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        # Validation accuracy
        if X_val is not None:
            val_logits = forward_batch(X_val, params, d, E, H)
            val_pred = val_logits.argmax(axis=1)
            val_acc = (val_pred == y_val).mean()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = [p.copy() for p in params]
    if best_params is not None:
        params = best_params
    return params

def forward_batch(X, params, d, E, H):
    n = X.shape[0]
    emb = params[0]
    emb_seq = emb[X]  # (n, d, E)
    flat = emb_seq.reshape(n, -1)
    if H > 0:
        W1, b1, W2, b2 = params[1], params[2], params[3], params[4]
        hidden = flat @ W1 + b1
        hidden_relu = np.maximum(hidden, 0)
        logits = hidden_relu @ W2 + b2
    else:
        W1, b1 = params[1], params[2]
        logits = flat @ W1 + b1
    return logits

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # Load all data
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_all.shape[0]
    # Split: last 20% for tuning
    split = int(0.8 * n)
    X_tr, X_val = X_all[:split], X_all[split:]
    y_tr, y_val = y_all[:split], y_all[split:]
    rng = np.random.default_rng(SEED)
    best_config = None
    best_val_acc = -1.0
    best_params = None
    print("Searching hyperparameters...", file=sys.stderr)
    for E in EMB_DIMS:
        for H in HIDDEN_SIZES:
            print(f"  Trying E={E}, H={H}", file=sys.stderr)
            params = train_model(X_tr, y_tr, X_val, y_val, d, E, H, C, rng, MAX_EPOCHS_SEARCH, LR, L2, BATCH)
            # Evaluate on validation
            val_logits = forward_batch(X_val, params, d, E, H)
            val_pred = val_logits.argmax(axis=1)
            val_acc = (val_pred == y_val).mean()
            print(f"    val_acc={val_acc:.4f}", file=sys.stderr)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_config = (E, H)
                best_params = params
    E, H = best_config
    print(f"Best config: E={E}, H={H}, val_acc={best_val_acc:.4f}", file=sys.stderr)
    # Retrain on full data
    print("Retraining on full data...", file=sys.stderr)
    final_params = train_model(X_all, y_all, None, None, d, E, H, C, rng, MAX_EPOCHS_FINAL, LR, L2, BATCH)
    # Pack weights
    emb = final_params[0]
    W1 = final_params[1]
    b1 = final_params[2]
    if H > 0:
        W2 = final_params[3]
        b2 = final_params[4]
    else:
        W2 = None
        b2 = None
    # Compute expected size
    emb_size = d * 256 * E
    if H > 0:
        W1_size = d * E * H
        W2_size = H * C
        total_floats = emb_size + W1_size + H + W2_size + C
    else:
        W1_size = d * E * C
        total_floats = emb_size + W1_size + C
    # Write
    magic = 0x454D424D  # 'EMBM'
    header = struct.pack("<5i", magic, d, E, H, C)
    out = header
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if H > 0:
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)
    print(f"Written {len(out)} bytes", file=sys.stderr)

if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format (little-endian):
 *   int32 magic=0x454D424D, d, E, H, C
 *   float32 emb[d][256][E]
 *   float32 W1[d*E][H or C], b1[H or C]
 *   if H>0: float32 W2[H][C], b2[C]
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
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H < 0 || C <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected_floats = (long)d * 256 * E + (long)d * E * (H ? H : C) + (H ? H : C);
    if (H) expected_floats += (long)H * C + C;
    if (nw != 20 + 4 * expected_floats) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, 20 + 4 * expected_floats); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * (H ? H : C);
    const float *W2 = H ? b1 + (H ? H : C) : NULL;
    const float *b2 = H ? W2 + (long)H * C : NULL;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = malloc(sizeof(float) * d * E);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Embedding lookup and concatenation
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(flat + j * E, e, E * sizeof(float));
        }
        if (H) {
            // hidden = flat @ W1 + b1, ReLU
            for (int h = 0; h < H; h++) {
                float sum = b1[h];
                for (int f = 0; f < d * E; f++) sum += flat[f] * W1[(long)f * H + h];
                hidden[h] = sum > 0.0f ? sum : 0.0f;
            }
            // logits = hidden @ W2 + b2
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
                logits[c] = sum;
            }
        } else {
            // logits = flat @ W1 + b1
            for (int c = 0; c < C; c++) {
                float sum = b1[c];
                for (int f = 0; f < d * E; f++) sum += flat[f] * W1[(long)f * C + c];
                logits[c] = sum;
            }
        }
        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```