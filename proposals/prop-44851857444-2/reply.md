```hypothesis
Family: label-embmlp — position-specific embeddings (dim D) concatenated, then a small ReLU MLP (hidden size H) to 65 classes.
Why: Linear model (label-linear) uses 16×256 one-hot features (4096 dim) but cannot capture cross-position interactions. Embeddings compress each byte into a dense vector (e.g., D=16 → 256 dim), allowing the MLP to learn interactions with far fewer parameters (~16·256·D + (16·D)·H + H·65). This should improve accuracy over linear (33.75%) while keeping inference fast (only a few thousand FLOPs/item). Hyperparameters (D, H, lr) are tuned on the last 20% of train.in.
Failure signal: If dev accuracy ≤ 34%, the embedding+MLP structure does not capture enough sequential structure for this task.
```

```python
"""label-embmlp: position-specific embeddings + optional 1-hidden-layer MLP for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 42), single thread, stdlib+numpy only."""
import json, math, struct, sys
import numpy as np

# Fixed seed for all randomness
SEED = 42

# Hyperparameter grid (tuned on last 20% of train.in)
D_CHOICES = [8, 16, 32]
H_CHOICES = [0, 32, 64]          # 0 = linear on embeddings
LR_CHOICES = [1e-3, 3e-3]
BATCH = 64
MAX_EPOCHS_GRID = 30
MAX_EPOCHS_FINAL = 50
L2 = 1e-4
VAL_FRAC = 0.2

def read_records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def softmax_cross_entropy_grad(logits, y):
    """Return gradient of softmax cross-entropy w.r.t logits (already averaged over batch)."""
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    return probs / len(y)

def train_model(X, y, X_val, y_val, D, H, lr, epochs, rng):
    """Train and return (val_acc, params). params dict with keys: emb, W1, b1, W2, b2 (or W, b if H=0)."""
    n, d = X.shape
    C = 65
    # Initialize embeddings: [d, 256, D]
    emb = rng.uniform(-0.1, 0.1, (d, 256, D)).astype(np.float32)
    if H:
        W1 = rng.uniform(-1, 1, (d * D, H)).astype(np.float32) * math.sqrt(6.0 / (d * D))
        b1 = np.zeros(H, np.float32)
        W2 = rng.uniform(-1, 1, (H, C)).astype(np.float32) * math.sqrt(6.0 / H)
        b2 = np.zeros(C, np.float32)
        params = [emb, W1, b1, W2, b2]
    else:
        W = rng.uniform(-1, 1, (d * D, C)).astype(np.float32) * math.sqrt(6.0 / (d * D))
        b = np.zeros(C, np.float32)
        params = [emb, W, b]

    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(X_batch):
        # X_batch: (batch, d) uint8
        batch = X_batch.shape[0]
        # Gather embeddings: (batch, d, D)
        x = np.stack([emb[j, X_batch[:, j]] for j in range(d)], axis=1).astype(np.float32)
        x = x.reshape(batch, d * D)
        if H:
            h = x @ W1 + b1
            h = np.maximum(h, 0)
            out = h @ W2 + b2
        else:
            out = x @ W + b
        return out, x, (h if H else None)

    best_val_acc = -1.0
    best_params = None

    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            logits, x_flat, h = forward(xb)
            g = softmax_cross_entropy_grad(logits, yb)  # (B, C)

            if H:
                # g: (B, C)
                grad_W2 = h.T @ g + L2 * W2
                grad_b2 = g.sum(0)
                gh = g @ W2.T * (h > 0)
                grad_W1 = x_flat.T @ gh + L2 * W1
                grad_b1 = gh.sum(0)
                # embedding gradients: for each position j, accumulate gh contributions
                # x_flat = concat_j emb[j, xb[:, j]]
                # We need to scatter-add gh @ W1.T into embeddings
                grad_emb = np.zeros_like(emb)
                # gh @ W1.T shape (B, d*D)
                gx = gh @ W1.T
                for j in range(d):
                    gj = gx[:, j*D:(j+1)*D]
                    np.add.at(grad_emb[j], xb[:, j], gj)
                grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            else:
                grad_W = x_flat.T @ g + L2 * W
                grad_b = g.sum(0)
                grad_emb = np.zeros_like(emb)
                gx = g @ W.T  # (B, d*D)
                for j in range(d):
                    gj = gx[:, j*D:(j+1)*D]
                    np.add.at(grad_emb[j], xb[:, j], gj)
                grads = [grad_emb, grad_W, grad_b]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m = m * 0.9 + 0.1 * gr
                v = v * 0.999 + 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
                state[i] = (m, v)

        # Validation accuracy
        val_logits, _, _ = forward(X_val)
        val_pred = val_logits.argmax(axis=1)
        val_acc = (val_pred == y_val).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = [p.copy() for p in params]

    return best_val_acc, best_params

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(read_records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in read_records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    val_size = int(n * VAL_FRAC)
    train_size = n - val_size
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]

    rng = np.random.default_rng(SEED)
    best_overall = None
    best_cfg = None

    for D in D_CHOICES:
        for H in H_CHOICES:
            for lr in LR_CHOICES:
                cfg_rng = np.random.default_rng(SEED + hash((D, H, lr)) % 2**32)
                val_acc, params = train_model(X_train, y_train, X_val, y_val, D, H, lr, MAX_EPOCHS_GRID, cfg_rng)
                print(f"D={D} H={H} lr={lr} val_acc={val_acc:.4f}", file=sys.stderr)
                if best_overall is None or val_acc > best_overall:
                    best_overall = val_acc
                    best_cfg = (D, H, lr)
                    best_params = params

    print(f"Best config: D={best_cfg[0]} H={best_cfg[1]} lr={best_cfg[2]} val_acc={best_overall:.4f}", file=sys.stderr)

    # Retrain on full data with best config
    D, H, lr = best_cfg
    final_rng = np.random.default_rng(SEED + 12345)
    _, final_params = train_model(X, y, X_val, y_val, D, H, lr, MAX_EPOCHS_FINAL, final_rng)

    # Pack weights.bin
    # Header: magic, d, D, H, C (all int32 little-endian)
    magic = 0x454D4244  # 'EMBD'
    header = struct.pack("<5i", magic, d, D, H, C)
    emb = final_params[0]
    if H:
        W1, b1, W2, b2 = final_params[1], final_params[2], final_params[3], final_params[4]
        body = emb.astype("<f4").tobytes() + W1.astype("<f4").tobytes() + b1.astype("<f4").tobytes() + W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    else:
        W, b = final_params[1], final_params[2]
        body = emb.astype("<f4").tobytes() + W.astype("<f4").tobytes() + b.astype("<f4").tobytes()

    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes (preceded by 4-byte LE length). Output record: 1 byte class index. */
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
    uint32_t k = buf[*pos] | (buf[*pos+1]<<8) | (buf[*pos+2]<<16) | ((uint32_t)buf[*pos+3]<<24);
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k;
    const unsigned char *r = buf + *pos + 4;
    *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, (k>>8)&255, (k>>16)&255, (k>>24)&255};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0;
    long N = atol(argv[3]);
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || D <= 0 || H < 0) { fprintf(stderr, "bad header dims\n"); return 3; }

    size_t emb_count = (size_t)d * 256 * D;
    size_t expected = 20 + emb_count * sizeof(float);
    if (H) {
        expected += ((size_t)d * D * H + H + (size_t)H * C + C) * sizeof(float);
    } else {
        expected += ((size_t)d * D * C + C) * sizeof(float);
    }
    if (nw != (long)expected) { fprintf(stderr, "bad weights size: got %ld expect %zu\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + emb_count;
    const float *b1 = W1 + (size_t)d * D * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (size_t)H * C;
    const float *W = emb + emb_count;
    const float *b = W + (size_t)d * D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *x_flat = malloc(sizeof(float) * d * D);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *scores = malloc(sizeof(float) * C);
    long pos = 0;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((size_t)j * 256 + xi[j]) * D;
            memcpy(x_flat + j * D, e, D * sizeof(float));
        }

        if (H) {
            // hidden = ReLU(x_flat @ W1 + b1)
            for (int u = 0; u < H; u++) {
                float sum = b1[u];
                for (int v = 0; v < d * D; v++) sum += x_flat[v] * W1[(size_t)v * H + u];
                hidden[u] = sum > 0.0f ? sum : 0.0f;
            }
            // scores = hidden @ W2 + b2
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int u = 0; u < H; u++) sum += hidden[u] * W2[(size_t)u * C + c];
                scores[c] = sum;
            }
        } else {
            // scores = x_flat @ W + b
            for (int c = 0; c < C; c++) {
                float sum = b[c];
                for (int v = 0; v < d * D; v++) sum += x_flat[v] * W[(size_t)v * C + c];
                scores[c] = sum;
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```