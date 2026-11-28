```hypothesis
Family: label-emb (entity embeddings + tiny MLP). All 14 features are already integer indices (0-255), so we learn a per-feature embedding table (dim=4) and feed the concatenated embeddings (56-d) into a one-hidden-layer ReLU MLP (16 units) then a 2-class linear head. This captures non-linear ordinal effects and reduces categorical cardinality without one-hot blowup. Expect ~0.86 dev accuracy at ~1500 Ir/item. If accuracy ≤0.852 or Ir/item >3000, the approach fails.
```

```python
"""label-emb: entity embeddings + 1-hidden-layer MLP for tabular byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x454D4231, d, emb_dim, H, C; float32 emb[d][256][emb_dim], W1[emb_dim*d][H], b1[H], W2[H][C], b2[C].
Hyperparameters (emb_dim, H, lr, L2, batch, epochs) chosen on last 20% tuning split, then retrained on all data."""
import json, math, struct, sys

import numpy as np

SEED = 1
# Hyperparameter grid for tuning split
GRID = [
    {"emb_dim": 2, "H": 0, "lr": 3e-3, "L2": 1e-4, "batch": 64, "epochs": 100},
    {"emb_dim": 4, "H": 0, "lr": 3e-3, "L2": 1e-4, "batch": 64, "epochs": 100},
    {"emb_dim": 4, "H": 16, "lr": 3e-3, "L2": 1e-4, "batch": 64, "epochs": 100},
    {"emb_dim": 8, "H": 16, "lr": 3e-3, "L2": 1e-4, "batch": 64, "epochs": 100},
]
SAMPLES = 600000
MAX_EPOCHS = 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_model(X, y, C, hp, rng):
    n, d = X.shape
    emb_dim, H = hp["emb_dim"], hp["H"]
    # Embeddings: d x 256 x emb_dim
    emb = (rng.uniform(-0.1, 0.1, (d, 256, emb_dim)) * math.sqrt(6.0 / 256)).astype(np.float32)
    inp_dim = d * emb_dim
    if H:
        W1 = (rng.uniform(-1, 1, (inp_dim, H)) * math.sqrt(6.0 / inp_dim)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [emb, W1, b1, W2, b2]
    else:
        W = np.zeros((inp_dim, C), np.float32)
        b = np.zeros(C, np.float32)
        params = [emb, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    lr, L2, batch = hp["lr"], hp["L2"], hp["batch"]
    max_epochs = min(hp["epochs"], math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = X[idx], y[idx]
            # Embedding lookup
            emb_seq = [emb[j][xb[:, j]] for j in range(d)]  # list of (batch, emb_dim)
            z = np.concatenate(emb_seq, axis=1)  # (batch, inp_dim)
            if H:
                a = z @ W1 + b1
                h = np.maximum(a, 0)
                logits = h @ W2 + b2
            else:
                logits = z @ W + b
            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            g = probs / len(yb)
            if H:
                gh = g @ W2.T * (a > 0)
                grad_W2 = h.T @ g + L2 * W2
                grad_b2 = g.sum(0)
                grad_W1 = z.T @ gh + L2 * W1
                grad_b1 = gh.sum(0)
                # Embedding gradients
                grad_emb = np.zeros_like(emb)
                off = 0
                for j in range(d):
                    grad_emb_j = gh[:, off:off + emb_dim].T @ np.eye(256)[xb[:, j]]
                    grad_emb[j] += grad_emb_j.T
                    off += emb_dim
                grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            else:
                grad_W = z.T @ g + L2 * W
                grad_b = g.sum(0)
                grad_emb = np.zeros_like(emb)
                off = 0
                for j in range(d):
                    grad_emb_j = g[:, off:off + emb_dim].T @ np.eye(256)[xb[:, j]]
                    grad_emb[j] += grad_emb_j.T
                    off += emb_dim
                grads = [grad_emb, grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr; v *= 0.999; v += 0.001 * gr * gr
                p -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return params


def eval_acc(X, y, C, params):
    d, emb_dim = params[0].shape[0], params[0].shape[2]
    H = len(params) == 5
    emb = params[0]
    n = X.shape[0]
    correct = 0
    for i in range(n):
        emb_seq = [emb[j][X[i, j]] for j in range(d)]
        z = np.concatenate(emb_seq)
        if H:
            W1, b1, W2, b2 = params[1], params[2], params[3], params[4]
            h = np.maximum(z @ W1 + b1, 0)
            logits = h @ W2 + b2
        else:
            W, b = params[1], params[2]
            logits = z @ W + b
        if logits.argmax() == y[i]:
            correct += 1
    return correct / n


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    n = X.shape[0]
    # Split: last 20% for tuning (by file order)
    split = int(0.8 * n)
    X_train, y_train = X[:split], y[:split]
    X_tune, y_tune = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_hp, best_acc = None, -1.0
    for hp in GRID:
        # Use a deterministic sub-seed per config
        sub_rng = np.random.default_rng(SEED + hash(str(hp)) % 2**32)
        params = train_model(X_train, y_train, C, hp, sub_rng)
        acc = eval_acc(X_tune, y_tune, C, params)
        if acc > best_acc:
            best_acc, best_hp = acc, hp

    # Retrain on all data with best hyperparameters
    final_rng = np.random.default_rng(SEED + 12345)
    final_params = train_model(X, y, C, best_hp, final_rng)

    # Serialize weights
    emb = final_params[0]  # (d, 256, emb_dim)
    emb_dim = emb.shape[2]
    H = best_hp["H"]
    out = struct.pack("<4i", 0x454D4231, d, emb_dim, H, C)
    out += emb.astype("<f4").tobytes()
    if H:
        W1, b1, W2, b2 = final_params[1], final_params[2], final_params[3], final_params[4]
        out += W1.astype("<f4").tobytes() + b1.astype("<f4").tobytes()
        out += W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    else:
        W, b = final_params[1], final_params[2]
        out += W.astype("<f4").tobytes() + b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d from weights). Output: 1 byte class index.
 * Weights format: magic 0x454D4231, int32 d, emb_dim, H, C;
 * float32 emb[d][256][emb_dim], then if H: W1[inp_dim][H], b1[H], W2[H][C], b2[C]
 * else: W[inp_dim][C], b[C]  (inp_dim = d*emb_dim). */
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
    if (nw < 20 || hd[0] != 0x454D4231) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], emb_dim = hd[2], H = hd[3], C = hd[4];
    int inp_dim = d * emb_dim;
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }
    size_t emb_sz = (size_t)d * 256 * emb_dim;
    size_t head_sz = 20;
    size_t expected;
    if (H) {
        expected = head_sz + 4 * (emb_sz + inp_dim * H + H + H * C + C);
    } else {
        expected = head_sz + 4 * (emb_sz + inp_dim * C + C);
    }
    if (nw != (long)expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + emb_sz;
    const float *b1 = W1 + inp_dim * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + H * C;
    const float *W = emb + emb_sz;
    const float *b = W + inp_dim * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *inp = malloc(sizeof(float) * inp_dim);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Embedding lookup
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((size_t)j * 256 + xi[j]) * emb_dim;
            memcpy(inp + j * emb_dim, e, sizeof(float) * emb_dim);
        }
        if (H) {
            // Hidden layer: ReLU(inp @ W1 + b1)
            for (int h = 0; h < H; h++) {
                float sum = b1[h];
                for (int k = 0; k < inp_dim; k++) sum += inp[k] * W1[(size_t)k * H + h];
                hidden[h] = sum > 0.0f ? sum : 0.0f;
            }
            // Output layer
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int h = 0; h < H; h++) sum += hidden[h] * W2[(size_t)h * C + c];
                logits[c] = sum;
            }
        } else {
            // Linear
            for (int c = 0; c < C; c++) {
                float sum = b[c];
                for (int k = 0; k < inp_dim; k++) sum += inp[k] * W[(size_t)k * C + c];
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