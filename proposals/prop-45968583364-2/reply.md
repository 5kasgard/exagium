```hypothesis
Family: label-embedding-mlp. Uses per-position character embeddings (dim E) concatenated into a 16*E vector, followed by an optional ReLU hidden layer (size H) and a linear output layer. Embeddings capture character similarities and reduce dimensionality vs one-hot; the hidden layer adds nonlinear interactions. Tuned on the last 20% of train. Expected to beat label-linear (0.3375) with fewer parameters and faster inference. If it fails to exceed linear, the task may not benefit from embeddings or nonlinearity at this scale.
```

```python
"""label-embedding-mlp: per-position embeddings + optional ReLU hidden layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Hyperparameters (E, H, lr, L2) chosen on last 20% of train, then retrained on all.
Weights format: magic 0x454D4244, int32 d,E,H,C; float32 embed[d][256][E];
if H>0: W1[d*E][H], b1[H], W2[H][C], b2[C] else: Wout[d*E][C], bout[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameter grid (tuned on validation split)
GRID = [
    {"E": 8,  "H": 0,   "lr": 3e-3, "l2": 1e-4},
    {"E": 16, "H": 0,   "lr": 3e-3, "l2": 1e-4},
    {"E": 32, "H": 0,   "lr": 3e-3, "l2": 1e-4},
    {"E": 8,  "H": 32,  "lr": 1e-3, "l2": 1e-4},
    {"E": 16, "H": 32,  "lr": 1e-3, "l2": 1e-4},
    {"E": 16, "H": 64,  "lr": 1e-3, "l2": 1e-4},
    {"E": 32, "H": 64,  "lr": 1e-3, "l2": 1e-5},
]
SEED = 1
SAMPLES = 600000
BATCH = 64
MAX_EPOCHS = 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(logits, y):
    return (logits.argmax(axis=1) == y).mean()


def train_model(X, y, Xv, yv, d, C, cfg, rng, epochs, batch):
    n, nv = X.shape[0], Xv.shape[0]
    E, H, lr, l2 = cfg["E"], cfg["H"], cfg["lr"], cfg["l2"]
    # Initialize parameters
    embed = (rng.normal(0, 1/math.sqrt(E), (d, 256, E))).astype(np.float32)
    if H:
        W1 = (rng.normal(0, math.sqrt(2.0/(d*E)), (d*E, H))).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.normal(0, math.sqrt(2.0/H), (H, C))).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [embed, W1, b1, W2, b2]
    else:
        Wout = (rng.normal(0, math.sqrt(2.0/(d*E)), (d*E, C))).astype(np.float32)
        bout = np.zeros(C, np.float32)
        params = [embed, Wout, bout]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    def forward(Xb):
        # Xb: (batch, d) uint8
        # embed lookup: (batch, d, E)
        emb = embed[Xb, np.arange(d)]  # (batch, d, E)
        flat = emb.reshape(len(Xb), -1)  # (batch, d*E)
        if H:
            h = np.maximum(flat @ W1 + b1, 0, dtype=np.float32)
            return h @ W2 + b2
        else:
            return flat @ Wout + bout
    def backward(Xb, yb, logits):
        batch_sz = len(yb)
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        probs[np.arange(batch_sz), yb] -= 1.0
        probs /= batch_sz
        grads = [np.zeros_like(p) for p in params]
        if H:
            h = np.maximum(flat @ W1 + b1, 0)
            grad_W2 = h.T @ probs
            grad_b2 = probs.sum(0)
            dh = probs @ W2.T * (h > 0)
            grad_W1 = flat.T @ dh
            grad_b1 = dh.sum(0)
            dflat = dh @ W1.T
            # embed gradient
            for j in range(d):
                idx = Xb[:, j]
                np.add.at(grads[0][j], idx, dflat[:, j*E:(j+1)*E])
            grads[1] = grad_W1 + l2 * W1
            grads[2] = grad_b1
            grads[3] = grad_W2 + l2 * W2
            grads[4] = grad_b2
        else:
            grad_Wout = flat.T @ probs
            grad_bout = probs.sum(0)
            dflat = probs @ Wout.T
            for j in range(d):
                idx = Xb[:, j]
                np.add.at(grads[0][j], idx, dflat[:, j*E:(j+1)*E])
            grads[1] = grad_Wout + l2 * Wout
            grads[2] = grad_bout
        return grads
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s+batch]
            Xb, yb = X[idx], y[idx]
            logits = forward(Xb)
            grads = backward(Xb, yb, logits)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9**t)
                v_hat = v / (1 - 0.999**t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    # Validation accuracy
    logits_v = forward(Xv)
    acc = accuracy(logits_v, yv)
    return acc, params


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # Load all training data
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_all = len(X_all)
    # Split: last 20% for tuning
    split = int(0.8 * n_all)
    X_tr, y_tr = X_all[:split], y_all[:split]
    X_val, y_val = X_all[split:], y_all[split:]
    rng = np.random.default_rng(SEED)
    # Determine epochs per config (sample budget)
    epochs_tr = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_tr)))
    best_acc, best_cfg, best_params = -1, None, None
    for cfg in GRID:
        acc, params = train_model(X_tr, y_tr, X_val, y_val, d, C, cfg, rng, epochs_tr, BATCH)
        if acc > best_acc:
            best_acc, best_cfg, best_params = acc, cfg, params
    # Retrain on full data
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n_all))
    _, final_params = train_model(X_all, y_all, X_all[:1], y_all[:1], d, C, best_cfg, rng, epochs_full, BATCH)
    # Pack weights
    embed = final_params[0]
    if best_cfg["H"]:
        W1, b1, W2, b2 = final_params[1], final_params[2], final_params[3], final_params[4]
        out = struct.pack("<5i", 0x454D4244, d, best_cfg["E"], best_cfg["H"], C)
        out += embed.astype("<f4").tobytes()
        out += W1.astype("<f4").tobytes() + b1.astype("<f4").tobytes()
        out += W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    else:
        Wout, bout = final_params[1], final_params[2]
        out = struct.pack("<5i", 0x454D4244, d, best_cfg["E"], 0, C)
        out += embed.astype("<f4").tobytes()
        out += Wout.astype("<f4").tobytes() + bout.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embedding-mlp inference: ./infer weights.bin in.rec N
 * Each input record is 16 bytes; output is one byte class index. */
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
    unsigned char h[4] = {k & 255, k>>8 & 255, k>>16 & 255, k>>24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H < 0) { fprintf(stderr, "bad header values\n"); return 3; }
    long embed_bytes = (long)d * 256 * E * 4;
    long expected;
    if (H) {
        expected = 20 + embed_bytes + ((long)d*E*H + H + (long)H*C + C) * 4;
    } else {
        expected = 20 + embed_bytes + ((long)d*E*C + C) * 4;
    }
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }
    const float *embed = (const float *)(hd + 5);
    const float *W1, *b1, *W2, *b2, *Wout, *bout;
    if (H) {
        W1 = embed + (long)d * 256 * E;
        b1 = W1 + (long)d * E * H;
        W2 = b1 + H;
        b2 = W2 + (long)H * C;
    } else {
        Wout = embed + (long)d * 256 * E;
        bout = Wout + (long)d * E * C;
    }
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = malloc(sizeof(float) * d * E);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *logits = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *src = embed + ((long)j * 256 + xi[j]) * E;
            memcpy(flat + j * E, src, E * sizeof(float));
        }
        if (H) {
            // hidden = ReLU(flat * W1 + b1)
            for (int h = 0; h < H; h++) {
                float sum = b1[h];
                for (int k = 0; k < d*E; k++) sum += flat[k] * W1[(long)k * H + h];
                hidden[h] = sum > 0 ? sum : 0;
            }
            // logits = hidden * W2 + b2
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
                logits[c] = sum;
            }
        } else {
            // logits = flat * Wout + bout
            for (int c = 0; c < C; c++) {
                float sum = bout[c];
                for (int k = 0; k < d*E; k++) sum += flat[k] * Wout[(long)k * C + c];
                logits[c] = sum;
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```