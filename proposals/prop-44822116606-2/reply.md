```hypothesis
Family: label-embmlp. Uses per-position byte embeddings (size E) summed into a single vector, then one ReLU hidden layer (size H) and a linear output. Embeddings drastically reduce parameters vs one-hot linear (16*256*E vs 16*256*C) while allowing the hidden layer to model interactions. Expect higher accuracy than label-linear at similar or lower inference cost. Failure if dev accuracy ≤ 0.3375 or instructions/item exceed label-linear by >20%.
```

```python
"""label-embmlp: Embedding + MLP for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Grid-searches (E,H) on last 20% of train, then retrains on all data.
Exports weights: magic 0x454D424D, d, E, H, C; float32 emb[d*256*E], b1[H], W1[E*H], b2[C], W2[H*C] (if H>0)
or emb, bout[C], Wout[E*C] (if H=0)."""
import json, math, struct, sys

import numpy as np

# Grid search space (chosen after tuning on the last 20% of train.in)
GRID = [(16, 64), (16, 128), (32, 64), (32, 128)]
SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
TUNE_EPOCHS = 5  # quick epochs for grid search
TUNE_FRAC = 0.2  # last 20%


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def split_data(X, y, frac):
    n = X.shape[0]
    cut = int(n * (1 - frac))
    return X[:cut], y[:cut], X[cut:], y[cut:]


def train_model(Xtr, ytr, Xte, yte, d, C, E, H, rng, epochs):
    n, _ = Xtr.shape
    # Embeddings: [d, 256, E]
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / E)).astype(np.float32)
    if H:
        W1 = (rng.uniform(-1, 1, (E, H)) * math.sqrt(6.0 / E)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [emb, W1, b1, W2, b2]
    else:
        Wout = (rng.uniform(-1, 1, (E, C)) * math.sqrt(6.0 / E)).astype(np.float32)
        bout = np.zeros(C, np.float32)
        params = [emb, Wout, bout]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = Xtr[idx], ytr[idx]
            # forward
            # sum embeddings: [batch, E]
            h = np.zeros((len(yb), E), np.float32)
            for j in range(d):
                h += emb[j, xb[:, j]]
            if H:
                a = h @ W1 + b1
                h_relu = np.maximum(a, 0)
                logits = h_relu @ W2 + b2
            else:
                logits = h @ Wout + bout
            # softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad_out = (probs / len(yb)).astype(np.float32)
            # backward
            if H:
                grad_W2 = h_relu.T @ grad_out + L2 * W2
                grad_b2 = grad_out.sum(0)
                grad_h_relu = grad_out @ W2.T
                grad_a = grad_h_relu * (a > 0)
                grad_W1 = h.T @ grad_a + L2 * W1
                grad_b1 = grad_a.sum(0)
                grad_h = grad_a @ W1.T
                # embedding gradients
                grad_emb = np.zeros_like(emb)
                for j in range(d):
                    # for each position, scatter-add grad_h into grad_emb[j, xb[:,j]]
                    np.add.at(grad_emb[j], xb[:, j], grad_h)
                grad_emb += L2 * emb
                grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            else:
                grad_Wout = h.T @ grad_out + L2 * Wout
                grad_bout = grad_out.sum(0)
                grad_h = grad_out @ Wout.T
                grad_emb = np.zeros_like(emb)
                for j in range(d):
                    np.add.at(grad_emb[j], xb[:, j], grad_h)
                grad_emb += L2 * emb
                grads = [grad_emb, grad_Wout, grad_bout]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    # evaluate on tune
    h = np.zeros((Xte.shape[0], E), np.float32)
    for j in range(d):
        h += emb[j, Xte[:, j]]
    if H:
        h_relu = np.maximum(h @ W1 + b1, 0)
        logits = h_relu @ W2 + b2
    else:
        logits = h @ Wout + bout
    pred = logits.argmax(axis=1)
    acc = (pred == yte).mean()
    return acc, params


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = card["input"]["bytes"], card["classes"]
    rng = np.random.default_rng(SEED)
    Xtr, ytr, Xte, yte = split_data(X, y, TUNE_FRAC)
    best_acc, best_params, best_cfg = -1, None, None
    for E, H in GRID:
        acc, params = train_model(Xtr, ytr, Xte, yte, d, C, E, H, rng, TUNE_EPOCHS)
        if acc > best_acc:
            best_acc, best_params, best_cfg = acc, params, (E, H)
    E, H = best_cfg
    # retrain on all data
    _, best_params = train_model(X, y, X[:1], y[:1], d, C, E, H, rng, min(MAX_EPOCHS, math.ceil(SAMPLES / len(X))))
    # pack weights
    emb = best_params[0]
    if H:
        W1, b1, W2, b2 = best_params[1], best_params[2], best_params[3], best_params[4]
        W = H
        out = struct.pack("<5i", 0x454D424D, d, E, H, C)
        out += emb.astype("<f4").tobytes()
        out += b1.astype("<f4").tobytes()
        out += W1.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()
        out += W2.astype("<f4").tobytes()
    else:
        Wout, bout = best_params[1], best_params[2]
        W = E
        out = struct.pack("<5i", 0x454D424D, d, E, 0, C)
        out += emb.astype("<f4").tobytes()
        out += bout.astype("<f4").tobytes()
        out += Wout.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embmlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: one byte class index.
 * Weights format: magic 0x454D424D, int32 d, E, H, C; float32 emb[d*256*E];
 * if H>0: float32 b1[H], W1[E*H], b2[C], W2[H*C]
 * else:   float32 bout[C], Wout[E*C] */
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
    if (nw < 20 || hd[0] != 0x454D424D) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights header\n"); return 3; }
    long expected = 20;
    expected += 4L * d * 256 * E;
    if (H) expected += 4L * (H + E * H + C + H * C);
    else   expected += 4L * (C + E * C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *b1 = emb + d * 256 * E;
    const float *W1 = b1 + (H ? H : 0);
    const float *b2 = W1 + (H ? E * H : 0);
    const float *W2 = b2 + (H ? C : 0);
    const float *bout = b1;
    const float *Wout = bout + (H ? 0 : C);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * E);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *logits = malloc(sizeof(float) * C);
    unsigned char out_byte;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // sum embeddings
        for (int e = 0; e < E; e++) h[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *eptr = emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) h[e] += eptr[e];
        }
        if (H) {
            // hidden = ReLU(h @ W1 + b1)
            for (int u = 0; u < H; u++) {
                float sum = b1[u];
                for (int e = 0; e < E; e++) sum += h[e] * W1[(long)e * H + u];
                hidden[u] = sum > 0.0f ? sum : 0.0f;
            }
            // logits = hidden @ W2 + b2
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int u = 0; u < H; u++) sum += hidden[u] * W2[(long)u * C + c];
                logits[c] = sum;
            }
        } else {
            // logits = h @ Wout + bout
            for (int c = 0; c < C; c++) {
                float sum = bout[c];
                for (int e = 0; e < E; e++) sum += h[e] * Wout[(long)e * C + c];
                logits[c] = sum;
            }
        }
        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```