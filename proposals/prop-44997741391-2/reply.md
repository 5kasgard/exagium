```hypothesis
Family: label-embed (per-position byte embeddings summed, then a tiny MLP).
Why: The 16-byte context has strong positional patterns; one-hot linear (label-linear) uses 4096×65 weights but ignores interactions.
Summing learned 16×256×E embeddings (E=16) captures per-position statistics with 65k params, then a 16-unit ReLU layer adds non-linearity.
This should beat label-linear's 0.3375 accuracy at similar CPU cost (embedding lookup + small matmul).
Failure: If dev accuracy ≤ 0.3375, the family is not effective for this task.
```

```python
"""label-embed: per-position byte embeddings summed, then optional ReLU layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# hyper-parameter grid searched on the tuning split (last 20% of train.in)
EMBED_SIZES = [8, 16, 32]
HIDDEN_SIZES = [0, 16, 32]          # 0 = linear on summed embeddings
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(logits, y):
    return (logits.argmax(axis=1) == y).mean()


def train_eval(X_tr, y_tr, X_val, y_val, d, C, E, H, rng):
    """Train on X_tr, return validation accuracy."""
    # embeddings: list of d arrays (256, E)
    emb = [(rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / (256 + E))).astype(np.float32)
           for _ in range(d)]
    if H:
        W1 = (rng.uniform(-1, 1, (E, H)) * math.sqrt(6.0 / E)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = emb + [W1, b1, W2, b2]
    else:
        W = (rng.uniform(-1, 1, (E, C)) * math.sqrt(6.0 / E)).astype(np.float32)
        b = np.zeros(C, np.float32)
        params = emb + [W, b]

    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    n_tr = X_tr.shape[0]
    steps = min(MAX_EPOCHS, math.ceil(SAMPLES / n_tr))
    t = 0

    def forward(X, emb, H, W1=None, b1=None, W2=None, b2=None, W=None, b=None):
        # X: (n, d) uint8
        n = X.shape[0]
        # sum embeddings
        summed = np.zeros((n, E), np.float32)
        for j in range(d):
            summed += emb[j][X[:, j]]
        if H:
            hidden = np.maximum(summed @ W1 + b1, 0)
            return hidden @ W2 + b2
        else:
            return summed @ W + b

    def softmax_grad(logits, y):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(len(y)), y] -= 1.0
        return (probs / len(y)).astype(np.float32)

    for _ in range(steps):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_tr[idx], y_tr[idx]
            # forward
            if H:
                summed = np.zeros((len(idx), E), np.float32)
                for j in range(d):
                    summed += emb[j][Xb[:, j]]
                hidden_pre = summed @ W1 + b1
                hidden = np.maximum(hidden_pre, 0)
                logits = hidden @ W2 + b2
            else:
                summed = np.zeros((len(idx), E), np.float32)
                for j in range(d):
                    summed += emb[j][Xb[:, j]]
                logits = summed @ W + b
            # gradient
            g = softmax_grad(logits, yb)
            if H:
                grad_W2 = hidden.T @ g + L2 * W2
                grad_b2 = g.sum(axis=0)
                gh = g @ W2.T * (hidden_pre > 0)
                grad_W1 = summed.T @ gh + L2 * W1
                grad_b1 = gh.sum(axis=0)
                g_summed = gh @ W1.T
                grads_emb = []
                for j in range(d):
                    grad = np.zeros((256, E), np.float32)
                    np.add.at(grad, Xb[:, j], g_summed)
                    grad += L2 * emb[j]
                    grads_emb.append(grad)
                grads = grads_emb + [grad_W1, grad_b1, grad_W2, grad_b2]
            else:
                grad_W = summed.T @ g + L2 * W
                grad_b = g.sum(axis=0)
                g_summed = g @ W.T
                grads_emb = []
                for j in range(d):
                    grad = np.zeros((256, E), np.float32)
                    np.add.at(grad, Xb[:, j], g_summed)
                    grad += L2 * emb[j]
                    grads_emb.append(grad)
                grads = grads_emb + [grad_W, grad_b]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # validation accuracy
    val_logits = forward(X_val, emb, H, W1=W1, b1=b1, W2=W2, b2=b2, W=W, b=b)
    return accuracy(val_logits, y_val), params


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = X.shape[1], card["classes"]
    n = X.shape[0]
    split = int(n * 0.8)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    rng = np.random.default_rng(SEED)
    best_acc, best_params, best_E, best_H = -1, None, None, None

    for E in EMBED_SIZES:
        for H in HIDDEN_SIZES:
            rng_split = np.random.default_rng(SEED + E * 100 + H)
            acc, params = train_eval(X_tr, y_tr, X_val, y_val, d, C, E, H, rng_split)
            print(f"E={E} H={H} val_acc={acc:.4f}", file=sys.stderr)
            if acc > best_acc:
                best_acc, best_params, best_E, best_H = acc, params, E, H

    print(f"Best: E={best_E} H={best_H} acc={best_acc:.4f}", file=sys.stderr)
    # retrain on full data with best config
    rng_full = np.random.default_rng(SEED + best_E * 100 + best_H)
    _, final_params = train_eval(X, y, X[:1], y[:1], d, C, best_E, best_H, rng_full)  # dummy val
    emb = final_params[:d]
    if best_H:
        W1, b1, W2, b2 = final_params[d:]
    else:
        W, b = final_params[d:]

    # write weights.bin
    # magic 0x454D4244 ("EMBD"), d, E, H, C
    out = struct.pack("<5i", 0x454D4244, d, best_E, best_H, C)
    for e in emb:
        out += e.astype("<f4").tobytes()
    if best_H:
        out += W1.astype("<f4").tobytes() + b1.astype("<f4").tobytes()
        out += W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    else:
        out += W.astype("<f4").tobytes() + b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: one byte class index.
 * Weights format: magic 0x454D4244, int32 d, E, H, C;
 *   d tables of 256×E float32 (embeddings);
 *   if H>0: float32 W1[E][H], b1[H], W2[H][C], b2[C]
 *   else:   float32 W[E][C], b[C] */
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d < 1 || E < 1 || H < 0) { fprintf(stderr, "bad weights dims\n"); return 3; }

    long expected = 20L + 4L * d * 256 * E;
    if (H) expected += 4L * (E * H + H + H * C + C);
    else   expected += 4L * (E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1, *b1, *W2, *b2, *W, *b;
    if (H) {
        W1 = emb + d * 256 * E;
        b1 = W1 + E * H;
        W2 = b1 + H;
        b2 = W2 + H * C;
    } else {
        W = emb + d * 256 * E;
        b = W + E * C;
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *summed = malloc(sizeof(float) * E);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memset(summed, 0, sizeof(float) * E);
        for (int j = 0; j < d; j++) {
            const float *row = emb + (j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) summed[e] += row[e];
        }

        if (H) {
            for (int h = 0; h < H; h++) {
                float v = b1[h];
                for (int e = 0; e < E; e++) v += summed[e] * W1[e * H + h];
                hidden[h] = v > 0.0f ? v : 0.0f;
            }
            for (int c = 0; c < C; c++) {
                float v = b2[c];
                for (int h = 0; h < H; h++) v += hidden[h] * W2[h * C + c];
                logits[c] = v;
            }
        } else {
            for (int c = 0; c < C; c++) {
                float v = b[c];
                for (int e = 0; e < E; e++) v += summed[e] * W[e * C + c];
                logits[c] = v;
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```