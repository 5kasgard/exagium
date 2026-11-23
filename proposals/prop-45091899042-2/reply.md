```hypothesis
Family: label-embed. Learns a low-rank factorization of the linear model by embedding each of the 16 context positions into a D-dimensional space (D=8), then a single linear layer to 65 classes. This captures character similarities and position-specific patterns with far fewer parameters (≈41k vs 266k) and less inference work (≈8k multiply-adds vs 266k). It should beat label-linear on accuracy by sharing statistical strength across characters. Failure: if dev accuracy ≤ 0.3375 or inference cost exceeds label-linear.
```

```python
"""label-embed: positional embeddings + linear output for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Tunes embedding dimension D and learning rate on last 20% of train.in, then trains on all data.
Weights format: magic 0x454D4245, int32 d, D, C; float32 embed[d][256][D], W_out[d*D][C], b_out[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
# Tuning grid
D_CANDIDATES = [4, 8, 16]
LR_CANDIDATES = [1e-2, 3e-3, 1e-3]
EPOCHS_TUNE = 15
EPOCHS_FULL = 30
BATCH = 256
L2 = 1e-5
VAL_FRAC = 0.2  # last 20%


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def adam_step(params, grads, states, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    for p, g, (m, v) in zip(params, grads, states):
        m[:] = beta1 * m + (1 - beta1) * g
        v[:] = beta2 * v + (1 - beta2) * g * g
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        p -= lr * m_hat / (np.sqrt(v_hat) + eps)


def train_eval(X_tr, y_tr, X_val, y_val, d, D, C, lr, epochs, rng):
    # embed: (d, 256, D), W_out: (d*D, C), b_out: (C,)
    embed = (rng.uniform(-1, 1, (d, 256, D)) * math.sqrt(6.0 / (256 + D))).astype(np.float32)
    W_out = (rng.uniform(-1, 1, (d * D, C)) * math.sqrt(6.0 / (d * D + C))).astype(np.float32)
    b_out = np.zeros(C, np.float32)
    params = [embed, W_out, b_out]
    states = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X_tr.shape[0]
    best_val = -1.0
    best_params = None
    t = 0
    for ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X_tr[idx], y_tr[idx]
            bs = len(yb)
            # forward
            # embeddings: (bs, d, D)
            emb = embed[xb]  # (bs, d, D)
            flat = emb.reshape(bs, d * D)  # (bs, d*D)
            logits = flat @ W_out + b_out  # (bs, C)
            # softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            probs /= bs
            # gradients
            grad_W_out = flat.T @ probs + L2 * W_out
            grad_b_out = probs.sum(axis=0)
            grad_flat = probs @ W_out.T  # (bs, d*D)
            grad_emb = grad_flat.reshape(bs, d, D)
            grad_embed = np.zeros_like(embed)
            np.add.at(grad_embed, (xb, np.arange(256)[None, :]? ), 0)  # placeholder
            # efficient scatter-add for embed gradients
            for j in range(d):
                np.add.at(grad_embed[j], xb[:, j], grad_emb[:, j, :])
            grad_embed += L2 * embed
            grads = [grad_embed, grad_W_out, grad_b_out]
            t += 1
            adam_step(params, grads, states, t, lr)
        # validation accuracy
        emb_val = embed[X_val].reshape(-1, d * D)
        logits_val = emb_val @ W_out + b_out
        pred = logits_val.argmax(axis=1)
        val_acc = (pred == y_val).mean()
        if val_acc > best_val:
            best_val = val_acc
            best_params = [p.copy() for p in params]
    return best_val, best_params


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    split = int(n * (1 - VAL_FRAC))
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    rng = np.random.default_rng(SEED)
    best_overall = -1.0
    best_cfg = None
    best_params = None

    for D in D_CANDIDATES:
        for lr in LR_CANDIDATES:
            val_acc, params = train_eval(X_tr, y_tr, X_val, y_val, d, D, C, lr, EPOCHS_TUNE, rng)
            if val_acc > best_overall:
                best_overall = val_acc
                best_cfg = (D, lr)
                best_params = params
    print(f"Tuned: D={best_cfg[0]}, lr={best_cfg[1]}, val_acc={best_overall:.4f}", file=sys.stderr)

    # Retrain on full data
    D, lr = best_cfg
    embed, W_out, b_out = best_params
    params = [embed, W_out, b_out]
    states = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for ep in range(EPOCHS_FULL):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = len(yb)
            emb = embed[xb]
            flat = emb.reshape(bs, d * D)
            logits = flat @ W_out + b_out
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            probs /= bs
            grad_W_out = flat.T @ probs + L2 * W_out
            grad_b_out = probs.sum(axis=0)
            grad_flat = probs @ W_out.T
            grad_emb = grad_flat.reshape(bs, d, D)
            grad_embed = np.zeros_like(embed)
            for j in range(d):
                np.add.at(grad_embed[j], xb[:, j], grad_emb[:, j, :])
            grad_embed += L2 * embed
            grads = [grad_embed, grad_W_out, grad_b_out]
            t += 1
            adam_step(params, grads, states, t, lr)

    embed, W_out, b_out = params
    # Write weights
    magic = 0x454D4245  # 'EMBE'
    out = struct.pack("<4i", magic, d, D, C)
    out += embed.astype("<f4").tobytes()
    out += W_out.astype("<f4").tobytes()
    out += b_out.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: magic 0x454D4245, int32 d, D, C; float embed[d][256][D], W_out[d*D][C], b_out[C]. */
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
    if (nw < 16 || hd[0] != 0x454D4245) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3];
    if (d != 16 || C != 65 || D <= 0 || D > 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * D + (long)d * D * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *embed = (const float *)(hd + 4);
    const float *W_out = embed + (long)d * 256 * D;
    const float *b_out = W_out + (long)d * D * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = malloc(sizeof(float) * d * D);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        /* gather embeddings */
        for (int j = 0; j < d; j++) {
            const float *e = embed + ((long)j * 256 + xi[j]) * D;
            memcpy(flat + j * D, e, D * sizeof(float));
        }
        /* logits = flat @ W_out + b_out */
        for (int c = 0; c < C; c++) logits[c] = b_out[c];
        for (int idx = 0; idx < d * D; idx++) {
            float f = flat[idx];
            if (f == 0.0f) continue;
            const float *w = W_out + (long)idx * C;
            for (int c = 0; c < C; c++) logits[c] += f * w[c];
        }
        /* argmax */
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```