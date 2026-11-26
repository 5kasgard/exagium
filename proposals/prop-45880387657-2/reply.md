```hypothesis
Family: label-emb (per-position byte embeddings + linear output).
Why: The linear model uses a full 256×65 weight matrix per position (16×256×65 params), which overfits on 24k samples. A low-rank factorization via embeddings (16×256×E + E×65) shares statistical strength across characters and positions, acting as a regularized linear model. With E tuned on the 20% tuning split (e.g., 16, 32, 64, 128), it should match or beat label-linear accuracy at lower CPU cost (only E×65 mul-adds per item vs 4096×65). Failure would show if even the best E underfits (accuracy ≤ label-linear) or if the tuning split is too noisy to pick a good E.
```

```python
"""label-emb: per-position byte embeddings + linear output.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Hyperparameter: embedding dimension E chosen on last 20% of train.in (file order),
then retrained on all train. Adam, L2, fixed epochs."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
EPOCHS = 30          # epochs for each hyperparameter trial
SAMPLES = 600000     # max samples per trial (like label-linear)
EMB_CANDIDATES = [16, 32, 64, 128]  # embedding sizes to try

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_model(X, y, val_X, val_y, E, C, rng):
    d = X.shape[1]
    # parameters
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / (256 + E))).astype(np.float32)
    W = (rng.uniform(-1, 1, (E, C)) * math.sqrt(6.0 / (E + C))).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [emb, W, b]
    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = X.shape[0]
    steps = min(EPOCHS, math.ceil(SAMPLES / n))
    for epoch in range(steps):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            bs = xb.shape[0]
            # forward
            # sum embeddings: (bs, E)
            sum_emb = np.zeros((bs, E), np.float32)
            for j in range(d):
                sum_emb += emb[j, xb[:, j]]
            logits = sum_emb @ W + b  # (bs, C)
            # softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bs), yb] -= 1.0
            probs /= bs
            # gradients
            grad_W = sum_emb.T @ probs + L2 * W
            grad_b = probs.sum(axis=0)
            grad_emb = np.zeros_like(emb)
            # backprop through sum_emb
            grad_sum_emb = probs @ W.T  # (bs, E)
            for j in range(d):
                # scatter-add grad_sum_emb into grad_emb[j]
                np.add.at(grad_emb[j], xb[:, j], grad_sum_emb)
            grad_emb += L2 * emb
            grads = [grad_emb, grad_W, grad_b]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    # validation accuracy
    sum_emb_val = np.zeros((val_X.shape[0], E), np.float32)
    for j in range(d):
        sum_emb_val += emb[j, val_X[:, j]]
    logits_val = sum_emb_val @ W + b
    preds = logits_val.argmax(axis=1)
    acc = (preds == val_y).mean()
    return params, acc

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X.shape[0]
    val_size = n // 5
    train_X, train_y = X[:-val_size], y[:-val_size]
    val_X, val_y = X[-val_size:], y[-val_size:]

    rng = np.random.default_rng(SEED)
    best_E = EMB_CANDIDATES[0]
    best_acc = -1.0
    best_params = None

    for E in EMB_CANDIDATES:
        # each trial gets a fresh deterministic seed
        trial_rng = np.random.default_rng(SEED + E)
        params, acc = train_model(train_X, train_y, val_X, val_y, E, C, trial_rng)
        if acc > best_acc:
            best_acc = acc
            best_E = E
            best_params = params

    # retrain on full data with best_E
    full_rng = np.random.default_rng(SEED + best_E + 1000)
    final_params, _ = train_model(X, y, val_X, val_y, best_E, C, full_rng)
    emb, W, b = final_params

    # write weights: magic, d, E, C, then emb[d][256][E], W[E][C], b[C]
    magic = 0x4C424D45  # "EMBL" little-endian
    out = struct.pack("<4i", magic, d, best_E, C)
    out += emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index).
 * Weights format: magic(0x4C424D45), int32 d, int32 E, int32 C,
 *   float32 emb[d][256][E], float32 W[E][C], float32 b[C]. */
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
    if (nw < 16 || hd[0] != 0x4C424D45) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (d != 16 || C < 1 || C > 256 || E < 1) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * E + (long)E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * E;
    const float *b = W + (long)E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = (float *)malloc(sizeof(float) * E);
    float *logits = (float *)malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(acc, 0, sizeof(float) * E);
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            for (int k = 0; k < E; k++) acc[k] += e[k];
        }
        memcpy(logits, b, sizeof(float) * C);
        for (int k = 0; k < E; k++) {
            float ak = acc[k];
            if (ak == 0.0f) continue;
            const float *wk = W + (long)k * C;
            for (int c = 0; c < C; c++) logits[c] += ak * wk[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(acc); free(logits);
    return 0;
}
```