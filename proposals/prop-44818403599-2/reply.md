```hypothesis
Family: label-emb-mlp — shared byte embeddings + one ReLU layer.
Why: The 16-byte context has strong local patterns; sharing embeddings across positions captures n-gram statistics with far fewer parameters than per-position one-hot, allowing a wider hidden layer within the same CPU budget. Tuning embedding size (E) and hidden size (H) on the 20% tuning split should beat label-linear’s 33.75% accuracy while keeping inference fast (one embedding gather per byte, then two small matrix multiplies).
Failure signal: If dev accuracy ≤ 0.34 or instructions/item ≥ 2× label-linear, the family does not justify its complexity.
```

```python
"""label-emb-mlp: shared embeddings + one ReLU layer for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format (little-endian):
  int32 magic=0x454D424C, d, E, H, C
  float32 emb[256][E]
  float32 W1[d*E][H], b1[H]
  float32 W2[H][C], b2[C]
If H==0: linear on concatenated embeddings (W1[d*E][C], b1[C]).
"""
import json, math, struct, sys

import numpy as np

# Hyper-parameters (will be tuned on the last 20% of train.in)
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def tune_hyperparams(card, X, y):
    """Grid search on last 20% of data (tuning split)."""
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_va, y_va = X[split:], y[split:]

    best_acc = -1.0
    best_E, best_H = 8, 64

    for E in (8, 12, 16):
        for H in (32, 64, 96):
            # Quick training (few epochs) for tuning
            rng = np.random.default_rng(SEED)
            emb = (rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
            d = card["input"]["bytes"]
            k = d * E
            if H:
                W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
                b1 = np.zeros(H, np.float32)
                W2 = (rng.uniform(-1, 1, (H, card["classes"])) * math.sqrt(6.0 / H)).astype(np.float32)
                b2 = np.zeros(card["classes"], np.float32)
                params = [emb, W1, b1, W2, b2]
            else:
                W1 = (rng.uniform(-1, 1, (k, card["classes"])) * math.sqrt(6.0 / k)).astype(np.float32)
                b1 = np.zeros(card["classes"], np.float32)
                params = [emb, W1, b1]

            state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
            t = 0
            epochs = min(20, math.ceil(SAMPLES / n))
            for _ in range(epochs):
                order = rng.permutation(len(X_tr))
                for s in range(0, len(X_tr), BATCH):
                    idx = order[s:s + BATCH]
                    xb, yb = X_tr[idx], y_tr[idx]
                    # forward
                    flat = emb[xb].reshape(len(xb), -1)  # (B, d*E)
                    if H:
                        h = np.maximum(flat @ W1 + b1, 0)
                        logits = h @ W2 + b2
                    else:
                        logits = flat @ W1 + b1
                    # softmax + grad
                    logits -= logits.max(axis=1, keepdims=True)
                    exp = np.exp(logits)
                    probs = exp / exp.sum(axis=1, keepdims=True)
                    probs[np.arange(len(yb)), yb] -= 1.0
                    grads = probs / len(yb)  # (B, C)

                    if H:
                        g2 = grads
                        gh = g2 @ W2.T * (h > 0)
                        gW2 = h.T @ g2 + L2 * W2
                        gb2 = g2.sum(0)
                        gW1 = flat.T @ gh + L2 * W1
                        gb1 = gh.sum(0)
                        # embedding grad
                        gflat = gh @ W1.T
                        gemb = np.zeros_like(emb)
                        np.add.at(gemb, xb.ravel(), gflat.reshape(-1, E))
                    else:
                        gW1 = flat.T @ grads + L2 * W1
                        gb1 = grads.sum(0)
                        gflat = grads @ W1.T
                        gemb = np.zeros_like(emb)
                        np.add.at(gemb, xb.ravel(), gflat.reshape(-1, E))

                    t += 1
                    grad_list = [gemb, gW1, gb1] + ([gW2, gb2] if H else [])
                    for i, (p, gr) in enumerate(zip(params, grad_list)):
                        m, v = state[i]
                        m *= 0.9; m += 0.1 * gr
                        v *= 0.999; v += 0.001 * gr * gr
                        p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

            # validation accuracy
            flat_va = emb[X_va].reshape(len(X_va), -1)
            if H:
                h_va = np.maximum(flat_va @ W1 + b1, 0)
                logits_va = h_va @ W2 + b2
            else:
                logits_va = flat_va @ W1 + b1
            pred = logits_va.argmax(axis=1)
            acc = (pred == y_va).mean()
            if acc > best_acc:
                best_acc, best_E, best_H = acc, E, H
    return best_E, best_H

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]

    # Tune on last 20% of train.in (file order)
    E, H = tune_hyperparams(card, X, y)

    # Retrain on full training set with chosen hyper-parameters
    rng = np.random.default_rng(SEED)
    emb = (rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
    k = d * E
    if H:
        W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [emb, W1, b1, W2, b2]
    else:
        W1 = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
        b1 = np.zeros(C, np.float32)
        params = [emb, W1, b1]

    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / len(X)))):
        order = rng.permutation(len(X))
        for s in range(0, len(X), BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            flat = emb[xb].reshape(len(xb), -1)
            if H:
                h = np.maximum(flat @ W1 + b1, 0)
                logits = h @ W2 + b2
            else:
                logits = flat @ W1 + b1
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            probs = exp / exp.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grads = probs / len(yb)

            if H:
                g2 = grads
                gh = g2 @ W2.T * (h > 0)
                gW2 = h.T @ g2 + L2 * W2
                gb2 = g2.sum(0)
                gW1 = flat.T @ gh + L2 * W1
                gb1 = gh.sum(0)
                gflat = gh @ W1.T
                gemb = np.zeros_like(emb)
                np.add.at(gemb, xb.ravel(), gflat.reshape(-1, E))
                grad_list = [gemb, gW1, gb1, gW2, gb2]
            else:
                gW1 = flat.T @ grads + L2 * W1
                gb1 = grads.sum(0)
                gflat = grads @ W1.T
                gemb = np.zeros_like(emb)
                np.add.at(gemb, xb.ravel(), gflat.reshape(-1, E))
                grad_list = [gemb, gW1, gb1]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grad_list)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Write weights.bin
    magic = 0x454D424C  # "EMBL"
    out = struct.pack("<5i", magic, d, E, H, C)
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if H:
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes (d=16). Output record: 1 byte (class index 0..64).
 * Weights format (little-endian):
 *   int32 magic=0x454D424C, d, E, H, C
 *   float32 emb[256][E]
 *   float32 W1[d*E][H], b1[H]
 *   float32 W2[H][C], b2[C]   (only if H>0)
 * If H==0: W1[d*E][C], b1[C] (linear on concatenated embeddings).
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H < 0) { fprintf(stderr, "bad header values\n"); return 3; }
    long expected = 20L + (long)256 * E + (long)d * E * (H ? H : C) + (H ? H : C);
    if (H) expected += (long)H * C + C;
    if (nw != 4 * expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, 4L * expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + 256 * E;
    const float *b1 = W1 + (long)d * E * (H ? H : C);
    const float *W2 = b1 + (H ? H : C);
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* Concatenate embeddings for the 16 bytes into a temporary buffer on stack */
        float concat[16 * 16];  /* max E=16, d=16 => 256 floats, fits on stack */
        for (int j = 0; j < d; j++) {
            const float *e = emb + (long)xi[j] * E;
            memcpy(concat + j * E, e, sizeof(float) * E);
        }

        if (H) {
            /* hidden = ReLU(concat * W1 + b1) */
            for (int h = 0; h < H; h++) {
                float sum = b1[h];
                for (int k = 0; k < d * E; k++) sum += concat[k] * W1[(long)k * H + h];
                hidden[h] = sum > 0.0f ? sum : 0.0f;
            }
            /* logits = hidden * W2 + b2 */
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
                logits[c] = sum;
            }
        } else {
            /* logits = concat * W1 + b1  (W1 is d*E x C, b1 is C) */
            for (int c = 0; c < C; c++) {
                float sum = b1[c];
                for (int k = 0; k < d * E; k++) sum += concat[k] * W1[(long)k * C + c];
                logits[c] = sum;
            }
        }

        /* argmax */
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```