```hypothesis
The label-embed family uses per-position byte embeddings (size E) concatenated into a 16E-dimensional vector followed by a linear layer to 65 classes. This factorizes the large one-hot weight matrix (16*256*65) into embeddings (16*256*E) + linear (16E*65), reducing parameters and improving generalization on character prediction where local byte patterns share semantics. Tuning E on the last 20% of train should beat label-linear's 33.75% accuracy; if it doesn't, the factorization is too restrictive for this task.
```
```python
"""label-embed: Positional byte embeddings + linear classifier for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weight format (little-endian):
  uint32 magic=0x454D4245 ('EMBE'), d=16, E, C=65
  float32 emb[d][256][E]       (16 * 256 * E)
  float32 W[16*E][C]           ((16*E) * C)
  float32 b[C]                 (C)
"""
import json, math, struct, sys

import numpy as np

# Hyperparameters (E will be tuned on tuning split)
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

def tune_E(card, X, y):
    """Quick tuning of embedding size E on last 20% of data."""
    n = X.shape[0]
    split = int(0.8 * n)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    C = card["classes"]
    best_E, best_acc = 4, -1.0
    for E in [4, 8, 16, 32]:
        rng = np.random.default_rng(SEED)
        emb = (rng.uniform(-1, 1, (16, 256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
        W = (rng.uniform(-1, 1, (16 * E, C)) * math.sqrt(6.0 / (16 * E))).astype(np.float32)
        b = np.zeros(C, np.float32)
        params = [emb, W, b]
        state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
        t = 0
        n_tr = X_tr.shape[0]
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_tr))
        for _ in range(epochs):
            order = rng.permutation(n_tr)
            for s in range(0, n_tr, BATCH):
                idx = order[s:s + BATCH]
                xb, yb = X_tr[idx], y_tr[idx]
                # Forward: lookup embeddings and concat
                batch_emb = emb[:, xb.T, :].transpose(1, 0, 2).reshape(len(xb), 16 * E)
                logits = batch_emb @ W + b
                logits -= logits.max(axis=1, keepdims=True)
                exp_logits = np.exp(logits)
                probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                grad_logits = probs
                grad_logits[np.arange(len(yb)), yb] -= 1.0
                grad_logits /= len(yb)
                # Backward
                grad_W = batch_emb.T @ grad_logits + L2 * W
                grad_b = grad_logits.sum(0)
                grad_batch_emb = grad_logits @ W.T
                grad_emb = np.zeros_like(emb)
                for j in range(16):
                    for i_b, byte_val in enumerate(xb[:, j]):
                        grad_emb[j, byte_val] += grad_batch_emb[i_b, j*E:(j+1)*E]
                grad_emb += L2 * emb
                grads = [grad_emb, grad_W, grad_b]
                t += 1
                for i, (p, gr) in enumerate(zip(params, grads)):
                    m, v = state[i]
                    m *= 0.9; m += 0.1 * gr; v *= 0.999; v += 0.001 * gr * gr
                    p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        # Validate
        val_emb = emb[:, X_val.T, :].transpose(1, 0, 2).reshape(len(X_val), 16 * E)
        val_logits = val_emb @ W + b
        preds = val_logits.argmax(axis=1)
        acc = (preds == y_val).mean()
        if acc > best_acc:
            best_acc, best_E = acc, E
    return best_E

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]  # 16

    # Tune E on last 20% (tuning split)
    E = tune_E(card, X, y)

    # Train on full data with best E
    rng = np.random.default_rng(SEED)
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
    W = (rng.uniform(-1, 1, (d * E, C)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b = np.zeros(C, np.float32)
    params = [emb, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = X.shape[0]
    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            batch_emb = emb[:, xb.T, :].transpose(1, 0, 2).reshape(len(xb), d * E)
            logits = batch_emb @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            grad_logits = probs
            grad_logits[np.arange(len(yb)), yb] -= 1.0
            grad_logits /= len(yb)
            grad_W = batch_emb.T @ grad_logits + L2 * W
            grad_b = grad_logits.sum(0)
            grad_batch_emb = grad_logits @ W.T
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                for i_b, byte_val in enumerate(xb[:, j]):
                    grad_emb[j, byte_val] += grad_batch_emb[i_b, j*E:(j+1)*E]
            grad_emb += L2 * emb
            grads = [grad_emb, grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr; v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)

    # Write weights
    out = struct.pack("<4I", 0x454D4245, d, E, C)
    out += emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output: 1 byte class index (0..64).
 * Weight format (little-endian):
 *   uint32 magic=0x454D4245 ('EMBE'), d=16, E, C=65
 *   float emb[d][256][E]
 *   float W[d*E][C]
 *   float b[C]
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
    uint32_t *hd = (uint32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x454D4245) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (d != 16 || C != 65 || E < 1 || E > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16 + 4L * ((long)d * 256 * E + (long)d * E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 4);
    const float *W = emb + (long)d * 256 * E;
    const float *b = W + (long)d * E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * E);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(concat + j * E, e, E * sizeof(float));
        }

        // Linear layer: scores = concat @ W + b
        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int idx = 0; idx < d * E; idx++) {
            float v = concat[idx];
            if (v == 0.0f) continue;
            const float *w_col = W + (long)idx * C;
            for (int c = 0; c < C; c++) scores[c] += v * w_col[c];
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```