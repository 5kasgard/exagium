```hypothesis
The label-emb-mlp family learns an embedding vector for each of the 256 possible values of every feature, sums them, and passes the result through a small ReLU hidden layer. This captures non‑linear feature interactions while keeping the first‑layer dimension low (14×8 instead of one‑hot’s ~3k), so it should beat label-linear’s 0.8524 accuracy on Adult and run faster than label-mlp16. If dev accuracy stays ≤0.8524 or inference cost exceeds label-linear, the family does not justify its complexity.
```
```python
"""label-emb-mlp: embedding + one hidden ReLU layer for tabular byte features.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed=1), single thread, stdlib+numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (fixed, no tuning split used)
E = 8          # embedding dimension
H = 16         # hidden layer size (0 for linear on embeddings)
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    emb = [rng.normal(0, 1/math.sqrt(E), size=(256, E)).astype(np.float32) for _ in range(d)]
    if H:
        W1 = rng.normal(0, math.sqrt(6.0/E), size=(E, H)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = rng.normal(0, math.sqrt(6.0/H), size=(H, C)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = emb + [W1, b1, W2, b2]
    else:
        W_out = rng.normal(0, math.sqrt(6.0/E), size=(E, C)).astype(np.float32)
        b_out = np.zeros(C, np.float32)
        params = emb + [W_out, b_out]

    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]
            yb = y[idx]
            bsz = len(idx)

            # Forward
            sum_emb = np.zeros((bsz, E), np.float32)
            for j in range(d):
                sum_emb += emb[j][xb[:, j]]
            if H:
                hidden_pre = sum_emb @ W1 + b1
                hidden = np.maximum(hidden_pre, 0)
                logits = hidden @ W2 + b2
            else:
                logits = sum_emb @ W_out + b_out

            # Softmax cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(bsz), yb] -= 1.0
            probs /= bsz  # dL/dlogits

            # Backward
            grads = [None] * len(params)
            if H:
                grad_W2 = hidden.T @ probs + L2 * W2
                grad_b2 = probs.sum(axis=0)
                grad_hidden = probs @ W2.T
                grad_hidden[hidden == 0] = 0
                grad_W1 = sum_emb.T @ grad_hidden + L2 * W1
                grad_b1 = grad_hidden.sum(axis=0)
                grad_sum_emb = grad_hidden @ W1.T
                # embedding gradients
                for j in range(d):
                    g = np.zeros_like(emb[j])
                    np.add.at(g, xb[:, j], grad_sum_emb)
                    g += L2 * emb[j]
                    grads[j] = g
                grads[-4] = grad_W1
                grads[-3] = grad_b1
                grads[-2] = grad_W2
                grads[-1] = grad_b2
            else:
                grad_W_out = sum_emb.T @ probs + L2 * W_out
                grad_b_out = probs.sum(axis=0)
                grad_sum_emb = probs @ W_out.T
                for j in range(d):
                    g = np.zeros_like(emb[j])
                    np.add.at(g, xb[:, j], grad_sum_emb)
                    g += L2 * emb[j]
                    grads[j] = g
                grads[-2] = grad_W_out
                grads[-1] = grad_b_out

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    # Header: magic, d, E, H, C (5 int32)
    magic = 0x454D424C  # "LMBE" in little-endian
    header = struct.pack("<5i", magic, d, E, H, C)
    out = bytearray(header)
    # Embeddings: d * 256 * E floats
    for j in range(d):
        out += emb[j].astype("<f4").tobytes()
    if H:
        out += W1.astype("<f4").tobytes()
        out += b1.astype("<f4").tobytes()
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()
    else:
        out += W_out.astype("<f4").tobytes()
        out += b_out.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-emb-mlp inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output: one byte class index. */
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
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h,1,4,f); fwrite(r,1,k,f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }

    long expected = 20
        + (long)d * 256 * E * 4
        + (H ? (long)E*H*4 + H*4 + (long)H*C*4 + C*4
             : (long)E*C*4 + C*4);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (H ? (long)E*H : 0);
    const float *W2 = b1 + (H ? H : 0);
    const float *b2 = W2 + (H ? (long)H*C : 0);
    const float *W_out = W1, *b_out = b1;  // aliases when H==0

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = malloc(sizeof(float) * E);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        memset(sum_emb, 0, E * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            for (int v = 0; v < E; v++) sum_emb[v] += e[v];
        }

        if (H) {
            for (int h = 0; h < H; h++) {
                float acc = b1[h];
                for (int v = 0; v < E; v++) acc += sum_emb[v] * W1[(long)v * H + h];
                hidden[h] = acc > 0.0f ? acc : 0.0f;
            }
            for (int c = 0; c < C; c++) {
                float acc = b2[c];
                for (int h = 0; h < H; h++) acc += hidden[h] * W2[(long)h * C + c];
                logits[c] = acc;
            }
        } else {
            for (int c = 0; c < C; c++) {
                float acc = b_out[c];
                for (int v = 0; v < E; v++) acc += sum_emb[v] * W_out[(long)v * C + c];
                logits[c] = acc;
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```