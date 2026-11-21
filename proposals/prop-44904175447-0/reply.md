```hypothesis
Family: label-emb (embedding + linear MLP). Treats all 14 features as categorical with 256 values, learns a 4-d embedding per feature value plus a per-class linear weight, sums embeddings across features, passes through a tiny ReLU MLP (8 hidden units), adds linear logits. Expects to beat label-linear (0.8524) by capturing interactions via embeddings while staying cheaper than label-mlp16 (2325 Ir/item). Failure: dev accuracy ≤ 0.8524 or Ir/item ≥ 2000.
```

```python
"""label-emb: embedding + linear MLP for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Architecture: d features, each value 0..255 -> embedding (D=4) + linear weight (C=2).
Sum embeddings -> ReLU MLP (H=8) -> logits; add linear logits -> softmax.
Adam lr=3e-3, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)).
Weights format (little-endian):
  int32 magic=0x454D424C, d, D, H, C
  for each feature j: float32 linear_weight[256*C], embedding[256*D]
  float32 linear_bias[C]
  if H>0: float32 W1[D*H], b1[H], W2[H*C], b2[C]
  else:   float32 W1[D*C], b1[C]"""
import json, math, struct, sys

import numpy as np

D, H = 4, 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x454D424C  # 'EMBL'

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n, d = X.shape
    C = card["classes"]
    assert C == 2

    rng = np.random.default_rng(SEED)
    # Parameters
    lin_w = rng.normal(0, 0.01, (d, 256, C)).astype(np.float32)
    emb   = rng.normal(0, 0.01, (d, 256, D)).astype(np.float32)
    lin_b = np.zeros(C, np.float32)
    if H:
        W1 = rng.normal(0, math.sqrt(2.0/D), (D, H)).astype(np.float32); b1 = np.zeros(H, np.float32)
        W2 = rng.normal(0, math.sqrt(2.0/H), (H, C)).astype(np.float32); b2 = np.zeros(C, np.float32)
        params = [lin_w, emb, lin_b, W1, b1, W2, b2]
    else:
        W1 = rng.normal(0, math.sqrt(2.0/D), (D, C)).astype(np.float32); b1 = np.zeros(C, np.float32)
        params = [lin_w, emb, lin_b, W1, b1]

    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward_batch(idx):
        x = X[idx]  # (b, d)
        b = x.shape[0]
        # linear logits
        lin_logits = lin_b + np.sum(lin_w[np.arange(d), x.T, :], axis=0)  # (b, C)
        # embedding sum
        emb_sum = np.sum(emb[np.arange(d), x.T, :], axis=0)  # (b, D)
        if H:
            hidden = np.maximum(emb_sum @ W1 + b1, 0)
            mlp_logits = hidden @ W2 + b2
        else:
            mlp_logits = emb_sum @ W1 + b1
        return lin_logits + mlp_logits

    def loss_grad(logits, yy):
        # softmax cross-entropy gradient
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(len(yy)), yy] -= 1.0
        return probs.astype(np.float32) / len(yy)

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            logits = forward_batch(idx)
            g = loss_grad(logits, y[idx])  # (b, C)

            # gradients
            b = idx.shape[0]
            x = X[idx]  # (b, d)
            # linear weights and bias
            grad_lin_w = np.zeros_like(lin_w)
            for j in range(d):
                vals = x[:, j]
                np.add.at(grad_lin_w[j], vals, g)
            grad_lin_b = g.sum(axis=0)

            # embeddings
            grad_emb = np.zeros_like(emb)
            # backprop through MLP
            if H:
                emb_sum = np.sum(emb[np.arange(d), x.T, :], axis=0)  # (b, D)
                hidden_pre = emb_sum @ W1 + b1
                hidden = np.maximum(hidden_pre, 0)
                # grad on mlp_logits = g
                grad_W2 = hidden.T @ g
                grad_b2 = g.sum(axis=0)
                grad_hidden = g @ W2.T
                grad_hidden[hidden_pre <= 0] = 0
                grad_W1 = emb_sum.T @ grad_hidden
                grad_b1 = grad_hidden.sum(axis=0)
                grad_emb_sum = grad_hidden @ W1.T  # (b, D)
            else:
                emb_sum = np.sum(emb[np.arange(d), x.T, :], axis=0)
                grad_W1 = emb_sum.T @ g
                grad_b1 = g.sum(axis=0)
                grad_emb_sum = g @ W1.T

            # distribute grad_emb_sum to each feature's embedding
            for j in range(d):
                vals = x[:, j]
                np.add.at(grad_emb[j], vals, grad_emb_sum)

            # L2 regularization on weights (not biases)
            grad_lin_w += L2 * lin_w
            grad_emb   += L2 * emb
            if H:
                grad_W1 += L2 * W1
                grad_W2 += L2 * W2
            else:
                grad_W1 += L2 * W1

            grads = [grad_lin_w, grad_emb, grad_lin_b, grad_W1, grad_b1]
            if H:
                grads += [grad_W2, grad_b2]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    out = struct.pack("<5i", MAGIC, d, D, H, C)
    for j in range(d):
        out += lin_w[j].astype("<f4").tobytes()
        out += emb[j].astype("<f4").tobytes()
    out += lin_b.astype("<f4").tobytes()
    if H:
        out += W1.astype("<f4").tobytes() + b1.astype("<f4").tobytes()
        out += W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes()
    else:
        out += W1.astype("<f4").tobytes() + b1.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-emb inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=14). Output record: 1 byte class index.
 * Weights format (little-endian):
 *   int32 magic=0x454D424C, d, D, H, C
 *   for each feature j: float32 linear_weight[256*C], embedding[256*D]
 *   float32 linear_bias[C]
 *   if H>0: float32 W1[D*H], b1[H], W2[H*C], b2[C]
 *   else:   float32 W1[D*C], b1[C] */
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
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }

    /* expected size:
     * 5 int32 = 20
     * d * (256*C + 256*D) * 4
     * C * 4
     * if H: (D*H + H + H*C + C) * 4
     * else: (D*C + C) * 4
     */
    long expected = 20
        + (long)d * (256L*C + 256L*D) * 4
        + (long)C * 4;
    if (H) expected += (long)(D*H + H + H*C + C) * 4;
    else   expected += (long)(D*C + C) * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld vs %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 5);
    const float *lin_w = p; p += (long)d * 256 * C;
    const float *emb   = p; p += (long)d * 256 * D;
    const float *lin_b = p; p += C;
    const float *W1 = p;
    if (H) { p += (long)D * H; }
    else   { p += (long)D * C; }
    const float *b1 = p; p += H ? H : C;
    const float *W2 = H ? p : NULL; if (H) p += (long)H * C;
    const float *b2 = H ? p : NULL;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * D);
    float *logits  = malloc(sizeof(float) * C);
    float *hidden  = H ? malloc(sizeof(float) * H) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        /* linear logits = lin_b + sum_j lin_w[j][xi[j]] */
        for (int c = 0; c < C; c++) logits[c] = lin_b[c];
        for (int j = 0; j < d; j++) {
            const float *w = lin_w + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) logits[c] += w[c];
        }

        /* embedding sum */
        for (int u = 0; u < D; u++) emb_sum[u] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            for (int u = 0; u < D; u++) emb_sum[u] += e[u];
        }

        /* MLP */
        if (H) {
            for (int u = 0; u < H; u++) {
                float sum = b1[u];
                for (int v = 0; v < D; v++) sum += emb_sum[v] * W1[(long)v * H + u];
                hidden[u] = sum > 0.0f ? sum : 0.0f;
            }
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int u = 0; u < H; u++) sum += hidden[u] * W2[(long)u * C + c];
                logits[c] += sum;
            }
        } else {
            for (int c = 0; c < C; c++) {
                float sum = b1[c];
                for (int v = 0; v < D; v++) sum += emb_sum[v] * W1[(long)v * C + c];
                logits[c] += sum;
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```