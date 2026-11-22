```hypothesis
Family: label-fm (factorization machine with per-class embeddings). 
Why: Adult has 14 mixed ordinal/categorical features; pairwise interactions capture feature crosses (e.g., education × occupation) that linear models miss. A factorization machine models all pairwise interactions via low-rank embeddings (k=8) with O(d·k) inference cost, cheaper than GBDT and more expressive than linear. 
Failure signal: If dev accuracy ≤ 0.8524 (label-linear) or Ir/item > 2000, the family does not justify its complexity.
```

```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Architecture: per-class bias, linear weights (256 per feature), and embeddings (256 x k per feature).
Logit_c = bias_c + sum_j linear_c[j][x_j] + 0.5 * ( (sum_j emb_c[j][x_j])^2 - sum_j emb_c[j][x_j]^2 ).sum()
Adam lr=1e-2, L2=1e-4, batch=64, epochs=min(100, ceil(600000/n)). k=8.
Weights format: int32 magic 0x464D31, d, k, C; float32 bias[C]; then for each class c, feature j:
  linear_c_j[256], emb_c_j[256*k] (row-major)."""
import json, math, struct, sys

import numpy as np

K = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 64, 600000, 100


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
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # parameters: bias[C], linear[C, d, 256], emb[C, d, 256, K]
    bias = np.zeros(C, np.float32)
    linear = np.zeros((C, d, 256), np.float32)
    emb = (rng.normal(0, 0.01, (C, d, 256, K))).astype(np.float32)

    params = [bias, linear, emb]
    shapes = [p.shape for p in params]
    flat = np.concatenate([p.ravel() for p in params])
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    t = 0

    def unpack(flat_arr):
        idx = 0
        out = []
        for shp in shapes:
            sz = int(np.prod(shp))
            out.append(flat_arr[idx:idx+sz].reshape(shp))
            idx += sz
        return out

    def pack(bs, ls, es):
        return np.concatenate([bs.ravel(), ls.ravel(), es.ravel()])

    def forward_batch(Xb):
        # Xb: (b, d) uint8
        b = Xb.shape[0]
        logits = np.zeros((b, C), np.float32)
        for c in range(C):
            lin = bias[c] + np.sum(linear[c, np.arange(d), Xb.T], axis=0)  # (b,)
            emb_sum = np.zeros((b, K), np.float32)
            emb_sq_sum = np.zeros((b, K), np.float32)
            for j in range(d):
                e = emb[c, j, Xb[:, j]]  # (b, K)
                emb_sum += e
                emb_sq_sum += e * e
            inter = 0.5 * np.sum(emb_sum * emb_sum - emb_sq_sum, axis=1)
            logits[:, c] = lin + inter
        return logits

    def loss_grad(logits, yb):
        # logits (b, C), yb (b,)
        b = logits.shape[0]
        maxl = logits.max(axis=1, keepdims=True)
        exp = np.exp(logits - maxl)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(b), yb] -= 1.0
        probs /= b
        loss = -np.log(np.maximum(1e-12, exp[np.arange(b), yb] / exp.sum(axis=1))).mean()
        return loss, probs.astype(np.float32)

    def backward_batch(Xb, grads):
        # grads: (b, C) = dL/dlogits
        b = Xb.shape[0]
        g_bias = np.zeros(C, np.float32)
        g_linear = np.zeros((C, d, 256), np.float32)
        g_emb = np.zeros((C, d, 256, K), np.float32)
        for c in range(C):
            gc = grads[:, c]  # (b,)
            g_bias[c] = gc.sum()
            # linear gradients
            for j in range(d):
                vals = Xb[:, j]
                np.add.at(g_linear[c, j], vals, gc)
            # embedding gradients
            emb_sum_c = np.zeros((b, K), np.float32)
            for j in range(d):
                emb_sum_c += emb[c, j, Xb[:, j]]
            for j in range(d):
                e = emb[c, j, Xb[:, j]]  # (b, K)
                # d(inter)/de = e_sum - e
                g_e = gc[:, None] * (emb_sum_c - e)  # (b, K)
                np.add.at(g_emb[c, j], Xb[:, j], g_e)
        return pack(g_bias, g_linear, g_emb)

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X[idx], y[idx]
            logits = forward_batch(Xb)
            loss, grads = loss_grad(logits, yb)
            g_flat = backward_batch(Xb, grads)
            # L2 regularization on linear and emb (not bias)
            g_flat[bias.size:] += L2 * flat[bias.size:]
            t += 1
            m = 0.9 * m + 0.1 * g_flat
            v = 0.999 * v + 0.001 * g_flat * g_flat
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            flat -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
        # optional: print progress
        # if ep % 10 == 0: print(f"epoch {ep} loss {loss:.4f}", file=sys.stderr)

    bias, linear, emb = unpack(flat)
    # write weights.bin
    out = struct.pack("<4i", 0x464D31, d, K, C)
    out += bias.astype("<f4").tobytes()
    for c in range(C):
        for j in range(d):
            out += linear[c, j].astype("<f4").tobytes()
            out += emb[c, j].astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Factorization Machine inference for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins).
 * Weights format: int32 magic 0x464D31, d, k, C; float32 bias[C];
 *   for each class c, feature j: linear[256], emb[256*k] (row-major). */
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
    if (nw < 16 || hd[0] != 0x464D31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || K < 1 || K > 256 || d < 1 || d > 256) {
        fprintf(stderr, "bad weights dims\n"); return 3;
    }
    long expected = 16L + (long)C * 4 + (long)C * d * (256L + 256L * K) * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + C;
    const float *emb = linear + (long)C * d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *logits = malloc(C * sizeof(float));
    float *emb_sum = malloc(K * sizeof(float));
    float *emb_sq_sum = malloc(K * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) {
            float lin = bias[c];
            const float *lin_ptr = linear + (long)c * d * 256;
            const float *emb_ptr = emb + (long)c * d * 256 * K;
            memset(emb_sum, 0, K * sizeof(float));
            memset(emb_sq_sum, 0, K * sizeof(float));
            for (int j = 0; j < d; j++) {
                int v = xi[j];
                lin += lin_ptr[(long)j * 256 + v];
                const float *e = emb_ptr + ((long)j * 256 + v) * K;
                for (int f = 0; f < K; f++) {
                    float ev = e[f];
                    emb_sum[f] += ev;
                    emb_sq_sum[f] += ev * ev;
                }
            }
            float inter = 0.0f;
            for (int f = 0; f < K; f++) {
                inter += emb_sum[f] * emb_sum[f] - emb_sq_sum[f];
            }
            logits[c] = lin + 0.5f * inter;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```