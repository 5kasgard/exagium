```hypothesis
Family: label-fm (Factorization Machine with shared embeddings).
Why: Adult has 14 fields with mixed ordinal/categorical types; FM captures pairwise interactions with O(k·d) parameters and compute (k=4 latent dim). Linear model ignores interactions; MLP16 uses 16 hidden units (≈d·16 params) but no explicit field-aware structure. FM's field embeddings are smaller and faster than MLP16's dense layer, and should beat linear accuracy.
Failure signal: dev accuracy ≤ label-linear (0.8524) or Ir/item ≥ label-mlp16 (2325).
```

```python
"""label-fm: Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes each of the 14 bytes as a field with 256 values (0-255).
Model: logit_c = bias_c + sum_i W_c[i, x_i] + sum_{i<j} <V_i[x_i], V_j[x_j]>
  where V_i[v] in R^k (k=4). Trains with Adam, L2, early stop on tuning split (last 20%)."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (k chosen on tuning split)
K_CHOICES = [2, 4, 8]
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 1e-2, 1e-4, 256, 600000, 30
MAGIC = 0x464D4C31  # 'FML1'

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
    assert d == 14 and C == 2

    # Split: last 20% for tuning
    split = int(n * 0.8)
    X_tr, y_tr = X[:split], y[:split]
    X_va, y_va = X[split:], y[split:]
    n_tr = X_tr.shape[0]

    rng = np.random.default_rng(SEED)

    def init_params(k):
        # bias[C], linear[C, d, 256], embed[d, 256, k]
        scale = 0.01
        bias = rng.normal(0, scale, C).astype(np.float32)
        linear = rng.normal(0, scale, (C, d, 256)).astype(np.float32)
        embed = rng.normal(0, scale, (d, 256, k)).astype(np.float32)
        return bias, linear, embed

    def forward(Xb, bias, linear, embed):
        # Xb: (b, d) uint8
        b = Xb.shape[0]
        k = embed.shape[2]
        # linear term
        lin = np.zeros((b, C), np.float32)
        for c in range(C):
            for i in range(d):
                lin[:, c] += linear[c, i, Xb[:, i]]
        # interaction term: 0.5 * sum_k (sum_i e_{i,k})^2 - sum_i e_{i,k}^2
        # embeddings for batch: (b, d, k)
        emb = embed[Xb]  # (b, d, k)
        sum_emb = emb.sum(axis=1)          # (b, k)
        sq_sum_emb = (emb ** 2).sum(axis=1) # (b, k)
        inter = 0.5 * ((sum_emb ** 2 - sq_sum_emb).sum(axis=1))  # (b,)
        logits = lin + inter[:, None] + bias[None, :]
        return logits

    def loss_acc(logits, yb):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        loss = -np.log(probs[np.arange(len(yb)), yb] + 1e-12).mean()
        acc = (logits.argmax(axis=1) == yb).mean()
        return loss, acc

    def train_eval(k, epochs, Xtr, ytr, Xva, yva):
        bias, linear, embed = init_params(k)
        params = [bias, linear, embed]
        state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
        t = 0
        best_acc = -1.0
        best_params = None
        for ep in range(epochs):
            order = rng.permutation(n_tr)
            for s in range(0, n_tr, BATCH):
                idx = order[s:s + BATCH]
                Xb, yb = Xtr[idx], ytr[idx]
                # forward
                b = Xb.shape[0]
                kdim = embed.shape[2]
                emb = embed[Xb]  # (b, d, k)
                sum_emb = emb.sum(axis=1)
                sq_sum_emb = (emb ** 2).sum(axis=1)
                inter = 0.5 * ((sum_emb ** 2 - sq_sum_emb).sum(axis=1))
                lin = np.zeros((b, C), np.float32)
                for c in range(C):
                    for i in range(d):
                        lin[:, c] += linear[c, i, Xb[:, i]]
                logits = lin + inter[:, None] + bias[None, :]
                # softmax grad
                logits -= logits.max(axis=1, keepdims=True)
                exp = np.exp(logits)
                probs = exp / exp.sum(axis=1, keepdims=True)
                probs[np.arange(b), yb] -= 1.0
                probs /= b
                # gradients
                g_bias = probs.sum(axis=0)
                g_linear = np.zeros_like(linear)
                g_embed = np.zeros_like(embed)
                for c in range(C):
                    for i in range(d):
                        g_linear[c, i, Xb[:, i]] += probs[:, c]
                # interaction grad: dL/de_{i,k} = (sum_j e_{j,k} - e_{i,k}) * probs_sum
                probs_sum = probs.sum(axis=1)  # (b,)
                for i in range(d):
                    e_i = emb[:, i, :]  # (b, k)
                    grad_e = (sum_emb - e_i) * probs_sum[:, None]  # (b, k)
                    # accumulate into g_embed for each value
                    for val in range(256):
                        mask = (Xb[:, i] == val)
                        if mask.any():
                            g_embed[i, val] += grad_e[mask].sum(axis=0)
                # L2
                g_bias += L2 * bias
                g_linear += L2 * linear
                g_embed += L2 * embed
                grads = [g_bias, g_linear, g_embed]
                t += 1
                for p, gr, (m, v) in zip(params, grads, state):
                    m *= 0.9; m += 0.1 * gr
                    v *= 0.999; v += 0.001 * gr * gr
                    m_hat = m / (1 - 0.9 ** t)
                    v_hat = v / (1 - 0.999 ** t)
                    p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
            # validate
            va_logits = forward(Xva, bias, linear, embed)
            _, va_acc = loss_acc(va_logits, yva)
            if va_acc > best_acc:
                best_acc = va_acc
                best_params = [p.copy() for p in params]
        return best_acc, best_params

    # Choose k on tuning split
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_tr))
    best_k, best_acc, best_params = -1, -1.0, None
    for k in K_CHOICES:
        acc, params = train_eval(k, epochs, X_tr, y_tr, X_va, y_va)
        if acc > best_acc:
            best_acc, best_k, best_params = acc, k, params
    # Retrain on full data with best_k
    _, final_params = train_eval(best_k, epochs, X, y, X_va[:1], y_va[:1])  # dummy val
    bias, linear, embed = final_params
    k = best_k

    # Pack weights: magic, d, k, C, bias[C], linear[C*d*256], embed[d*256*k]
    out = struct.pack("<4i", MAGIC, d, k, C)
    out += bias.astype("<f4").tobytes()
    out += linear.astype("<f4").tobytes()
    out += embed.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Input record: 14 bytes (one per field). Output: 1 byte class index.
 * Weights layout: int32 magic, d, k, C; float32 bias[C]; linear[C][d][256]; embed[d][256][k]. */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (d != 14 || C != 2 || k <= 0 || k > 32) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16
        + (long)C * 4
        + (long)C * d * 256 * 4
        + (long)d * 256 * k * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *bias = (const float *)(hd + 4);
    const float *linear = bias + C;
    const float *embed = linear + (long)C * d * 256;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_emb = (float *)malloc(sizeof(float) * k);
    float *sq_sum_emb = (float *)malloc(sizeof(float) * k);
    float logits[2];

    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if (len != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }

        // zero accumulators
        for (int kk = 0; kk < k; kk++) sum_emb[kk] = sq_sum_emb[kk] = 0.0f;
        // linear term
        for (int c = 0; c < C; c++) logits[c] = bias[c];
        for (int f = 0; f < d; f++) {
            unsigned char v = xi[f];
            for (int c = 0; c < C; c++) logits[c] += linear[((long)c * d + f) * 256 + v];
            // embeddings
            const float *e = embed + ((long)f * 256 + v) * k;
            for (int kk = 0; kk < k; kk++) {
                float val = e[kk];
                sum_emb[kk] += val;
                sq_sum_emb[kk] += val * val;
            }
        }
        // interaction
        float inter = 0.0f;
        for (int kk = 0; kk < k; kk++) inter += sum_emb[kk] * sum_emb[kk] - sq_sum_emb[kk];
        inter *= 0.5f;
        for (int c = 0; c < C; c++) logits[c] += inter;

        unsigned char best = logits[1] > logits[0] ? 1 : 0;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```