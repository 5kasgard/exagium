```hypothesis
Prototype-based classifier (label-proto) with 3 prototypes per class trained via gradient descent on softmax of negative squared Euclidean distances. Features are z-scored per pixel. The 8x8 digit images have clustered class structure that prototypes can capture non-linearly, while inference remains ~30*64=1920 FLOPs. Fails if classes aren't compactly clusterable or prototypes collapse during training.
```
```python
"""Prototype family for label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Learns P prototypes per class (default 3) by minimizing cross-entropy on
negative squared Euclidean distance to z-scored features. Adam lr=1e-2,
epochs=100, deterministic seed=1. Exports: magic 0x50524F54, d, P, C;
float32 mean[d], std[d]; float32 proto[P*C][d]."""
import json, math, struct, sys

import numpy as np

P = 3
SEED, LR, EPOCHS = 1, 1e-2, 100


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
    C = card["classes"]
    d = X.shape[1]
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    # z-score per feature (all ordinal)
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    Z = ((X - mu) / sd).astype(np.float32)

    # split for validation (last 20% in file order)
    split = int(0.8 * n)
    Z_tr, Z_val = Z[:split], Z[split:]
    y_tr, y_val = y[:split], y[split:]
    n_tr = Z_tr.shape[0]

    # initialize prototypes: pick random samples from each class
    proto = np.zeros((P * C, d), dtype=np.float32)
    for c in range(C):
        idx = np.where(y_tr == c)[0]
        if len(idx) == 0:
            idx = np.where(y == c)[0]
        for p in range(P):
            proto[c * P + p] = Z_tr[rng.choice(idx)] if len(idx) else rng.normal(size=d).astype(np.float32)

    # Adam state
    m = np.zeros_like(proto)
    v = np.zeros_like(proto)
    t = 0

    def forward(Zb):
        # Zb: (B, d), proto: (K, d) -> dist: (B, K)
        # ||Zb - proto||^2 = Zb^2 + proto^2 - 2 Zb proto^T
        Zb2 = np.sum(Zb * Zb, axis=1, keepdims=True)      # (B,1)
        W2 = np.sum(proto * proto, axis=1, keepdims=True).T # (1,K)
        cross = Zb @ proto.T                                # (B,K)
        dist = Zb2 + W2 - 2 * cross
        logits = -dist
        return logits, dist

    def loss_acc(logits, yb):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        prob = exp / exp.sum(axis=1, keepdims=True)
        nll = -np.log(prob[np.arange(len(yb)), yb] + 1e-12).mean()
        acc = (prob.argmax(axis=1) == yb).mean()
        return nll, acc, prob

    best_val_acc = -1.0
    best_proto = proto.copy()

    for epoch in range(EPOCHS):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, 64):
            idx = order[s:s+64]
            Zb, yb = Z_tr[idx], y_tr[idx]
            logits, dist = forward(Zb)
            nll, _, prob = loss_acc(logits, yb)
            # gradient: dL/dproto = 2 * (prob - onehot) @ Zb  (since logits = -dist, dlogits/ddist = -1, ddist/dproto = 2(proto - Zb))
            # Actually: L = -log softmax(-dist)_y
            # dL/ddist = prob - onehot  (shape B,K)
            # ddist/dproto = 2(proto - Zb) -> for each k: 2 * sum_b (prob_bk - 1_{y=k}) * (proto_k - Zb_b)
            grad = 2 * ((prob.T @ proto) - (prob.T @ Zb))  # This is wrong
            # Correct: for each prototype k: grad_k = 2 * sum_b (prob[b,k] - (yb==k)) * (proto[k] - Zb[b])
            # = 2 * ( (prob[:,k].sum()) * proto[k] - prob[:,k] @ Zb )
            # Vectorized:
            coeff = prob.T @ np.ones(len(yb))  # (K,) sum_b prob[b,k]
            grad = 2 * (coeff[:, None] * proto - prob.T @ Zb)  # (K,d)
            t += 1
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad * grad
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            proto -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

        # validation
        logits_val, _ = forward(Z_val)
        _, val_acc, _ = loss_acc(logits_val, y_val)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_proto = proto.copy()

    proto = best_proto

    # final training on all data for a few more epochs (optional, but keeps determinism simple)
    # just use the best_proto from validation

    # Export
    magic = 0x50524F54  # 'PROT'
    out = struct.pack("<4i", magic, d, P, C)
    out += mu.astype("<f4").tobytes()
    out += sd.astype("<f4").tobytes()
    out += proto.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Prototype family for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the class of nearest prototype.
 * Weights format: magic(0x50524F54), d, P, C; float32 mean[d], std[d]; float32 proto[P*C][d]. */
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
    if (nw < 16 || hd[0] != 0x50524F54) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], P = hd[2], C = hd[3];
    if (d <= 0 || P <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 16 + 2L * d * 4 + (long)P * C * d * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *mean = (const float *)(hd + 4);
    const float *std = mean + d;
    const float *proto = std + d;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        float best_dist = 1e30f;
        unsigned char best_class = 0;

        for (int c = 0; c < C; c++) {
            for (int p = 0; p < P; p++) {
                const float *w = proto + ((long)c * P + p) * d;
                float dist = 0.0f;
                for (int j = 0; j < d; j++) {
                    float z = (xi[j] - mean[j]) / std[j];
                    float diff = z - w[j];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_class = (unsigned char)c;
                }
            }
        }
        put_rec(stdout, &best_class, 1);
    }
    return 0;
}
```