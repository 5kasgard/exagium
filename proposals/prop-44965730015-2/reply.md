```hypothesis
We propose "label-shared-emb": a shared character embedding (256×E) plus positional embeddings (16×E), concatenated and fed to a linear classifier. This captures character similarities and positional effects with far fewer parameters than label-linear's per-position one-hot (41k vs 266k). It should exceed label-linear's 33.75% dev accuracy at similar or lower inference cost because embedding lookups are cheap and the linear layer is smaller. If accuracy on the tuning split does not beat label-linear, the hypothesis is falsified.
```

```python
"""label-shared-emb: shared character embeddings + positional embeddings + linear classifier.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Weights format: magic 0x53454D42, int32 d, E, C; float32 byte_emb[256*E], pos_emb[d*E], W[(d*E)*C], b[C]."""
import json, math, struct, sys

import numpy as np

E = 16          # embedding dimension
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

MAGIC = 0x53454D42  # "SEMB"

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # Xavier initialization
    byte_emb = rng.normal(0, math.sqrt(2.0 / (256 + E)), size=(256, E)).astype(np.float32)
    pos_emb  = rng.normal(0, math.sqrt(2.0 / (d + E)), size=(d, E)).astype(np.float32)
    W = rng.normal(0, math.sqrt(2.0 / (d*E + C)), size=(d*E, C)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    params = [byte_emb, pos_emb, W, b]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(X_batch):
        # X_batch: (B, d) uint8
        B = X_batch.shape[0]
        Z = np.empty((B, d*E), dtype=np.float32)
        for j in range(d):
            Z[:, j*E:(j+1)*E] = byte_emb[X_batch[:, j]] + pos_emb[j]
        logits = Z @ W + b
        return Z, logits

    def softmax_grad(logits, y_batch):
        B = logits.shape[0]
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs[np.arange(B), y_batch] -= 1.0
        return probs / B  # (B, C)

    for _ in range(min(MAX_EPOCHS, math.ceil(SAMPLES / n))):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X[idx], y[idx]
            Z, logits = forward(Xb)
            G = softmax_grad(logits, yb)                 # (B, C)
            # Gradients
            gW = Z.T @ G + L2 * W
            gb = G.sum(axis=0)
            gZ = G @ W.T                                 # (B, d*E)
            # Split gZ into per-position gradients
            g_byte_emb = np.zeros_like(byte_emb)
            g_pos_emb  = np.zeros_like(pos_emb)
            for j in range(d):
                gZj = gZ[:, j*E:(j+1)*E]
                # byte_emb gradient: scatter-add
                np.add.at(g_byte_emb, Xb[:, j], gZj)
                # pos_emb gradient: sum over batch
                g_pos_emb[j] = gZj.sum(axis=0)
            grads = [g_byte_emb, g_pos_emb, gW, gb]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    out = struct.pack("<4i", MAGIC, d, E, C)
    out += byte_emb.astype("<f4").tobytes()
    out += pos_emb.astype("<f4").tobytes()
    out += W.astype("<f4").tobytes()
    out += b.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-shared-emb inference: ./infer weights.bin in.rec N
 * Each input record is d bytes (d=16); output is one byte (class index). */
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
    if (nw < 16 || hd[0] != 0x53454D42) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * (256L * E + (long)d * E + (long)d * E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *byte_emb = (const float *)(hd + 4);
    const float *pos_emb  = byte_emb + 256L * E;
    const float *W        = pos_emb  + (long)d * E;
    const float *b        = W        + (long)d * E * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *z = malloc(sizeof(float) * (long)d * E);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d; j++) {
            const float *be = byte_emb + (long)xi[j] * E;
            const float *pe = pos_emb  + (long)j * E;
            float *zj = z + (long)j * E;
            for (int e = 0; e < E; e++) zj[e] = be[e] + pe[e];
        }

        for (int c = 0; c < C; c++) scores[c] = b[c];
        for (int k = 0; k < d * E; k++) {
            float zk = z[k];
            if (zk == 0.0f) continue;
            const float *wk = W + (long)k * C;
            for (int c = 0; c < C; c++) scores[c] += zk * wk[c];
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```