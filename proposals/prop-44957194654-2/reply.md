```hypothesis
Family: label-embed (positional byte embeddings + 1-hidden-layer MLP).
Why: Learns distributed representations of byte-position pairs (16*256*8 params) and nonlinear interactions via 64 ReLU units, capturing n-gram statistics better than linear one-hot (266k params) while being smaller and faster than label-mlp16. Expect >0.35 dev accuracy at ~10k Ir/item.
Failure: If accuracy <= label-linear (0.3375) or instructions >20k/item, the embedding bottleneck or undertraining hurts.
```
```python
import json, math, struct, sys
import numpy as np

D = 8
H = 64
SEED, LR, L2, BATCH, EPOCHS = 1, 1e-3, 1e-4, 128, 50

def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X_recs = records(sys.argv[2])
    y_recs = records(sys.argv[3])
    X = np.frombuffer(b"".join(X_recs), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in y_recs], dtype=np.int64)
    n = X.shape[0]
    rng = np.random.default_rng(SEED)

    emb = (rng.normal(0, 0.01, (d, 256, D))).astype(np.float32)
    W1 = (rng.normal(0, np.sqrt(2.0/(d*D)), (d*D, H))).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.normal(0, np.sqrt(2.0/H), (H, C))).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [emb, W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    def forward(X_batch):
        B = X_batch.shape[0]
        x_emb = emb[np.arange(d), X_batch.T].transpose(1,0,2)
        x_flat = x_emb.reshape(B, d*D)
        h_pre = x_flat @ W1 + b1
        h = np.maximum(h_pre, 0)
        logits = h @ W2 + b2
        return logits, h, h_pre, x_flat

    def loss_grad(logits, y_batch):
        B = logits.shape[0]
        logits_max = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs[np.arange(B), y_batch] -= 1.0
        return probs / B

    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            X_batch = X[idx]
            y_batch = y[idx]
            B = len(idx)

            logits, h, h_pre, x_flat = forward(X_batch)
            g_logits = loss_grad(logits, y_batch)

            dW2 = h.T @ g_logits
            db2 = g_logits.sum(axis=0)
            dh = g_logits @ W2.T
            dh_pre = dh * (h_pre > 0)
            dW1 = x_flat.T @ dh_pre
            db1 = dh_pre.sum(axis=0)
            dx_flat = dh_pre @ W1.T
            dx_emb = dx_flat.reshape(B, d, D)

            demb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(demb[j], X_batch[:, j], dx_emb[:, j])

            grads = [demb, dW1, db1, dW2, db2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                gr = gr + L2 * p
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9**t)
                v_hat = v[i] / (1 - 0.999**t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    W1_T = W1.T.astype(np.float32)
    W2_T = W2.T.astype(np.float32)
    magic = 0x454D4245
    header = struct.pack("<5i", magic, d, D, H, C)
    out = header
    out += emb.astype("<f4").tobytes()
    out += W1_T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2_T.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
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
    uint32_t k = buf[*pos] | (buf[*pos+1]<<8) | (buf[*pos+2]<<16) | ((uint32_t)buf[*pos+3]<<24);
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k;
    const unsigned char *r = buf + *pos + 4;
    *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, (k>>8)&255, (k>>16)&255, (k>>24)&255};
    fwrite(h, 1, 4, f);
    fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x454D4245) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d <= 0 || D <= 0 || H <= 0 || C <= 0 || C > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected_floats = (long)d * 256 * D + (long)H * d * D + H + (long)C * H + C;
    if (nw != 20 + 4 * expected_floats) { fprintf(stderr, "bad weights size\n"); return 3; }
    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * D;
    const float *b1 = W1 + (long)H * d * D;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)C * H;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *x_flat = malloc(sizeof(float) * d * D);
    float *h = malloc(sizeof(float) * H);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * D;
            memcpy(x_flat + j * D, e, D * sizeof(float));
        }
        for (int u = 0; u < H; u++) {
            float sum = b1[u];
            const float *w = W1 + (long)u * d * D;
            for (int i = 0; i < d * D; i++) sum += x_flat[i] * w[i];
            h[u] = sum > 0.0f ? sum : 0.0f;
        }
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            const float *w = W2 + (long)c * H;
            for (int u = 0; u < H; u++) sum += h[u] * w[u];
            logits[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```