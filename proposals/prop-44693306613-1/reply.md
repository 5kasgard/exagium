```hypothesis
Family: label-mlp8c (8-hidden-unit MLP with 17-entry per-pixel tables). Optdigits pixels only take 17 values (0–16); using compact 17-entry lookup tables instead of 256 reduces first-layer parameters 15× vs label-linear (64×17×8 = 8.7k vs 64×256×10 = 163k) while adding non-linearity. Expect dev accuracy > 0.9613 (label-linear) at lower Ir/item. Failure if accuracy ≤ 0.9613 or Ir/item ≥ label-linear's 5059.
```
```python
import json, math, struct, sys
import numpy as np

H = 8
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x4D4C5038  # 'MLP8'

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
    assert d == 64 and C == 10
    # Values are 0..16 per card description
    V = 17

    rng = np.random.default_rng(SEED)
    # Tables: per pixel (d), per value (V), per hidden unit (H)
    tables = (rng.uniform(-1, 1, (d, V, H)) * math.sqrt(6.0 / V)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
    b2 = np.zeros(C, np.float32)

    params = [tables, b1, W2, b2]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    n = X.shape[0]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = X[idx], y[idx]
            # Forward
            hid = b1 + np.sum(tables[np.arange(d)[:, None], xb.T], axis=0)  # (BATCH, H)
            hid_relu = np.maximum(hid, 0)
            logits = hid_relu @ W2 + b2
            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad_out = (probs / len(yb)).astype(np.float32)  # (BATCH, C)
            # Backprop
            grad_W2 = hid_relu.T @ grad_out + L2 * W2
            grad_b2 = grad_out.sum(axis=0)
            grad_hid = grad_out @ W2.T * (hid > 0)
            grad_b1 = grad_hid.sum(axis=0)
            grad_tables = np.zeros_like(tables)
            for j in range(d):
                vals = xb[:, j]
                for v in range(V):
                    mask = (vals == v)
                    if np.any(mask):
                        grad_tables[j, v] = grad_hid[mask].sum(axis=0)
            grad_tables += L2 * tables
            grads = [grad_tables, grad_b1, grad_W2, grad_b2]
            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    out = struct.pack("<4i", MAGIC, d, H, C)
    out += tables.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
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
    if (nw < 16 || hd[0] != 0x4D4C5038) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3];
    const int V = 17;
    long expected = 16 + 4L * ((long)d * V * H + H + (long)H * C + C);
    if (C < 1 || C > 256 || nw != expected) {
        fprintf(stderr, "bad weights size: got %ld expect %ld\n", nw, expected); return 3;
    }
    const float *tables = (const float *)(hd + 4);
    const float *b1 = tables + (long)d * V * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hid = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int h = 0; h < H; h++) hid[h] = b1[h];
        for (int j = 0; j < d; j++) {
            unsigned char v = xi[j];
            if (v >= V) v = V - 1; // safety, though data guarantees 0..16
            const float *t = tables + ((long)j * V + v) * H;
            for (int h = 0; h < H; h++) hid[h] += t[h];
        }
        for (int h = 0; h < H; h++) if (hid[h] < 0.0f) hid[h] = 0.0f;
        for (int c = 0; c < C; c++) scores[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float hv = hid[h];
            const float *w = W2 + (long)h * C;
            for (int c = 0; c < C; c++) scores[c] += hv * w[c];
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```