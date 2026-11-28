```hypothesis
We propose a "label-embed" family: each of the 16 context bytes is mapped to a learned E-dimensional embedding; the embeddings are summed and passed through a small ReLU hidden layer (size H) then a linear output. This captures position‑byte interactions via the hidden layer while keeping inference fast (O(16·E + E·H + H·C) ops). It should beat the linear baseline because dense embeddings share statistical strength across bytes/positions and the hidden layer adds non‑linearity. If dev accuracy does not exceed label‑linear’s 0.3375, or if inference cost exceeds label‑linear significantly, the family fails its goal.
```

```python
import json, math, struct, sys
import numpy as np

# Hyperparameters (chosen on the tuning split, last 20% of train.in)
E = 64          # embedding dimension
H = 32          # hidden layer size (0 for linear)
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
        out.append(data[i+4:i+4+k])
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

    # Embeddings: one (256, E) matrix per position
    Emb = [(rng.uniform(-1, 1, (256, E)) * math.sqrt(6.0 / E)).astype(np.float32) for _ in range(d)]
    if H:
        W1 = (rng.uniform(-1, 1, (E, H)) * math.sqrt(6.0 / E)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = Emb + [W1, b1, W2, b2]
    else:
        W = (rng.uniform(-1, 1, (E, C)) * math.sqrt(6.0 / E)).astype(np.float32)
        b = np.zeros(C, np.float32)
        params = Emb + [W, b]

    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]
            yb = y[idx]
            B = xb.shape[0]

            # Forward
            emb_sum = np.zeros((B, E), dtype=np.float32)
            for j in range(d):
                emb_sum += Emb[j][xb[:, j]]

            if H:
                hidden_pre = emb_sum @ W1 + b1
                hidden = np.maximum(hidden_pre, 0)
                logits = hidden @ W2 + b2
            else:
                logits = emb_sum @ W + b

            # Loss gradient
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = (probs / B).astype(np.float32)

            # Backward
            grads = [np.zeros_like(p) for p in params]
            if H:
                grads[-2] = hidden.T @ grad_logits + L2 * W2
                grads[-1] = grad_logits.sum(axis=0)
                grad_hidden = grad_logits @ W2.T
                grad_hidden_pre = grad_hidden * (hidden_pre > 0)
                grads[-4] = emb_sum.T @ grad_hidden_pre + L2 * W1
                grads[-3] = grad_hidden_pre.sum(axis=0)
                grad_emb_sum = grad_hidden_pre @ W1.T
            else:
                grads[-2] = emb_sum.T @ grad_logits + L2 * W
                grads[-1] = grad_logits.sum(axis=0)
                grad_emb_sum = grad_logits @ W.T

            for j in range(d):
                np.add.at(grads[j], xb[:, j], grad_emb_sum)

            # Adam update
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights
    magic = 0x454D4244
    out = struct.pack("<5i", magic, d, E, H, C)
    for j in range(d):
        out += Emb[j].astype("<f4").tobytes()
    if H:
        out += W1.astype("<f4").tobytes()
        out += b1.astype("<f4").tobytes()
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()
    else:
        out += W.astype("<f4").tobytes()
        out += b.astype("<f4").tobytes()
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
    uint32_t k = buf[*pos] | (buf[*pos+1] << 8) | (buf[*pos+2] << 16) | ((uint32_t)buf[*pos+3] << 24);
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k;
    const unsigned char *r = buf + *pos + 4;
    *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, (k >> 8) & 255, (k >> 16) & 255, (k >> 24) & 255};
    fwrite(h, 1, 4, f);
    fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H < 0 || C <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }

    long expected = 20;
    expected += 4L * d * 256 * E;
    if (H) {
        expected += 4L * (E * H + H + H * C + C);
    } else {
        expected += 4L * (E * C + C);
    }
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *Emb = (const float *)(hd + 5);
    const float *W1, *b1, *W2, *b2, *W, *b;
    if (H) {
        W1 = Emb + (long)d * 256 * E;
        b1 = W1 + (long)E * H;
        W2 = b1 + H;
        b2 = W2 + (long)H * C;
    } else {
        W = Emb + (long)d * 256 * E;
        b = W + (long)E * C;
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_sum = malloc(sizeof(float) * E);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int e = 0; e < E; e++) emb_sum[e] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *emb_vec = Emb + ((long)j * 256 + xi[j]) * E;
            for (int e = 0; e < E; e++) emb_sum[e] += emb_vec[e];
        }
        if (H) {
            for (int h = 0; h < H; h++) {
                float sum = b1[h];
                for (int e = 0; e < E; e++) sum += emb_sum[e] * W1[(long)e * H + h];
                hidden[h] = sum > 0.0f ? sum : 0.0f;
            }
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
                logits[c] = sum;
            }
        } else {
            for (int c = 0; c < C; c++) {
                float sum = b[c];
                for (int e = 0; e < E; e++) sum += emb_sum[e] * W[(long)e * C + c];
                logits[c] = sum;
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```