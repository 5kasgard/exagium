```hypothesis
Family: label-mlp (one-hidden-layer MLP with per-byte lookup tables).
Why: Optdigits pixels have local stroke interactions that linear models miss; a hidden ReLU layer captures these while lookup-table inference stays fast.
Tuning: hidden size chosen from {16,32,64} on the last 20% of train, then retrained on all data.
Failure: if dev accuracy ≤ label-linear (0.9613), the task is linearly separable in the encoded space or the hidden layer is too small.
```

```python
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
H_CHOICES = [16, 32, 64]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def build_enc(X, types):
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)))
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc


def encode(X, enc):
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)


def train_eval(Z_tr, y_tr, Z_va, y_va, H, C, rng):
    k = Z_tr.shape[1]
    if H:
        W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [W1, b1, W2, b2]
    else:
        W1 = np.zeros((k, C), np.float32)
        b1 = np.zeros(C, np.float32)
        params = [W1, b1]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n = len(Z_tr)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z, yy = Z_tr[idx], y_tr[idx]
            if H:
                a = z @ params[0] + params[1]
                h = np.maximum(a, 0)
                logits = h @ params[2] + params[3]
                logits -= logits.max(axis=1, keepdims=True)
                P = np.exp(logits)
                P /= P.sum(axis=1, keepdims=True)
                P[np.arange(len(yy)), yy] -= 1.0
                g = (P / len(yy)).astype(np.float32)
                gh = (g @ params[2].T) * (a > 0)
                grads = [
                    z.T @ gh + L2 * params[0],
                    gh.sum(0),
                    h.T @ g + L2 * params[2],
                    g.sum(0),
                ]
            else:
                logits = z @ params[0] + params[1]
                logits -= logits.max(axis=1, keepdims=True)
                P = np.exp(logits)
                P /= P.sum(axis=1, keepdims=True)
                P[np.arange(len(yy)), yy] -= 1.0
                g = (P / len(yy)).astype(np.float32)
                grads = [z.T @ g + L2 * params[0], g.sum(0)]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9
                m += 0.1 * gr
                v *= 0.999
                v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
    if H:
        a = Z_va @ params[0] + params[1]
        h = np.maximum(a, 0)
        logits = h @ params[2] + params[3]
    else:
        logits = Z_va @ params[0] + params[1]
    pred = logits.argmax(axis=1)
    return (pred == y_va).mean(), params


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    n = X.shape[0]
    split = int(n * 0.8)
    X_tr, X_va = X[:split], X[split:]
    y_tr, y_va = y[:split], y[split:]

    best_H, best_acc, best_params, best_enc = 0, -1, None, None
    for H in H_CHOICES:
        enc = build_enc(X_tr, card["input"]["types"])
        Z_tr = encode(X_tr, enc)
        Z_va = encode(X_va, enc)
        rng = np.random.default_rng(SEED)
        acc, params = train_eval(Z_tr, y_tr, Z_va, y_va, H, C, rng)
        if acc > best_acc:
            best_acc, best_H, best_params, best_enc = acc, H, params, enc

    # retrain on all data
    enc = build_enc(X, card["input"]["types"])
    Z = encode(X, enc)
    rng = np.random.default_rng(SEED)
    _, final_params = train_eval(Z, y, Z, y, best_H, C, rng)  # validation unused

    # pack weights (same format as label-linear example)
    off, T = 0, []
    W = best_H if best_H else C
    for E in enc:
        T.append((E.astype(np.float32) @ final_params[0][off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]
    out = struct.pack("<4i", 0x54424C31, X.shape[1], best_H, C) + np.stack(T).astype("<f4").tobytes() + final_params[1].astype("<f4").tobytes()
    if best_H:
        out += final_params[2].astype("<f4").tobytes() + final_params[3].astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* Table family (linear, or one ReLU layer) for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins). */
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
    if (nw < 16 || hd[0] != 0x54424C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], W = H ? H : C;
    if (C < 1 || C > 256 || nw != 16 + 4L * ((long)d * 256 * W + W + (H ? (long)H * C + C : 0))) {
        fprintf(stderr, "bad weights size\n"); return 3;
    }
    const float *T = (const float *)(hd + 4), *b1 = T + (long)d * 256 * W, *W2 = b1 + W, *b2 = W2 + (long)H * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W), *s = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int w = 0; w < W; w++) acc[w] = b1[w];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }
        const float *sc = acc;
        if (H) {
            for (int c = 0; c < C; c++) s[c] = b2[c];
            for (int u = 0; u < H; u++) {
                float h = acc[u] > 0.0f ? acc[u] : 0.0f;
                for (int c = 0; c < C; c++) s[c] += h * W2[(long)u * C + c];
            }
            sc = s;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```