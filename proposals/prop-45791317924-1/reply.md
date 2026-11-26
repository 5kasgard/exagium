```hypothesis
Family: label-mlp-tuned. A one-hidden-layer ReLU MLP with hidden size chosen from {4,8,12,16} by validation on the tuning split (last 20% of train.in). The small hidden layer captures pixel interactions missed by linear models, while inference remains fast (dot products + ReLU). If the best hidden size yields ≤0.5% accuracy gain over label-linear on the tuning split, the family does not justify its complexity.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
CANDIDATE_H = [0, 4, 8, 12, 16]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def compute_encoding(X, types):
    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)), dtype=np.float64)
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)
    return enc

def encode_data(X, enc):
    return np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)

def init_params(k, H, C, seed):
    rng = np.random.default_rng(seed)
    if H:
        W1 = (rng.uniform(-1, 1, (k, H)) * math.sqrt(6.0 / k)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        return [W1, b1, W2, b2]
    else:
        W1 = np.zeros((k, C), np.float32)
        b1 = np.zeros(C, np.float32)
        return [W1, b1]

def softmax_grad(logits, y):
    S = logits - logits.max(axis=1, keepdims=True)
    P = np.exp(S)
    P /= P.sum(axis=1, keepdims=True)
    P[np.arange(len(y)), y] -= 1.0
    return (P / len(y)).astype(np.float32)

def train_loop(Z, y, H, C, seed, epochs, batch, lr, l2):
    n, k = Z.shape
    rng = np.random.default_rng(seed)
    params = init_params(k, H, C, seed)
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            z, yy = Z[idx], y[idx]
            if H:
                a = z @ params[0] + params[1]
                h = np.maximum(a, 0)
                logits = h @ params[2] + params[3]
                g = softmax_grad(logits, yy)
                gh = (g @ params[2].T) * (a > 0)
                grads = [
                    z.T @ gh + l2 * params[0],
                    gh.sum(0),
                    h.T @ g + l2 * params[2],
                    g.sum(0)
                ]
            else:
                logits = z @ params[0] + params[1]
                g = softmax_grad(logits, yy)
                grads = [
                    z.T @ g + l2 * params[0],
                    g.sum(0)
                ]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return params

def accuracy(Z, y, params, H):
    if H:
        a = Z @ params[0] + params[1]
        h = np.maximum(a, 0)
        logits = h @ params[2] + params[3]
    else:
        logits = Z @ params[0] + params[1]
    return (logits.argmax(axis=1) == y).mean()

def write_weights(card, enc, params, H, C, out_path):
    d = card["input"]["bytes"]
    W = H if H else C
    T_list = []
    off = 0
    for E in enc:
        E_f32 = E.astype(np.float32)
        W1_slice = params[0][off:off + E.shape[1]]
        Tj = E_f32 @ W1_slice
        T_list.append(Tj)
        off += E.shape[1]
    T = np.stack(T_list)
    b1 = params[1].astype(np.float32)
    out = struct.pack("<4i", 0x54424C31, d, H, C)
    out += T.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if H:
        out += params[2].astype(np.float32).tobytes()
        out += params[3].astype(np.float32).tobytes()
    with open(out_path, "wb") as f:
        f.write(out)

def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    types = card["input"]["types"]
    n = len(X)
    val_size = n // 5
    split = n - val_size
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    best_H = None
    best_acc = -1.0
    for H in CANDIDATE_H:
        enc = compute_encoding(X_train, types)
        Z_train = encode_data(X_train, enc)
        Z_val = encode_data(X_val, enc)
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))
        params = train_loop(Z_train, y_train, H, C, seed=100+H, epochs=epochs, batch=BATCH, lr=LR, l2=L2)
        acc = accuracy(Z_val, y_val, params, H)
        print(f"H={H}, val_acc={acc:.4f}", file=sys.stderr)
        if acc > best_acc or (acc == best_acc and (best_H is None or H < best_H)):
            best_acc = acc
            best_H = H

    print(f"Selected H={best_H} with val_acc={best_acc:.4f}", file=sys.stderr)

    enc_full = compute_encoding(X, types)
    Z_full = encode_data(X, enc_full)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    final_params = train_loop(Z_full, y, best_H, C, seed=SEED, epochs=epochs, batch=BATCH, lr=LR, l2=L2)
    write_weights(card, enc_full, final_params, best_H, C, sys.argv[4])

if __name__ == "__main__":
    main()
```

```c
/* Table family (linear, or one ReLU layer) for label tasks: ./infer weights.bin in.rec N
 * Each input record is d bytes; output is one byte, the argmax class (first max wins). Format: see train.py. */
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

/* next record of a record file: 4-byte little-endian length, then the bytes. */
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