```hypothesis
We propose "label-fm", a Factorization Machine family that adds low-rank pairwise interactions to a linear model. Adult's mixed ordinal/categorical features benefit from interactions (e.g., education×occupation) which FM captures with O(d_emb·n_features) inference cost, keeping CPU low. We reuse label-linear's one-hot/z-score encoding, train with Adam on logistic loss, and select embedding size (2/4/8) and learning rate on the 20% tuning split. If dev accuracy does not exceed label-linear (0.8524) or inference cost exceeds label-mlp16 (2325 Ir/item), the hypothesis fails.
```
```python
import json, math, struct, sys

import numpy as np

SEED = 1
GRID = [
    {'d_emb': 2, 'lr': 0.01, 'l2': 1e-4},
    {'d_emb': 2, 'lr': 0.03, 'l2': 1e-4},
    {'d_emb': 4, 'lr': 0.01, 'l2': 1e-4},
    {'d_emb': 4, 'lr': 0.03, 'l2': 1e-4},
    {'d_emb': 8, 'lr': 0.01, 'l2': 1e-4},
    {'d_emb': 8, 'lr': 0.03, 'l2': 1e-4},
]
BATCH = 64
EPOCHS = 20


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def encode_data(X, types, enc_params):
    n = X.shape[0]
    k_total = sum(ep['k'] for ep in enc_params)
    Z = np.zeros((n, k_total), dtype=np.float32)
    off = 0
    for j, (t, ep) in enumerate(zip(types, enc_params)):
        if t == "ordinal":
            mu, sd = ep['mu'], ep['sd']
            Z[:, off] = (X[:, j].astype(np.float32) - mu) / sd
            off += 1
        else:
            map_arr = np.full(256, -1, dtype=np.int32)
            for val, idx in ep['map'].items():
                map_arr[val] = idx
            idxs = map_arr[X[:, j]]
            valid = idxs >= 0
            Z[np.arange(n)[valid], off + idxs[valid]] = 1.0
            off += ep['k']
    return Z


def train_fm(X_enc, y, d_emb, lr, l2, epochs, batch, rng):
    n, k = X_enc.shape
    bias = 0.0
    w = np.zeros(k, dtype=np.float32)
    V = rng.normal(0, 0.01, size=(k, d_emb)).astype(np.float32)

    m_bias, v_bias = 0.0, 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_V, v_V = np.zeros_like(V), np.zeros_like(V)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            x = X_enc[idx]
            yy = y[idx]

            linear = x @ w
            sum_vx = x @ V
            sum_v2x2 = (x * x) @ (V * V)
            fm = 0.5 * np.sum(sum_vx * sum_vx - sum_v2x2, axis=1)
            logits = bias + linear + fm

            p = 1.0 / (1.0 + np.exp(-logits))
            grad = p - yy

            g_bias = np.mean(grad)
            g_w = (x.T @ grad) / batch + l2 * w
            g_V = (x.T @ sum_vx - (x * x).T @ V) / batch + l2 * V

            t += 1
            m_bias = beta1 * m_bias + (1 - beta1) * g_bias
            v_bias = beta2 * v_bias + (1 - beta2) * g_bias * g_bias
            bias -= lr * (m_bias / (1 - beta1 ** t)) / (np.sqrt(v_bias / (1 - beta2 ** t)) + eps)

            m_w = beta1 * m_w + (1 - beta1) * g_w
            v_w = beta2 * v_w + (1 - beta2) * g_w * g_w
            w -= lr * (m_w / (1 - beta1 ** t)) / (np.sqrt(v_w / (1 - beta2 ** t)) + eps)

            m_V = beta1 * m_V + (1 - beta1) * g_V
            v_V = beta2 * v_V + (1 - beta2) * g_V * g_V
            V -= lr * (m_V / (1 - beta1 ** t)) / (np.sqrt(v_V / (1 - beta2 ** t)) + eps)

    return bias, w, V


def evaluate(bias, w, V, X_enc, y):
    linear = X_enc @ w
    sum_vx = X_enc @ V
    sum_v2x2 = (X_enc * X_enc) @ (V * V)
    fm = 0.5 * np.sum(sum_vx * sum_vx - sum_v2x2, axis=1)
    logits = bias + linear + fm
    preds = (logits > 0).astype(np.int64)
    return np.mean(preds == y)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    types = card["input"]["types"]
    n = X.shape[0]
    val_start = int(0.8 * n)

    X_train, y_train = X[:val_start], y[:val_start]
    X_val, y_val = X[val_start:], y[val_start:]

    enc_params = []
    for j, t in enumerate(types):
        if t == "ordinal":
            v = X_train[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            enc_params.append({'type': 'ordinal', 'mu': mu, 'sd': sd, 'k': 1})
        else:
            vals = np.unique(X_train[:, j])
            mp = {val: i for i, val in enumerate(vals)}
            enc_params.append({'type': 'categorical', 'vals': vals, 'map': mp, 'k': len(vals)})

    X_train_enc = encode_data(X_train, types, enc_params)
    X_val_enc = encode_data(X_val, types, enc_params)

    rng = np.random.default_rng(SEED)
    best_acc = -1.0
    best_params = None
    for params in GRID:
        bias, w, V = train_fm(X_train_enc, y_train, params['d_emb'], params['lr'], params['l2'], EPOCHS, BATCH, rng)
        acc = evaluate(bias, w, V, X_val_enc, y_val)
        if acc > best_acc:
            best_acc = acc
            best_params = params
        print(f"d_emb={params['d_emb']}, lr={params['lr']}, l2={params['l2']}: val_acc={acc:.4f}")

    print(f"Best: {best_params}, acc={best_acc:.4f}")

    enc_params_full = []
    for j, t in enumerate(types):
        if t == "ordinal":
            v = X[:, j].astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            enc_params_full.append({'type': 'ordinal', 'mu': mu, 'sd': sd, 'k': 1})
        else:
            vals = np.unique(X[:, j])
            mp = {val: i for i, val in enumerate(vals)}
            enc_params_full.append({'type': 'categorical', 'vals': vals, 'map': mp, 'k': len(vals)})

    X_full_enc = encode_data(X, types, enc_params_full)
    rng_full = np.random.default_rng(SEED)
    bias, w, V = train_fm(X_full_enc, y, best_params['d_emb'], best_params['lr'], best_params['l2'], EPOCHS, BATCH, rng_full)

    offsets = []
    off = 0
    for ep in enc_params_full:
        offsets.append((off, off + ep['k']))
        off += ep['k']

    d_emb = best_params['d_emb']
    linear_table = np.zeros((d, 256), dtype=np.float32)
    v_contrib = np.zeros((d, 256, d_emb), dtype=np.float32)
    v2_contrib = np.zeros((d, 256, d_emb), dtype=np.float32)

    for j in range(d):
        ep = enc_params_full[j]
        start, end = offsets[j]
        w_slice = w[start:end]
        V_slice = V[start:end, :]
        if ep['type'] == 'ordinal':
            mu, sd = ep['mu'], ep['sd']
            for val in range(256):
                x = (val - mu) / sd
                linear_table[j, val] = x * w_slice[0]
                for f in range(d_emb):
                    v_contrib[j, val, f] = x * V_slice[0, f]
                    v2_contrib[j, val, f] = (x * x) * (V_slice[0, f] * V_slice[0, f])
        else:
            for val, idx in ep['map'].items():
                linear_table[j, val] = w_slice[idx]
                for f in range(d_emb):
                    v_contrib[j, val, f] = V_slice[idx, f]
                    v2_contrib[j, val, f] = V_slice[idx, f] * V_slice[idx, f]

    magic = 0x464D4C31
    header = struct.pack("<4i", magic, d, d_emb, 2)
    bias_bytes = struct.pack("<f", bias)
    linear_bytes = linear_table.astype("<f4").tobytes()
    v_bytes = v_contrib.astype("<f4").tobytes()
    v2_bytes = v2_contrib.astype("<f4").tobytes()
    out = header + bias_bytes + linear_bytes + v_bytes + v2_bytes
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
    long nw, nx = 0, N = atol(argv[3]), pos = 