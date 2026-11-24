```hypothesis
Family: label-emb (position-specific byte embeddings + small MLP).
Why: Character-level language modeling benefits from distributed representations of bytes (e.g., letters share structure). A linear one-hot model cannot capture similarities between bytes; embeddings allow generalization. A small ReLU layer adds non-linear interactions between positions.
Failure condition: If dev accuracy ≤ 0.3375 (label-linear) or inference cost > 10k Ir/item, the family is not worthwhile.
```

```python
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import json, math, struct, sys
import numpy as np

SEED = 1
EMBED_DIMS = [16, 32]
HIDDEN_SIZES = [0, 64]
LR = 0.01
L2 = 1e-4
BATCH = 64
TUNE_EPOCHS = 5
FINAL_EPOCHS = 20
SAMPLES_PER_EPOCH = 600000

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
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X_all.shape[0]
    split = int(0.8 * n)
    X_train, X_tune = X_all[:split], X_all[split:]
    y_train, y_tune = y_all[:split], y_all[split:]

    best_acc = -1.0
    best_config = None
    best_params = None

    for E in EMBED_DIMS:
        for H in HIDDEN_SIZES:
            rng = np.random.default_rng(SEED + E*100 + H)
            emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
            if H:
                W1 = (rng.uniform(-1, 1, (d*E, H)) * math.sqrt(6.0 / (d*E))).astype(np.float32)
                b1 = np.zeros(H, np.float32)
                W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
                b2 = np.zeros(C, np.float32)
                params = [emb, W1, b1, W2, b2]
            else:
                W1 = (rng.uniform(-1, 1, (d*E, C)) * math.sqrt(6.0 / (d*E))).astype(np.float32)
                b1 = np.zeros(C, np.float32)
                params = [emb, W1, b1]

            state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
            t = 0
            n_train = X_train.shape[0]
            max_epochs = min(TUNE_EPOCHS, math.ceil(SAMPLES_PER_EPOCH / n_train))
            for _ in range(max_epochs):
                order = rng.permutation(n_train)
                for s in range(0, n_train, BATCH):
                    idx = order[s:s+BATCH]
                    xb = X_train[idx]
                    yb = y_train[idx]
                    batch_size = xb.shape[0]
                    concat = np.zeros((batch_size, d*E), np.float32)
                    for j in range(d):
                        concat[:, j*E:(j+1)*E] = emb[j, xb[:, j], :]
                    if H:
                        a = concat @ W1 + b1
                        h = np.maximum(a, 0)
                        logits = h @ W2 + b2
                    else:
                        logits = concat @ W1 + b1
                    logits -= logits.max(axis=1, keepdims=True)
                    exp_logits = np.exp(logits)
                    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                    probs[np.arange(batch_size), yb] -= 1.0
                    grad_logits = (probs / batch_size).astype(np.float32)
                    if H:
                        grad_W2 = h.T @ grad_logits + L2 * W2
                        grad_b2 = grad_logits.sum(0)
                        grad_h = grad_logits @ W2.T
                        grad_h *= (a > 0)
                        grad_W1 = concat.T @ grad_h + L2 * W1
                        grad_b1 = grad_h.sum(0)
                        grad_concat = grad_h @ W1.T
                    else:
                        grad_W1 = concat.T @ grad_logits + L2 * W1
                        grad_b1 = grad_logits.sum(0)
                        grad_concat = grad_logits @ W1.T
                    grad_emb = np.zeros_like(emb)
                    for j in range(d):
                        np.add.at(grad_emb[j], xb[:, j], grad_concat[:, j*E:(j+1)*E])
                    grads = [grad_emb, grad_W1, grad_b1]
                    if H:
                        grads += [grad_W2, grad_b2]
                    t += 1
                    for i, (p, gr) in enumerate(zip(params, grads)):
                        m, v = state[i]
                        m *= 0.9; m += 0.1 * gr
                        v *= 0.999; v += 0.001 * gr * gr
                        m_hat = m / (1 - 0.9 ** t)
                        v_hat = v / (1 - 0.999 ** t)
                        p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

            # evaluate on tune
            batch_size = X_tune.shape[0]
            concat = np.zeros((batch_size, d*E), np.float32)
            for j in range(d):
                concat[:, j*E:(j+1)*E] = emb[j, X_tune[:, j], :]
            if H:
                a = concat @ W1 + b1
                h = np.maximum(a, 0)
                logits = h @ W2 + b2
            else:
                logits = concat @ W1 + b1
            preds = logits.argmax(axis=1)
            acc = (preds == y_tune).mean()
            if acc > best_acc:
                best_acc = acc
                best_config = (E, H)
                best_params = (emb.copy(), W1.copy(), b1.copy(),
                               W2.copy() if H else None, b2.copy() if H else None)

    print(f"Best config: E={best_config[0]}, H={best_config[1]}, tune_acc={best_acc:.4f}", file=sys.stderr)

    # retrain on all data
    E, H = best_config
    rng = np.random.default_rng(SEED)
    emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
    if H:
        W1 = (rng.uniform(-1, 1, (d*E, H)) * math.sqrt(6.0 / (d*E))).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [emb, W1, b1, W2, b2]
    else:
        W1 = (rng.uniform(-1, 1, (d*E, C)) * math.sqrt(6.0 / (d*E))).astype(np.float32)
        b1 = np.zeros(C, np.float32)
        params = [emb, W1, b1]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n_all = X_all.shape[0]
    max_epochs = min(FINAL_EPOCHS, math.ceil(SAMPLES_PER_EPOCH / n_all))
    for _ in range(max_epochs):
        order = rng.permutation(n_all)
        for s in range(0, n_all, BATCH):
            idx = order[s:s+BATCH]
            xb = X_all[idx]
            yb = y_all[idx]
            batch_size = xb.shape[0]
            concat = np.zeros((batch_size, d*E), np.float32)
            for j in range(d):
                concat[:, j*E:(j+1)*E] = emb[j, xb[:, j], :]
            if H:
                a = concat @ W1 + b1
                h = np.maximum(a, 0)
                logits = h @ W2 + b2
            else:
                logits = concat @ W1 + b1
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(batch_size), yb] -= 1.0
            grad_logits = (probs / batch_size).astype(np.float32)
            if H:
                grad_W2 = h.T @ grad_logits + L2 * W2
                grad_b2 = grad_logits.sum(0)
                grad_h = grad_logits @ W2.T
                grad_h *= (a > 0)
                grad_W1 = concat.T @ grad_h + L2 * W1
                grad_b1 = grad_h.sum(0)
                grad_concat = grad_h @ W1.T
            else:
                grad_W1 = concat.T @ grad_logits + L2 * W1
                grad_b1 = grad_logits.sum(0)
                grad_concat = grad_logits @ W1.T
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], xb[:, j], grad_concat[:, j*E:(j+1)*E])
            grads = [grad_emb, grad_W1, grad_b1]
            if H:
                grads += [grad_W2, grad_b2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # write weights.bin
    # magic 0x454D424C, d, E, H, C
    out = struct.pack("<5i", 0x454D424C, d, E, H, C)
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if H:
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad weights dims\n"); return 3; }

    long W1_cols = H ? H : C;
    long expected = 20;
    expected += (long)d * 256 * E * 4;
    expected += (long)d * E * W1_cols * 4;
    expected += W1_cols * 4;
    if (H) {
        expected += (long)H * C * 4;
        expected += C * 4;
    }
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * W1_cols;
    const float *W2 = NULL;
    const float *b2 = NULL;
    if (H) {
        W2 = b1 + H;
        b2 = W2 + (long)H * C;
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * E);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int j = 0; j < d * E; j++) concat[j] = 0.0f;
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            for (int kk = 0; kk < E; kk++) concat[j * E + kk] += e[kk];
        }

        if (H) {
            for (int h = 0; h < H; h++) {
                float sum = b1[h];
                for (int kk = 0; kk < d * E; kk++) sum += concat[kk] * W1[(long)kk * H + h];
                hidden[h] = sum > 0.0f ? sum : 0.0f;
            }
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
                scores[c] = sum;
            }
        } else {
            for (int c = 0; c < C; c++) {
                float sum = b1[c];
                for (int kk = 0; kk < d * E; kk++) sum += concat[kk] * W1[(long)kk * C + c];
                scores[c] = sum;
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```