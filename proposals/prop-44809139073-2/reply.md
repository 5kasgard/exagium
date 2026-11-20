```hypothesis
We propose "label-emb", a family that learns per-position byte embeddings followed by a small MLP. This captures character n-gram statistics more efficiently than one-hot linear models, and should outperform label-linear on this language modeling task. If the embeddings fail to capture sufficient context, accuracy will not exceed label-linear.
```

```python
import json, math, struct, sys
import numpy as np

SEED = 1
D_CHOICES = [16, 32, 64]
H_CHOICES = [0, 16, 32]
LR = 3e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 50
SEARCH_EPOCHS = 10

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
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    n = X_bytes.shape[0]

    val_size = n // 5
    train_size = n - val_size
    X_train, X_val = X_bytes[:train_size], X_bytes[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    rng = np.random.default_rng(SEED)

    best_acc = -1.0
    best_params = None
    best_weights = None

    for D in D_CHOICES:
        for H in H_CHOICES:
            scale_emb = math.sqrt(6.0 / (256 + D))
            embed = rng.uniform(-scale_emb, scale_emb, (d, 256, D)).astype(np.float32)
            if H > 0:
                W1 = rng.uniform(-math.sqrt(6.0/(d*D)), math.sqrt(6.0/(d*D)), (d*D, H)).astype(np.float32)
                b1 = np.zeros(H, np.float32)
                W2 = rng.uniform(-math.sqrt(6.0/H), math.sqrt(6.0/H), (H, C)).astype(np.float32)
                b2 = np.zeros(C, np.float32)
                params = [embed, W1, b1, W2, b2]
            else:
                W = rng.uniform(-math.sqrt(6.0/(d*D)), math.sqrt(6.0/(d*D)), (d*D, C)).astype(np.float32)
                b = np.zeros(C, np.float32)
                params = [embed, W, b]

            state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
            t = 0

            for epoch in range(SEARCH_EPOCHS):
                order = rng.permutation(train_size)
                for s in range(0, train_size, BATCH):
                    idx = order[s:s+BATCH]
                    xb, yb = X_train[idx], y_train[idx]
                    batch_size = len(idx)
                    emb = np.zeros((batch_size, d*D), np.float32)
                    for j in range(d):
                        emb[:, j*D:(j+1)*D] = params[0][j, xb[:, j]]
                    if H > 0:
                        hidden = np.maximum(emb @ params[1] + params[2], 0)
                        logits = hidden @ params[3] + params[4]
                    else:
                        logits = emb @ params[1] + params[2]

                    logits -= logits.max(axis=1, keepdims=True)
                    exp_logits = np.exp(logits)
                    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                    grad_logits = probs
                    grad_logits[np.arange(batch_size), yb] -= 1.0
                    grad_logits /= batch_size

                    if H > 0:
                        grad_W2 = hidden.T @ grad_logits
                        grad_b2 = grad_logits.sum(axis=0)
                        grad_hidden = grad_logits @ params[3].T
                        grad_hidden *= (hidden > 0)
                        grad_W1 = emb.T @ grad_hidden
                        grad_b1 = grad_hidden.sum(axis=0)
                        grad_emb = grad_hidden @ params[1].T
                        grad_W2 += L2 * params[3]
                        grad_W1 += L2 * params[1]
                        grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
                    else:
                        grad_W = emb.T @ grad_logits
                        grad_b = grad_logits.sum(axis=0)
                        grad_emb = grad_logits @ params[1].T
                        grad_W += L2 * params[1]
                        grads = [grad_emb, grad_W, grad_b]

                    t += 1
                    for i, (p, gr) in enumerate(zip(params, grads)):
                        m, v = state[i]
                        m *= 0.9; m += 0.1 * gr
                        v *= 0.999; v += 0.001 * gr * gr
                        m_hat = m / (1 - 0.9**t)
                        v_hat = v / (1 - 0.999**t)
                        p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

            val_emb = np.zeros((val_size, d*D), np.float32)
            for j in range(d):
                val_emb[:, j*D:(j+1)*D] = params[0][j, X_val[:, j]]
            if H > 0:
                val_hidden = np.maximum(val_emb @ params[1] + params[2], 0)
                val_logits = val_hidden @ params[3] + params[4]
            else:
                val_logits = val_emb @ params[1] + params[2]
            val_pred = val_logits.argmax(axis=1)
            acc = (val_pred == y_val).mean()
            if acc > best_acc:
                best_acc = acc
                best_params = (D, H)
                best_weights = [p.copy() for p in params]

    D, H = best_params
    params = best_weights
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    for epoch in range(MAX_EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X_bytes[idx], y[idx]
            batch_size = len(idx)
            emb = np.zeros((batch_size, d*D), np.float32)
            for j in range(d):
                emb[:, j*D:(j+1)*D] = params[0][j, xb[:, j]]
            if H > 0:
                hidden = np.maximum(emb @ params[1] + params[2], 0)
                logits = hidden @ params[3] + params[4]
            else:
                logits = emb @ params[1] + params[2]

            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            grad_logits = probs
            grad_logits[np.arange(batch_size), yb] -= 1.0
            grad_logits /= batch_size

            if H > 0:
                grad_W2 = hidden.T @ grad_logits
                grad_b2 = grad_logits.sum(axis=0)
                grad_hidden = grad_logits @ params[3].T
                grad_hidden *= (hidden > 0)
                grad_W1 = emb.T @ grad_hidden
                grad_b1 = grad_hidden.sum(axis=0)
                grad_emb = grad_hidden @ params[1].T
                grad_W2 += L2 * params[3]
                grad_W1 += L2 * params[1]
                grads = [grad_emb, grad_W1, grad_b1, grad_W2, grad_b2]
            else:
                grad_W = emb.T @ grad_logits
                grad_b = grad_logits.sum(axis=0)
                grad_emb = grad_logits @ params[1].T
                grad_W += L2 * params[1]
                grads = [grad_emb, grad_W, grad_b]

            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9**t)
                v_hat = v / (1 - 0.999**t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    magic = 0x4C424D45
    header = struct.pack("<5i", magic, d, D, C, H)
    out = header
    out += params[0].astype("<f4").tobytes()
    if H > 0:
        out += params[1].astype("<f4").tobytes()
        out += params[2].astype("<f4").tobytes()
        out += params[3].astype("<f4").tobytes()
        out += params[4].astype("<f4").tobytes()
    else:
        out += params[1].astype("<f4").tobytes()
        out += params[2].astype("<f4").tobytes()
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
    if (nw < 20 || hd[0] != 0x4C424D45) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], C = hd[3], H = hd[4];
    if (d != 16 || C != 65 || D <= 0 || H < 0) { fprintf(stderr, "bad header values\n"); return 3; }
    int input_dim = d * D;
    long expected = 20;
    expected += (long)d * 256 * D * 4;
    if (H > 0) {
        expected += (long)input_dim * H * 4;
        expected += (long)H * 4;
        expected += (long)H * C * 4;
        expected += (long)C * 4;
    } else {
        expected += (long)input_dim * C * 4;
        expected += (long)C * 4;
    }
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *embed = (const float *)(hd + 5);
    const float *W1 = embed + (long)d * 256 * D;
    const float *b1 = W1 + (H > 0 ? (long)input_dim * H : 0);
    const float *W2 = b1 + (H > 0 ? H : 0);
    const float *b2 = W2 + (H > 0 ? (long)H * C : 0);
    const float *W = W1;
    const float *b = b1;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    float *emb = malloc(sizeof(float) * input_dim);
    float *hidden = H > 0 ? malloc(sizeof(float) * H) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int j = 0; j < d; j++) {
            const float *e = embed + ((long)j * 256 + xi[j]) * D;
            memcpy(emb + j * D, e, D * sizeof(float));
        }
        if (H > 0) {
            for (int h = 0; h < H; h++) {
                float sum = b1[h];
                for (int idx = 0; idx < input_dim; idx++) {
                    sum += emb[idx] * W1[(long)idx * H + h];
                }
                hidden[h] = sum > 0 ? sum : 0;
            }
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int h = 0; h < H; h++) {
                    sum += hidden[h] * W2[(long)h * C + c];
                }
                scores[c] = sum;
            }
        } else {
            for (int c = 0; c < C; c++) {
                float sum = b[c];
                for (int idx = 0; idx < input_dim; idx++) {
                    sum += emb[idx] * W[(long)idx * C + c];
                }
                scores[c] = sum;
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) {
            if (scores[c] > scores[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```