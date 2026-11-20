```hypothesis
Family: label-embedding-mlp – learned per-position byte embeddings (E=8) concatenated, one ReLU hidden layer (H=64), then linear to 65 classes.
Why: Embeddings reduce the 4096 one-hot dimensions to 128 dense features, allowing the hidden layer to model pairwise position interactions more efficiently than a linear model. The small E and H keep inference fast (≈2× label-linear instructions) while adding non-linearity.
Failure: If dev accuracy ≤ 0.3375 (label-linear) or instructions/item > 15000, the model does not justify its complexity.
```

```python
"""label-embedding-mlp: per-position embeddings + one ReLU hidden layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Hyperparameters chosen on last 20% of train.in (file order), then retrained on all train.
Weights format: magic 0x454D424C, d, E, H, C; float32 embeddings[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, BETA1, BETA2, EPS, L2 = 3e-3, 0.9, 0.999, 1e-8, 1e-4
BATCH, SAMPLES, MAX_EPOCHS = 64, 600000, 100
# Hyperparameter grid for tuning split
E_CHOICES = [8, 16]
H_CHOICES = [32, 64, 128]


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def init_params(d, E, H, C, rng):
    # embeddings: d x 256 x E
    emb = rng.normal(0, 0.01, (d, 256, E)).astype(np.float32)
    # W1: (d*E) x H, He init for ReLU
    fan_in = d * E
    W1 = rng.normal(0, math.sqrt(2.0 / fan_in), (fan_in, H)).astype(np.float32)
    b1 = np.zeros(H, np.float32)
    # W2: H x C, Xavier
    W2 = rng.normal(0, math.sqrt(6.0 / (H + C)), (H, C)).astype(np.float32)
    b2 = np.zeros(C, np.float32)
    return emb, W1, b1, W2, b2


def forward(emb, W1, b1, W2, b2, X_idx):
    # X_idx: (n, d) uint8
    n, d = X_idx.shape
    E = emb.shape[2]
    # gather embeddings
    flat_emb = emb.reshape(d * 256, E)
    flat_idx = (np.arange(d) * 256 + X_idx.T).T  # (n, d)
    x = flat_emb[flat_idx].reshape(n, d * E)  # (n, d*E)
    h = x @ W1 + b1
    h = np.maximum(h, 0)
    logits = h @ W2 + b2
    return logits, h, x


def loss_and_grad(logits, y, h, x, W1, W2, emb, L2):
    n = logits.shape[0]
    # softmax + cross-entropy gradient
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(n), y] -= 1.0
    probs /= n
    # output layer grads
    gW2 = h.T @ probs + L2 * W2
    gb2 = probs.sum(axis=0)
    # hidden layer grads
    gh = probs @ W2.T
    gh[h <= 0] = 0.0
    gW1 = x.T @ gh + L2 * W1
    gb1 = gh.sum(axis=0)
    # embedding grads
    d, E256, E = emb.shape
    g_emb = np.zeros_like(emb)
    # gh shape (n, H), W1 shape (d*E, H) -> dx = gh @ W1.T (n, d*E)
    dx = gh @ W1.T  # (n, d*E)
    dx = dx.reshape(n, d, E)
    # accumulate per position and byte value
    for j in range(d):
        idx = X_idx[:, j]  # (n,)
        for c in range(256):
            mask = (idx == c)
            if mask.any():
                g_emb[j, c] = dx[mask, j].sum(axis=0) + L2 * emb[j, c]
    loss = -np.log(probs[np.arange(n), y] + 1e-12).mean()
    return loss, g_emb, gW1, gb1, gW2, gb2


def train_loop(X_tr, y_tr, X_val, y_val, d, E, H, C, rng, epochs):
    emb, W1, b1, W2, b2 = init_params(d, E, H, C, rng)
    params = [emb, W1, b1, W2, b2]
    shapes = [p.shape for p in params]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    best_val_acc = -1.0
    best_params = None
    n_tr = X_tr.shape[0]
    for epoch in range(epochs):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X_tr[idx], y_tr[idx]
            logits, h, x = forward(emb, W1, b1, W2, b2, Xb)
            loss, g_emb, gW1, gb1, gW2, gb2 = loss_and_grad(logits, yb, h, x, W1, W2, emb, L2)
            grads = [g_emb, gW1, gb1, gW2, gb2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = BETA1 * m[i] + (1 - BETA1) * gr
                v[i] = BETA2 * v[i] + (1 - BETA2) * (gr * gr)
                m_hat = m[i] / (1 - BETA1 ** t)
                v_hat = v[i] / (1 - BETA2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + EPS)
        # validation accuracy
        logits_val, _, _ = forward(emb, W1, b1, W2, b2, X_val)
        pred = logits_val.argmax(axis=1)
        val_acc = (pred == y_val).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = [p.copy() for p in params]
    return best_params, best_val_acc


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    val_size = n // 5
    X_tr, y_tr = X[:-val_size], y[:-val_size]
    X_val, y_val = X[-val_size:], y[-val_size:]

    rng = np.random.default_rng(SEED)
    # tuning epochs: enough to see difference but not too long
    tune_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / max(1, X_tr.shape[0])))
    best_score = -1
    best_cfg = None
    best_params = None
    for E in E_CHOICES:
        for H in H_CHOICES:
            params, val_acc = train_loop(X_tr, y_tr, X_val, y_val, d, E, H, C, rng, tune_epochs)
            if val_acc > best_score:
                best_score = val_acc
                best_cfg = (E, H)
                best_params = params
    print(f"chosen E={best_cfg[0]}, H={best_cfg[1]}, val_acc={best_score:.4f}", file=sys.stderr)

    # retrain on full data with best config
    E, H = best_cfg
    full_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    emb, W1, b1, W2, b2 = init_params(d, E, H, C, rng)
    params = [emb, W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    for epoch in range(full_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            Xb, yb = X[idx], y[idx]
            logits, h, x = forward(emb, W1, b1, W2, b2, Xb)
            loss, g_emb, gW1, gb1, gW2, gb2 = loss_and_grad(logits, yb, h, x, W1, W2, emb, L2)
            grads = [g_emb, gW1, gb1, gW2, gb2]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = BETA1 * m[i] + (1 - BETA1) * gr
                v[i] = BETA2 * v[i] + (1 - BETA2) * (gr * gr)
                m_hat = m[i] / (1 - BETA1 ** t)
                v_hat = v[i] / (1 - BETA2 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + EPS)

    # save weights
    emb, W1, b1, W2, b2 = params
    magic = 0x454D424C  # 'LBME'
    header = struct.pack("<5I", magic, d, E, H, C)
    out = header
    out += emb.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    out += W2.astype("<f4").tobytes()
    out += b2.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```

```c
/* label-embedding-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index 0..64).
 * Weights format: magic 0x454D424C, d, E, H, C; float32 embeddings[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]. */
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
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 20
        + (long)d * 256 * E * 4
        + (long)d * E * H * 4
        + (long)H * 4
        + (long)H * C * 4
        + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *hidden = (float *)malloc(sizeof(float) * H);
    float *scores = (float *)malloc(sizeof(float) * C);
    float *concat = (float *)malloc(sizeof(float) * d * E);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(concat + j * E, e, E * sizeof(float));
        }

        // hidden layer: concat @ W1 + b1, ReLU
        for (int h = 0; h < H; h++) {
            float sum = b1[h];
            for (int f = 0; f < d * E; f++) sum += concat[f] * W1[(long)f * H + h];
            hidden[h] = sum > 0.0f ? sum : 0.0f;
        }

        // output layer: hidden @ W2 + b2
        for (int c = 0; c < C; c++) {
            float sum = b2[c];
            for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
            scores[c] = sum;
        }

        // argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(hidden); free(scores); free(concat);
    return 0;
}
```