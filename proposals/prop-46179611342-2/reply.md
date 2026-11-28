```hypothesis
Family: label-embed-mlp. Learns per-position byte embeddings (size E), concatenates them, then a ReLU hidden layer (size H) and linear output. Embeddings capture character similarities; concatenation preserves position order. This should beat label-linear (0.3375) by modeling interactions with few parameters (E=32, H=64 ~ 0.5M params) and stay faster than label-gbdt. Failure: if dev accuracy ≤ 0.34 or inference cost > 20k instructions/item.
```

```python
"""label-embed-mlp: per-position byte embeddings + 1-hidden-layer MLP for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy. Chooses (E,H) on last 20% of train.in, then trains on all.
Weights format: magic 0x454D4244, d, E, H, C; float32 embed[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH = 3e-3, 1e-4, 64
SAMPLES, MAX_EPOCHS = 600000, 100
GRID_EPOCHS = 5
E_CHOICES = (16, 32)
H_CHOICES = (32, 64)


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def accuracy(params, X, y, d, E, H, C):
    embed, W1, b1, W2, b2 = params
    n = X.shape[0]
    correct = 0
    for s in range(0, n, 1024):
        idx = slice(s, min(s + 1024, n))
        xb, yb = X[idx], y[idx]
        concat = np.zeros((len(xb), d * E), np.float32)
        for j in range(d):
            concat[:, j*E:(j+1)*E] = embed[j, xb[:, j]]
        h = np.maximum(concat @ W1 + b1, 0)
        logits = h @ W2 + b2
        pred = logits.argmax(1)
        correct += (pred == yb).sum()
    return correct / n


def train_loop(params, state, X, y, d, E, H, C, epochs, rng):
    embed, W1, b1, W2, b2 = params
    m_embed, v_embed, m_W1, v_W1, m_b1, v_b1, m_W2, v_W2, m_b2, v_b2 = state
    n = X.shape[0]
    t = 0
    concat = np.zeros((BATCH, d * E), np.float32)
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            if len(idx) == 0:
                continue
            xb, yb = X[idx], y[idx]
            bsz = len(xb)
            # forward
            concat[:bsz] = 0
            for j in range(d):
                concat[:bsz, j*E:(j+1)*E] = embed[j, xb[:, j]]
            h_pre = concat[:bsz] @ W1 + b1
            h = np.maximum(h_pre, 0)
            logits = h @ W2 + b2
            # loss gradient
            logits -= logits.max(1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(1, keepdims=True)
            probs[np.arange(bsz), yb] -= 1.0
            probs /= bsz
            # backward
            dW2 = h.T @ probs + L2 * W2
            db2 = probs.sum(0)
            dh = probs @ W2.T
            dh[h_pre <= 0] = 0
            dW1 = concat[:bsz].T @ dh + L2 * W1
            db1 = dh.sum(0)
            dconcat = dh @ W1.T
            dembed = np.zeros_like(embed)
            for j in range(d):
                np.add.at(dembed[j], xb[:, j], dconcat[:bsz, j*E:(j+1)*E])
            # Adam updates
            t += 1
            beta1, beta2 = 0.9, 0.999
            for param, grad, m, v in [
                (embed, dembed, m_embed, v_embed),
                (W1, dW1, m_W1, v_W1),
                (b1, db1, m_b1, v_b1),
                (W2, dW2, m_W2, v_W2),
                (b2, db2, m_b2, v_b2),
            ]:
                m *= beta1; m += (1 - beta1) * grad
                v *= beta2; v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d, C = X.shape[1], card["classes"]
    n = X.shape[0]
    split = int(0.8 * n)
    idx = np.arange(n)
    train_idx, val_idx = idx[:split], idx[split:]

    rng = np.random.default_rng(SEED)
    best_acc, best_params, best_state, best_E, best_H = -1, None, None, 0, 0

    for E in E_CHOICES:
        for H in H_CHOICES:
            # init
            embed = (rng.normal(0, 0.01, (d, 256, E))).astype(np.float32)
            W1 = (rng.normal(0, math.sqrt(2.0 / (d * E)), (d * E, H))).astype(np.float32)
            b1 = np.zeros(H, np.float32)
            W2 = (rng.normal(0, math.sqrt(2.0 / H), (H, C))).astype(np.float32)
            b2 = np.zeros(C, np.float32)
            params = (embed, W1, b1, W2, b2)
            state = tuple((np.zeros_like(p), np.zeros_like(p)) for p in params)
            # quick train on train split
            train_loop(params, state, X[train_idx], y[train_idx], d, E, H, C, GRID_EPOCHS, rng)
            acc = accuracy(params, X[val_idx], y[val_idx], d, E, H, C)
            if acc > best_acc:
                best_acc, best_params, best_state, best_E, best_H = acc, params, state, E, H

    # retrain on all data
    E, H = best_E, best_H
    embed, W1, b1, W2, b2 = best_params
    state = best_state
    train_loop((embed, W1, b1, W2, b2), state, X, y, d, E, H, C,
               min(MAX_EPOCHS, math.ceil(SAMPLES / n)), rng)

    # pack weights
    magic, d, E, H, C = 0x454D4244, d, E, H, C
    header = struct.pack("<5i", magic, d, E, H, C)
    body = (embed.astype("<f4").tobytes() +
            W1.astype("<f4").tobytes() + b1.astype("<f4").tobytes() +
            W2.astype("<f4").tobytes() + b2.astype("<f4").tobytes())
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```

```c
/* label-embed-mlp inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C < 1 || C > 256) { fprintf(stderr, "bad header dims\n"); return 3; }
    long expected = 20L + 4L * ((long)d * 256 * E + (long)d * E * H + H + (long)H * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *embed = (const float *)(hd + 5);
    const float *W1 = embed + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * H;
    const float *W2 = b1 + H;
    const float *b2 = W2 + (long)H * C;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * E);
    float *hidden = malloc(sizeof(float) * H);
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *src = embed + ((long)j * 256 + xi[j]) * E;
            float *dst = concat + j * E;
            memcpy(dst, src, E * sizeof(float));
        }

        // hidden layer: hidden = ReLU(concat @ W1 + b1)
        for (int h = 0; h < H; h++) hidden[h] = b1[h];
        for (int k_idx = 0; k_idx < d * E; k_idx++) {
            float ck = concat[k_idx];
            const float *w = W1 + (long)k_idx * H;
            for (int h = 0; h < H; h++) hidden[h] += ck * w[h];
        }
        for (int h = 0; h < H; h++) if (hidden[h] < 0.0f) hidden[h] = 0.0f;

        // output layer: scores = hidden @ W2 + b2
        for (int c = 0; c < C; c++) scores[c] = b2[c];
        for (int h = 0; h < H; h++) {
            float hh = hidden[h];
            const float *w = W2 + (long)h * C;
            for (int c = 0; c < C; c++) scores[c] += hh * w[c];
        }

        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(concat); free(hidden); free(scores);
    if (x) free(x);
    return 0;
}
```