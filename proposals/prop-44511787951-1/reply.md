```hypothesis
The label-embed family uses per-position byte embeddings (8-dim) followed by a 64-unit ReLU hidden layer and a linear output. This captures character similarities and positional context better than one-hot linear models, improving accuracy over label-linear (0.3375) while keeping inference cost low due to small embedding dimension and single hidden layer. If dev accuracy does not exceed 0.35, the approach fails.
```
```python
"""label-embed: per-position byte embeddings + one ReLU hidden layer for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only.
Weights format: int32 magic 0x454D4244, d, E, H, C; float32 embed[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C]."""
import json, math, struct, sys

import numpy as np

# Hyperparameters chosen on tuning split (last 20% of train.in)
EMBED_DIM = 8
HIDDEN_SIZE = 64
SEED = 1
LR = 1e-3
L2 = 1e-4
BATCH = 64
EPOCHS = 20

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
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65
    E = EMBED_DIM
    H = HIDDEN_SIZE
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize embeddings: (d, 256, E)
    embed = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / (256 + E))).astype(np.float32)

    # MLP weights
    if H:
        W1 = (rng.uniform(-1, 1, (d * E, H)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [embed, W1, b1, W2, b2]
    else:
        W1 = (rng.uniform(-1, 1, (d * E, C)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
        b1 = np.zeros(C, np.float32)
        params = [embed, W1, b1]

    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(X_batch):
        # X_batch: (B, d) uint8
        B = X_batch.shape[0]
        # Gather embeddings: (B, d, E)
        emb = embed[X_batch, np.arange(256)]  # This is wrong; need advanced indexing per position
        # Correct: for each position j, take embed[j, X_batch[:, j], :]
        emb_list = [embed[j, X_batch[:, j], :] for j in range(d)]
        emb = np.stack(emb_list, axis=1)  # (B, d, E)
        flat = emb.reshape(B, d * E)      # (B, d*E)
        if H:
            hidden = np.maximum(flat @ W1 + b1, 0)  # ReLU
            logits = hidden @ W2 + b2
        else:
            logits = flat @ W1 + b1
        return logits, flat, (emb if H else None)

    def backward(flat, hidden, logits, y_batch):
        B = len(y_batch)
        # Softmax cross-entropy gradient
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        probs[np.arange(B), y_batch] -= 1.0
        probs /= B  # (B, C)

        if H:
            # grad on W2, b2
            gW2 = hidden.T @ probs + L2 * W2
            gb2 = probs.sum(axis=0)
            # grad on hidden
            gh = probs @ W2.T
            gh *= (hidden > 0)  # ReLU mask
            # grad on W1, b1
            gW1 = flat.T @ gh + L2 * W1
            gb1 = gh.sum(axis=0)
            # grad on flat
            gflat = gh @ W1.T
            # grad on embeddings
            gemb = np.zeros_like(embed)
            for j in range(d):
                gemb_j = gflat[:, j*E:(j+1)*E]  # (B, E)
                idx = X_batch[:, j]
                np.add.at(gemb[j], idx, gemb_j)
            return [gemb, gW1, gb1, gW2, gb2]
        else:
            gW1 = flat.T @ probs + L2 * W1
            gb1 = probs.sum(axis=0)
            gflat = probs @ W1.T
            gemb = np.zeros_like(embed)
            for j in range(d):
                gemb_j = gflat[:, j*E:(j+1)*E]
                idx = X_batch[:, j]
                np.add.at(gemb[j], idx, gemb_j)
            return [gemb, gW1, gb1]

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            X_batch, y_batch = X[idx], y[idx]
            logits, flat, hidden = forward(X_batch)
            grads = backward(flat, hidden, logits, y_batch)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    # magic, d, E, H, C
    out = struct.pack("<5i", 0x454D4244, d, E, H, C)
    out += embed.astype("<f4").tobytes()
    out += params[1].astype("<f4").tobytes()  # W1
    out += params[2].astype("<f4").tobytes()  # b1
    if H:
        out += params[3].astype("<f4").tobytes()  # W2
        out += params[4].astype("<f4").tobytes()  # b2
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output: one byte class index.
 * Weights format: int32 magic 0x454D4244, d, E, H, C;
 * float32 embed[d][256][E], W1[d*E][H], b1[H], W2[H][C], b2[C] (if H>0). */
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
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad dims\n"); return 3; }
    long expected = 20;
    expected += (long)d * 256 * E * 4;
    expected += (long)d * E * (H ? H : C) * 4;
    expected += (H ? H : C) * 4;
    if (H) expected += (long)H * C * 4 + C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *embed = (const float *)(hd + 5);
    const float *W1 = embed + (long)d * 256 * E;
    const float *b1 = W1 + (long)d * E * (H ? H : C);
    const float *W2 = H ? b1 + H : NULL;
    const float *b2 = H ? W2 + (long)H * C : NULL;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = malloc(sizeof(float) * d * E);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *scores = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = embed + ((long)j * 256 + xi[j]) * E;
            memcpy(flat + j * E, e, E * sizeof(float));
        }

        if (H) {
            // hidden = relu(flat @ W1 + b1)
            for (int h = 0; h < H; h++) hidden[h] = b1[h];
            for (int f = 0; f < d * E; f++) {
                float val = flat[f];
                if (val == 0.0f) continue;
                const float *w = W1 + (long)f * H;
                for (int h = 0; h < H; h++) hidden[h] += val * w[h];
            }
            for (int h = 0; h < H; h++) if (hidden[h] < 0.0f) hidden[h] = 0.0f;
            // scores = hidden @ W2 + b2
            for (int c = 0; c < C; c++) scores[c] = b2[c];
            for (int h = 0; h < H; h++) {
                float hv = hidden[h];
                if (hv == 0.0f) continue;
                const float *w = W2 + (long)h * C;
                for (int c = 0; c < C; c++) scores[c] += hv * w[c];
            }
        } else {
            // scores = flat @ W1 + b1
            for (int c = 0; c < C; c++) scores[c] = b1[c];
            for (int f = 0; f < d * E; f++) {
                float val = flat[f];
                if (val == 0.0f) continue;
                const float *w = W1 + (long)f * C;
                for (int c = 0; c < C; c++) scores[c] += val * w[c];
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```