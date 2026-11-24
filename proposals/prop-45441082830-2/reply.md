```hypothesis
Family: label-embed-mlp. Learns per-position byte embeddings (size 8) followed by one ReLU hidden layer (size 32). This compresses the 4096 one-hot features to 128 dense features, improving generalization on character-level context. Expected to exceed label-linear's 0.3375 accuracy with similar or lower inference cost. Failure if dev accuracy ≤ 0.34 or instructions/item ≥ label-linear.
```

```python
"""label-embed-mlp: per-position byte embeddings + one ReLU hidden layer.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only."""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
E = 8          # embedding size per byte
H = 32         # hidden layer size (0 for linear)
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100

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
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize embeddings: (d, 256, E)
    scale_emb = math.sqrt(6.0 / (256 + E))
    embeddings = rng.uniform(-scale_emb, scale_emb, (d, 256, E)).astype(np.float32)

    # Initialize first layer: (d*E, H) or (d*E, C)
    fan_in1 = d * E
    fan_out1 = H if H else C
    scale1 = math.sqrt(6.0 / (fan_in1 + fan_out1))
    W1 = rng.uniform(-scale1, scale1, (fan_in1, fan_out1)).astype(np.float32)
    b1 = np.zeros(fan_out1, np.float32)

    # Initialize second layer if H > 0
    if H:
        scale2 = math.sqrt(6.0 / (H + C))
        W2 = rng.uniform(-scale2, scale2, (H, C)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [embeddings, W1, b1, W2, b2]
    else:
        params = [embeddings, W1, b1]

    # Adam state
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = len(yb)

            # Forward: gather embeddings
            emb_concat = np.zeros((B, d * E), np.float32)
            for j in range(d):
                emb_concat[:, j*E:(j+1)*E] = embeddings[j, xb[:, j]]

            if H:
                hidden = emb_concat @ W1 + b1
                hidden_relu = np.maximum(hidden, 0)
                logits = hidden_relu @ W2 + b2
            else:
                logits = emb_concat @ W1 + b1

            # Softmax + cross-entropy gradient
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            grad_logits = (probs / B).astype(np.float32)

            # Backward
            grads = [None] * len(params)
            if H:
                # grad_W2, grad_b2
                grad_W2 = hidden_relu.T @ grad_logits + L2 * W2
                grad_b2 = grad_logits.sum(axis=0)
                # grad_hidden_relu
                grad_hidden_relu = grad_logits @ W2.T
                # ReLU mask
                grad_hidden = grad_hidden_relu * (hidden > 0)
                # grad_W1, grad_b1
                grad_W1 = emb_concat.T @ grad_hidden + L2 * W1
                grad_b1 = grad_hidden.sum(axis=0)
                # grad_emb_concat
                grad_emb_concat = grad_hidden @ W1.T
                # grad_embeddings
                grad_embeddings = np.zeros_like(embeddings)
                for j in range(d):
                    np.add.at(grad_embeddings[j], xb[:, j], grad_emb_concat[:, j*E:(j+1)*E])
                grads = [grad_embeddings, grad_W1, grad_b1, grad_W2, grad_b2]
            else:
                grad_W1 = emb_concat.T @ grad_logits + L2 * W1
                grad_b1 = grad_logits.sum(axis=0)
                grad_emb_concat = grad_logits @ W1.T
                grad_embeddings = np.zeros_like(embeddings)
                for j in range(d):
                    np.add.at(grad_embeddings[j], xb[:, j], grad_emb_concat[:, j*E:(j+1)*E])
                grads = [grad_embeddings, grad_W1, grad_b1]

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
    # Header: magic, d, E, H, C
    magic = 0x454D4245  # 'EMBE'
    out = struct.pack("<5i", magic, d, E, H, C)
    # Embeddings: d * 256 * E floats
    out += embeddings.astype("<f4").tobytes()
    # W1, b1
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
/* label-embed-mlp inference: ./infer weights.bin in.rec N */
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
    if (nw < 20 || hd[0] != 0x454D4245) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || E != 8 || H != 32 || C != 65) { fprintf(stderr, "unexpected dims\n"); return 3; }

    size_t emb_size = (size_t)d * 256 * E;
    size_t w1_size = (size_t)d * E * (H ? H : C);
    size_t b1_size = H ? H : C;
    size_t w2_size = H ? (size_t)H * C : 0;
    size_t b2_size = H ? C : 0;
    long expected = 20 + 4L * (emb_size + w1_size + b1_size + w2_size + b2_size);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1 = emb + emb_size;
    const float *b1 = W1 + w1_size;
    const float *W2 = b1 + b1_size;
    const float *b2 = W2 + w2_size;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * E);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((size_t)j * 256 + xi[j]) * E;
            memcpy(concat + j * E, e, E * sizeof(float));
        }

        if (H) {
            // hidden = relu(concat @ W1 + b1)
            for (int h = 0; h < H; h++) {
                float sum = b1[h];
                for (int f = 0; f < d * E; f++) sum += concat[f] * W1[(size_t)f * H + h];
                hidden[h] = sum > 0.0f ? sum : 0.0f;
            }
            // logits = hidden @ W2 + b2
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int h = 0; h < H; h++) sum += hidden[h] * W2[(size_t)h * C + c];
                logits[c] = sum;
            }
        } else {
            // logits = concat @ W1 + b1
            for (int c = 0; c < C; c++) {
                float sum = b1[c];
                for (int f = 0; f < d * E; f++) sum += concat[f] * W1[(size_t)f * C + c];
                logits[c] = sum;
            }
        }

        // Argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(concat); free(hidden); free(logits);
    return 0;
}
```