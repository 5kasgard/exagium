```hypothesis
Family: label-embed. A small neural network with per-position byte embeddings (256 vocab) concatenated and fed through an optional ReLU hidden layer. Embeddings capture character similarities; the hidden layer models interactions. Tuning embedding size (16/32) and hidden size (0/64/128) on the last 20% of train should beat label-linear's 0.3375 accuracy. Failure: if best tuning accuracy ≤0.3375 or inference cost exceeds label-linear by >2x.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameter grid for tuning split
EMBED_DIMS = [16, 32]
HIDDEN_SIZES = [0, 64, 128]
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
TUNE_EPOCHS = 15
FULL_EPOCHS = 40


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_model(X, y, d, C, D, H, epochs, seed):
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    # embeddings: [d, 256, D]
    embed = (rng.uniform(-1, 1, (d, 256, D)) * math.sqrt(6.0 / (256 * D))).astype(np.float32)
    if H > 0:
        W1 = (rng.uniform(-1, 1, (d * D, H)) * math.sqrt(6.0 / (d * D))).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [embed, W1, b1, W2, b2]
    else:
        W1 = (rng.uniform(-1, 1, (d * D, C)) * math.sqrt(6.0 / (d * D))).astype(np.float32)
        b1 = np.zeros(C, np.float32)
        params = [embed, W1, b1]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb = X[idx]          # (batch, d)
            yb = y[idx]          # (batch,)
            # Embedding lookup: (batch, d, D) -> (batch, d*D)
            emb = embed[np.arange(d), xb.T].transpose(1, 0, 2).reshape(len(yb), -1)
            if H > 0:
                h = np.maximum(emb @ W1 + b1, 0, dtype=np.float32)
                logits = h @ W2 + b2
            else:
                logits = emb @ W1 + b1
            # Softmax cross-entropy gradient
            max_logits = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - max_logits, dtype=np.float32)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad_logits = (probs / len(yb)).astype(np.float32)
            # Backward
            if H > 0:
                grad_W2 = h.T @ grad_logits + L2 * W2
                grad_b2 = grad_logits.sum(0)
                grad_h = grad_logits @ W2.T
                grad_h *= (h > 0)
                grad_W1 = emb.T @ grad_h + L2 * W1
                grad_b1 = grad_h.sum(0)
                grad_emb = grad_h @ W1.T
            else:
                grad_W1 = emb.T @ grad_logits + L2 * W1
                grad_b1 = grad_logits.sum(0)
                grad_emb = grad_logits @ W1.T
            grad_emb = grad_emb.reshape(len(yb), d, D)
            grad_embed = np.zeros_like(embed)
            for j in range(d):
                np.add.at(grad_embed[j], xb[:, j], grad_emb[:, j])
            grads = [grad_embed, grad_W1, grad_b1] + ([grad_W2, grad_b2] if H > 0 else [])
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                p -= LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return params


def evaluate(params, X, y, d, C, D, H):
    embed = params[0]
    if H > 0:
        W1, b1, W2, b2 = params[1], params[2], params[3], params[4]
    else:
        W1, b1 = params[1], params[2]
    n = X.shape[0]
    correct = 0
    for s in range(0, n, 1024):
        xb = X[s:s + 1024]
        yb = y[s:s + 1024]
        emb = embed[np.arange(d), xb.T].transpose(1, 0, 2).reshape(len(yb), -1)
        if H > 0:
            h = np.maximum(emb @ W1 + b1, 0)
            logits = h @ W2 + b2
        else:
            logits = emb @ W1 + b1
        pred = logits.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / n


def write_weights(path, d, C, D, H, params):
    embed = params[0]
    if H > 0:
        W1, b1, W2, b2 = params[1], params[2], params[3], params[4]
    else:
        W1, b1 = params[1], params[2]
    magic = 0x454D4244  # "EMBD"
    out = struct.pack("<5i", magic, d, D, H, C)
    out += embed.astype("<f4").tobytes()
    out += W1.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    if H > 0:
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()
    open(path, "wb").write(out)


def main():
    card = json.load(open(sys.argv[1]))
    X_recs = records(sys.argv[2])
    y_recs = records(sys.argv[3])
    X = np.frombuffer(b"".join(X_recs), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in y_recs], dtype=np.int64)
    C = card["classes"]
    d = X.shape[1]
    assert d == 16

    # Split: last 20% for tuning
    n = X.shape[0]
    split = int(0.8 * n)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    best_acc = -1.0
    best_D, best_H = EMBED_DIMS[0], HIDDEN_SIZES[0]
    best_model = None

    for D in EMBED_DIMS:
        for H in HIDDEN_SIZES:
            model = train_model(X_train, y_train, d, C, D, H, TUNE_EPOCHS, SEED)
            acc = evaluate(model, X_val, y_val, d, C, D, H)
            if acc > best_acc:
                best_acc = acc
                best_D, best_H = D, H
                best_model = model

    print(f"Best: D={best_D}, H={best_H}, val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data
    final_model = train_model(X, y, d, C, best_D, best_H, FULL_EPOCHS, SEED)
    write_weights(sys.argv[4], d, C, best_D, best_H, final_model)


if __name__ == "__main__":
    main()
```

```c
/* label-embed inference: ./infer weights.bin in.rec N
 * Input record: 16 bytes. Output record: 1 byte (class index). */
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
    if (nw < 20 || hd[0] != 0x454D4244) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], D = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || D <= 0 || H < 0) { fprintf(stderr, "bad header dims\n"); return 3; }

    long expected = 20;
    expected += (long)d * 256 * D * 4;          // embeddings
    if (H > 0) {
        expected += (long)d * D * H * 4;        // W1
        expected += H * 4;                      // b1
        expected += (long)H * C * 4;            // W2
        expected += C * 4;                      // b2
    } else {
        expected += (long)d * D * C * 4;        // W1
        expected += C * 4;                      // b1
    }
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *embed = (const float *)(hd + 5);
    const float *W1 = embed + (long)d * 256 * D;
    const float *b1 = W1 + (H > 0 ? (long)d * D * H : (long)d * D * C);
    const float *W2 = b1 + (H > 0 ? H : C);
    const float *b2 = W2 + (H > 0 ? (long)H * C : 0);

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *emb_concat = malloc((long)d * D * sizeof(float));
    float *hidden = H > 0 ? malloc(H * sizeof(float)) : NULL;
    float *output = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Embedding lookup and concatenation
        for (int j = 0; j < d; j++) {
            const float *e = embed + ((long)j * 256 + xi[j]) * D;
            memcpy(emb_concat + j * D, e, D * sizeof(float));
        }

        if (H > 0) {
            // Hidden layer: ReLU(emb_concat * W1 + b1)
            for (int h = 0; h < H; h++) {
                float sum = b1[h];
                for (int w = 0; w < d * D; w++) sum += emb_concat[w] * W1[(long)w * H + h];
                hidden[h] = sum > 0.0f ? sum : 0.0f;
            }
            // Output layer: hidden * W2 + b2
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
                output[c] = sum;
            }
        } else {
            // Linear: emb_concat * W1 + b1
            for (int c = 0; c < C; c++) {
                float sum = b1[c];
                for (int w = 0; w < d * D; w++) sum += emb_concat[w] * W1[(long)w * C + c];
                output[c] = sum;
            }
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (output[c] > output[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```