```hypothesis
We propose the "label-embsum" family: a position-specific embedding sum model.
Each of the 16 context bytes is mapped to a learned D-dimensional embedding (position-specific),
summed, and passed through a linear layer to 65 classes.
This captures character similarities and positional effects with O(16*65*D) parameters,
far fewer than label-linear's one-hot linear model (4096*65).
It should achieve higher accuracy by learning distributed representations
while remaining fast at inference (16 embedding lookups + D*65 matvec).
If dev accuracy does not exceed label-linear's 33.75%, the hypothesis fails.
```

```python
import json, math, struct, sys

import numpy as np

# Hyperparameters
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
EPOCHS = 30
TUNE_EPOCHS = 15
CANDIDATE_D = [16, 32, 64, 128]
MAGIC = 0x454D4253  # 'EMBS' in little-endian


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def main():
    card = json.load(open(sys.argv[1]))
    class_chars = card["class_chars"]
    C = card["classes"]  # 65
    d = card["input"]["bytes"]  # 16

    # Build mapping from byte value to class index
    byte_to_class = np.full(256, 255, dtype=np.uint8)
    for idx, ch in enumerate(class_chars):
        byte_to_class[ord(ch)] = idx

    # Load data
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.uint8)

    # Map X to class indices
    X = byte_to_class[X_bytes]
    assert (X != 255).all(), "Unknown byte in input"

    n = X.shape[0]
    # Split: last 20% for tuning
    tune_size = n // 5
    train_X, tune_X = X[:-tune_size], X[-tune_size:]
    train_y, tune_y = y[:-tune_size], y[-tune_size:]
    n_train = train_X.shape[0]

    rng = np.random.default_rng(SEED)

    best_D = CANDIDATE_D[0]
    best_acc = -1.0
    best_params = None

    for D in CANDIDATE_D:
        # Initialize parameters
        emb = (rng.uniform(-1, 1, (d, C, D)) * math.sqrt(6.0 / (C + D))).astype(np.float32)
        W_out = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / (D + C))).astype(np.float32)
        b_out = np.zeros(C, dtype=np.float32)
        params = [emb, W_out, b_out]
        state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
        t = 0

        # Training on train split
        for epoch in range(TUNE_EPOCHS):
            order = rng.permutation(n_train)
            for s in range(0, n_train, BATCH):
                idx = order[s:s + BATCH]
                x = train_X[idx]  # (B, d)
                yy = train_y[idx]  # (B,)
                B = x.shape[0]
                h = np.zeros((B, D), dtype=np.float32)
                for j in range(d):
                    h += emb[j, x[:, j], :]
                logits = h @ W_out + b_out
                logits -= logits.max(axis=1, keepdims=True)
                exp_logits = np.exp(logits)
                probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                probs[np.arange(B), yy] -= 1.0
                grad_logits = (probs / B).astype(np.float32)
                grad_W_out = h.T @ grad_logits + L2 * W_out
                grad_b_out = grad_logits.sum(axis=0)
                grad_h = grad_logits @ W_out.T
                grad_emb = np.zeros_like(emb)
                for j in range(d):
                    np.add.at(grad_emb[j], x[:, j], grad_h)
                grad_emb += L2 * emb
                grads = [grad_emb, grad_W_out, grad_b_out]
                t += 1
                for i, (p, gr) in enumerate(zip(params, grads)):
                    m, v = state[i]
                    m *= 0.9; m += 0.1 * gr
                    v *= 0.999; v += 0.001 * gr * gr
                    m_hat = m / (1 - 0.9 ** t)
                    v_hat = v / (1 - 0.999 ** t)
                    p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

        # Evaluate on tune split
        B = tune_X.shape[0]
        h = np.zeros((B, D), dtype=np.float32)
        for j in range(d):
            h += emb[j, tune_X[:, j], :]
        logits = h @ W_out + b_out
        preds = logits.argmax(axis=1)
        acc = (preds == tune_y).mean()
        if acc > best_acc:
            best_acc = acc
            best_D = D
            best_params = [emb.copy(), W_out.copy(), b_out.copy()]

    # Retrain on full data with best_D
    D = best_D
    rng = np.random.default_rng(SEED)
    emb = (rng.uniform(-1, 1, (d, C, D)) * math.sqrt(6.0 / (C + D))).astype(np.float32)
    W_out = (rng.uniform(-1, 1, (D, C)) * math.sqrt(6.0 / (D + C))).astype(np.float32)
    b_out = np.zeros(C, dtype=np.float32)
    params = [emb, W_out, b_out]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            x = X[idx]
            yy = y[idx]
            B = x.shape[0]
            h = np.zeros((B, D), dtype=np.float32)
            for j in range(d):
                h += emb[j, x[:, j], :]
            logits = h @ W_out + b_out
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yy] -= 1.0
            grad_logits = (probs / B).astype(np.float32)
            grad_W_out = h.T @ grad_logits + L2 * W_out
            grad_b_out = grad_logits.sum(axis=0)
            grad_h = grad_logits @ W_out.T
            grad_emb = np.zeros_like(emb)
            for j in range(d):
                np.add.at(grad_emb[j], x[:, j], grad_h)
            grad_emb += L2 * emb
            grads = [grad_emb, grad_W_out, grad_b_out]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Write weights.bin
    emb_flat = emb.astype('<f4').tobytes()
    W_out_flat = W_out.astype('<f4').tobytes()
    b_out_flat = b_out.astype('<f4').tobytes()
    mapping_bytes = byte_to_class.astype(np.uint8).tobytes()
    header = struct.pack('<4i', MAGIC, d, D, C)
    out = header + mapping_bytes + emb_flat + W_out_flat + b_out_flat
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "bad weights: too small\n"); return 3; }
    uint32_t *hd = (uint32_t *)wb;
    uint32_t magic = hd[0];
    int d = (int32_t)hd[1];
    int D = (int32_t)hd[2];
    int C = (int32_t)hd[3];
    if (magic != 0x454D4253) { fprintf(stderr, "bad weights: magic mismatch\n"); return 3; }
    if (d != 16 || C != 65) { fprintf(stderr, "bad weights: unexpected d or C\n"); return 3; }

    unsigned char *mapping = wb + 16;
    float *emb = (float *)(mapping + 256);
    float *W_out = emb + (long)d * C * D;
    float *b_out = W_out + (long)D * C;

    long expected = 16 + 256 + (long)d * C * D * 4 + (long)D * C * 4 + (long)C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = malloc(sizeof(float) * D);
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        for (int u = 0; u < D; u++) h[u] = 0.0f;

        for (int j = 0; j < d; j++) {
            unsigned char byte_val = xi[j];
            unsigned char cls = mapping[byte_val];
            if (cls >= C) { fprintf(stderr, "unknown byte value %d at position %d\n", byte_val, j); return 3; }
            const float *emb_ptr = emb + ((long)j * C + cls) * D;
            for (int u = 0; u < D; u++) h[u] += emb_ptr[u];
        }

        for (int c = 0; c < C; c++) {
            float sum = b_out[c];
            for (int u = 0; u < D; u++) sum += h[u] * W_out[(long)u * C + c];
            logits[c] = sum;
        }

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```