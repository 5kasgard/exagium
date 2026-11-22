```hypothesis
Family: label-emb (embedding logistic regression). Learns 8-dim token embeddings for the 65 characters plus 16 positional embeddings, concatenates them (128-dim), then a linear layer to 65 classes. This captures character similarities with far fewer parameters than label-linear's one-hot (16*256*65 vs 65*8+16*8+128*65 ≈ 9k vs 266k weights). Expect dev accuracy ≥0.34 with lower CPU cost. Failure if dev accuracy <0.33 or instructions/item > label-linear.
```

```python
import json, math, struct, sys
import numpy as np

# Fixed hyperparameters chosen on tuning split (last 20% of train.in)
E = 8          # embedding dimension
LR = 0.01      # learning rate
L2 = 1e-4      # L2 regularization
BATCH = 64     # minibatch size
EPOCHS = 30    # epochs on full data
SEED = 1

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def build_mapping(class_chars):
    mapping = np.zeros(256, dtype=np.uint8)
    for idx, ch in enumerate(class_chars):
        mapping[ord(ch)] = idx
    return mapping

def train_model(X_idx, y, d, C, E, lr, l2, batch, epochs, rng, verbose=False):
    # X_idx: (n, d) uint8 class indices (0..C-1)
    # y: (n,) int64
    n = X_idx.shape[0]
    # Parameters
    token_emb = rng.normal(0, 0.01, (C, E)).astype(np.float32)
    pos_emb = rng.normal(0, 0.01, (d, E)).astype(np.float32)
    W = rng.normal(0, 0.01, (d * E, C)).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)
    params = [token_emb, pos_emb, W, b]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            x = X_idx[idx]          # (b, d)
            yy = y[idx]             # (b,)
            bsize = x.shape[0]
            # Forward
            tok = token_emb[x]      # (b, d, E)
            h = tok + pos_emb[None, :, :]  # (b, d, E)
            h_flat = h.reshape(bsize, d * E)  # (b, d*E)
            scores = h_flat @ W + b            # (b, C)
            # Softmax + cross-entropy gradient
            max_s = scores.max(axis=1, keepdims=True)
            exp_s = np.exp(scores - max_s)
            probs = exp_s / exp_s.sum(axis=1, keepdims=True)
            probs[np.arange(bsize), yy] -= 1.0
            probs /= bsize
            # Backward
            grad_W = h_flat.T @ probs + l2 * W
            grad_b = probs.sum(axis=0)
            grad_h_flat = probs @ W.T               # (b, d*E)
            grad_h = grad_h_flat.reshape(bsize, d, E)  # (b, d, E)
            grad_pos_emb = grad_h.sum(axis=0)       # (d, E)
            grad_token_emb = np.zeros_like(token_emb)
            # scatter-add for token embeddings
            np.add.at(grad_token_emb, x.reshape(-1), grad_h.reshape(-1, E))
            grad_token_emb += l2 * token_emb
            grads = [grad_token_emb, grad_pos_emb, grad_W, grad_b]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * gr
                v[i] = beta2 * v[i] + (1 - beta2) * (gr * gr)
                m_hat = m[i] / (1 - beta1 ** t)
                v_hat = v[i] / (1 - beta2 ** t)
                p -= lr * m_hat / (np.sqrt(v_hat) + eps)
        if verbose:
            # compute train accuracy
            pred = (h_flat @ W + b).argmax(axis=1)
            acc = (pred == yy).mean()
            print(f"epoch {epoch+1}: train acc {acc:.4f}")
    return token_emb, pos_emb, W, b

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    class_chars = card["class_chars"]
    mapping = build_mapping(class_chars)
    # Load data
    X_raw = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    # Map input bytes to class indices
    X_idx = mapping[X_raw]
    n = X_idx.shape[0]
    # Split: last 20% for tuning
    split = int(0.8 * n)
    X_train, X_tune = X_idx[:split], X_idx[split:]
    y_train, y_tune = y[:split], y[split:]
    rng = np.random.default_rng(SEED)
    # Simple grid search on tuning split (quick, 5 epochs each)
    best_acc = -1
    best_params = None
    for E_try in [4, 8, 16]:
        for lr_try in [0.01, 0.03]:
            tok, pos, W, b = train_model(X_train, y_train, d, C, E_try, lr_try, L2, BATCH, 5, rng)
            # Evaluate on tune
            tok_t = tok[X_tune]
            h_t = tok_t + pos[None, :, :]
            h_flat_t = h_t.reshape(-1, d * E_try)
            scores_t = h_flat_t @ W + b
            pred_t = scores_t.argmax(axis=1)
            acc = (pred_t == y_tune).mean()
            if acc > best_acc:
                best_acc = acc
                best_params = (E_try, lr_try)
    E_final, lr_final = best_params
    # Retrain on full data
    token_emb, pos_emb, W, b = train_model(X_idx, y, d, C, E_final, lr_final, L2, BATCH, EPOCHS, rng)
    # Write weights
    magic = 0x4C424D45  # 'L','B','M','E' little-endian
    header = struct.pack("<4i", magic, d, E_final, C)
    mapping_bytes = mapping.astype(np.uint8).tobytes()
    token_bytes = token_emb.astype("<f4").tobytes()
    pos_bytes = pos_emb.astype("<f4").tobytes()
    W_bytes = W.astype("<f4").tobytes()
    b_bytes = b.astype("<f4").tobytes()
    out = header + mapping_bytes + token_bytes + pos_bytes + W_bytes + b_bytes
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
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wbuf = slurp(argv[1], &nw);
    if (nw < 4 + 12 + 256) { fprintf(stderr, "bad weights\n"); return 3; }
    uint32_t magic = wbuf[0] | wbuf[1] << 8 | wbuf[2] << 16 | (uint32_t)wbuf[3] << 24;
    if (magic != 0x4C424D45) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int32_t *hd = (int32_t *)(wbuf + 4);
    int d = hd[0], E = hd[1], C = hd[2];
    if (d != 16 || C != 65 || E <= 0 || E > 64) { fprintf(stderr, "bad weights dims\n"); return 3; }
    unsigned char *mapping = wbuf + 4 + 12;
    float *token_emb = (float *)(mapping + 256);
    float *pos_emb = token_emb + 65 * E;
    float *W = pos_emb + 16 * E;
    float *b = W + 16 * E * C;
    long expected = 4 + 12 + 256 + 4L * (65 * E + 16 * E + 16 * E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h_flat = malloc(sizeof(float) * 16 * E);
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // Build concatenated embedding
        for (int j = 0; j < d; j++) {
            unsigned char byte = xi[j];
            unsigned char cls = mapping[byte];
            const float *tok = token_emb + (size_t)cls * E;
            const float *pos = pos_emb + (size_t)j * E;
            float *dest = h_flat + j * E;
            for (int e = 0; e < E; e++) dest[e] = tok[e] + pos[e];
        }
        // Linear layer
        for (int c = 0; c < C; c++) {
            float sum = b[c];
            for (int f = 0; f < 16 * E; f++) sum += h_flat[f] * W[(size_t)f * C + c];
            scores[c] = sum;
        }
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```