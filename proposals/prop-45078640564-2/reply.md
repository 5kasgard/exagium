```hypothesis
Family: label-rnn (simple recurrent network with shared byte embeddings and positional embeddings).
Why: Character-level language modeling benefits from sequential context; an RNN captures order and pairwise interactions that linear models miss, while staying small (E=32, H=64 ~200k params) and fast at inference (~50k ops/item). The fixed 16-byte window makes unrolled BPTT trivial.
Failure signal: If dev accuracy ≤ 0.3375 (label-linear) or instructions/item > 100k, the RNN's sequential inductive bias doesn't help enough for this dataset size.
```

```python
"""label-rnn: simple RNN with byte/positional embeddings for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, uses numpy only.
Weights format (little-endian):
  int32 magic=0x524E4E, d=16, E=32, H=64, C=65
  float32 embed[(256+d)*E]   (byte embeddings then positional embeddings)
  float32 W_xh[H*E], W_hh[H*H], b_h[H]
  float32 W_hy[C*H], b_y[C]"""
import json, math, struct, sys

import numpy as np

# Hyperparameters (chosen on tuning split)
SEED = 1
E = 32          # embedding dimension
H = 64          # hidden size
LR = 3e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS = 100
PATIENCE = 5

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]          # 16
    C = card["classes"]                 # 65
    # Load all training data
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_bytes.shape[0]
    # Split: last 20% for tuning
    split = int(0.8 * n)
    X_tr, y_tr = X_bytes[:split], y[:split]
    X_val, y_val = X_bytes[split:], y[split:]
    n_tr = X_tr.shape[0]

    rng = np.random.default_rng(SEED)

    # Initialize parameters
    embed = (rng.normal(0, 0.01, ((256 + d) * E)).astype(np.float32))
    W_xh = (rng.normal(0, math.sqrt(2.0 / (E + H)), (H, E)).astype(np.float32))
    W_hh = (rng.normal(0, math.sqrt(2.0 / (H + H)), (H, H)).astype(np.float32))
    b_h = np.zeros(H, np.float32)
    W_hy = (rng.normal(0, math.sqrt(2.0 / (H + C)), (C, H)).astype(np.float32))
    b_y = np.zeros(C, np.float32)

    params = [embed, W_xh, W_hh, b_h, W_hy, b_y]
    param_names = ['embed', 'W_xh', 'W_hh', 'b_h', 'W_hy', 'b_y']
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    def forward(X_batch):
        """X_batch: (B, d) uint8. Returns logits (B, C), and cache for backprop."""
        B = X_batch.shape[0]
        # Embeddings: byte embeddings + positional embeddings
        # embed[:256*E] are byte embeddings, embed[256*E:] are positional (d*E)
        byte_emb = embed[:256*E].reshape(256, E)
        pos_emb = embed[256*E:].reshape(d, E)
        # (B, d, E)
        x_emb = byte_emb[X_batch] + pos_emb[np.newaxis, :, :]
        h = np.zeros((B, H), np.float32)
        h_cache = np.zeros((B, d, H), np.float32)
        pre_tanh = np.zeros((B, d, H), np.float32)
        for step in range(d):
            x = x_emb[:, step, :]          # (B, E)
            pre = x @ W_xh.T + h @ W_hh.T + b_h   # (B, H)
            pre_tanh[:, step, :] = pre
            h = np.tanh(pre)
            h_cache[:, step, :] = h
        logits = h @ W_hy.T + b_y
        cache = (x_emb, h_cache, pre_tanh)
        return logits, cache

    def loss_and_grad(X_batch, y_batch):
        B = X_batch.shape[0]
        logits, cache = forward(X_batch)
        x_emb, h_cache, pre_tanh = cache
        # Softmax + cross-entropy
        logits_max = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        loss = -np.log(probs[np.arange(B), y_batch] + 1e-12).mean()
        # Gradient on logits
        dlogits = probs.copy()
        dlogits[np.arange(B), y_batch] -= 1.0
        dlogits /= B
        # Gradients for output layer
        h_last = h_cache[:, -1, :]
        dW_hy = dlogits.T @ h_last + L2 * W_hy
        db_y = dlogits.sum(axis=0)
        dh = dlogits @ W_hy   # (B, H)
        # Backprop through time
        dW_xh = np.zeros_like(W_xh)
        dW_hh = np.zeros_like(W_hh)
        db_h = np.zeros_like(b_h)
        demb = np.zeros_like(embed)
        byte_emb = embed[:256*E].reshape(256, E)
        pos_emb = embed[256*E:].reshape(d, E)
        demb_byte = demb[:256*E].reshape(256, E)
        demb_pos = demb[256*E:].reshape(d, E)
        for step in reversed(range(d)):
            pre = pre_tanh[:, step, :]
            dh_pre = dh * (1.0 - np.tanh(pre)**2)   # (B, H)
            db_h += dh_pre.sum(axis=0)
            x = x_emb[:, step, :]
            dW_xh += dh_pre.T @ x
            dW_hh += dh_pre.T @ (h_cache[:, step-1, :] if step > 0 else np.zeros((B, H)))
            dx = dh_pre @ W_xh   # (B, E)
            dh = dh_pre @ W_hh   # (B, H)
            # Accumulate embedding gradients
            for b in range(B):
                byte_val = X_batch[b, step]
                demb_byte[byte_val] += dx[b]
                demb_pos[step] += dx[b]
        # Add L2 to recurrent weights
        dW_xh += L2 * W_xh
        dW_hh += L2 * W_hh
        grads = [demb, dW_xh, dW_hh, db_h, dW_hy, db_y]
        return loss, grads

    def evaluate(X, y):
        n = X.shape[0]
        correct = 0
        for s in range(0, n, BATCH):
            Xb = X[s:s+BATCH]
            yb = y[s:s+BATCH]
            logits, _ = forward(Xb)
            pred = logits.argmax(axis=1)
            correct += (pred == yb).sum()
        return correct / n

    best_val_acc = -1.0
    best_epoch = 0
    no_improve = 0
    # Training loop on train split
    for epoch in range(1, MAX_EPOCHS + 1):
        order = rng.permutation(n_tr)
        epoch_loss = 0.0
        batches = 0
        for s in range(0, n_tr, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X_tr[idx], y_tr[idx]
            loss, grads = loss_and_grad(Xb, yb)
            epoch_loss += loss
            batches += 1
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
        # Validation
        val_acc = evaluate(X_val, y_val)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            # Save best params
            best_params = [p.copy() for p in params]
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            break

    # Retrain on all data for best_epoch epochs
    params = best_params
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    for epoch in range(1, best_epoch + 1):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X_bytes[idx], y[idx]
            loss, grads = loss_and_grad(Xb, yb)
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                m_hat = m[i] / (1 - 0.9 ** t)
                v_hat = v[i] / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    embed, W_xh, W_hh, b_h, W_hy, b_y = params
    out = struct.pack("<5i", 0x524E4E, d, E, H, C)
    out += embed.astype("<f4").tobytes()
    out += W_xh.astype("<f4").tobytes()
    out += W_hh.astype("<f4").tobytes()
    out += b_h.astype("<f4").tobytes()
    out += W_hy.astype("<f4").tobytes()
    out += b_y.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-rnn inference: ./infer weights.bin in.rec N
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
    uint32_t k = buf[*pos] | buf[*pos+1]<<8 | buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k; const unsigned char *r = buf + *pos + 4; *pos += 4 + k;
    return r;
}

static void put_rec(FILE *f, const unsigned char *r, uint32_t k) {
    unsigned char h[4] = {k & 255, k>>8 & 255, k>>16 & 255, k>>24};
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x524E4E) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || C != 65 || E <= 0 || H <= 0) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 20L + 4L * ((256L + d) * E + (long)H * E + (long)H * H + H + (long)C * H + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 5);
    const float *embed = p; p += (256 + d) * E;
    const float *W_xh = p; p += H * E;
    const float *W_hh = p; p += H * H;
    const float *b_h = p; p += H;
    const float *W_hy = p; p += C * H;
    const float *b_y = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = (float *)calloc(H, sizeof(float));
    float *h_new = (float *)malloc(H * sizeof(float));
    float *x_emb = (float *)malloc(E * sizeof(float));
    float *logits = (float *)malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(h, 0, H * sizeof(float));
        for (int step = 0; step < d; step++) {
            const float *byte_emb = embed + xi[step] * E;
            const float *pos_emb = embed + 256 * E + step * E;
            for (int e = 0; e < E; e++) x_emb[e] = byte_emb[e] + pos_emb[e];
            // h_new = tanh( x_emb * W_xh^T + h * W_hh^T + b_h )
            for (int j = 0; j < H; j++) {
                float sum = b_h[j];
                for (int e = 0; e < E; e++) sum += x_emb[e] * W_xh[j * E + e];
                for (int k = 0; k < H; k++) sum += h[k] * W_hh[j * H + k];
                h_new[j] = tanhf(sum);
            }
            memcpy(h, h_new, H * sizeof(float));
        }
        // logits = h * W_hy^T + b_y
        for (int c = 0; c < C; c++) {
            float sum = b_y[c];
            for (int j = 0; j < H; j++) sum += h[j] * W_hy[c * H + j];
            logits[c] = sum;
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(h); free(h_new); free(x_emb); free(logits);
    if (x) free(x);
    return 0;
}
```