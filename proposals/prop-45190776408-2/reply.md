```hypothesis
Family: label-fm (Factorization Machine with shared embeddings and class-specific interaction weights).
Why: Character-level language modeling benefits from pairwise interactions between nearby context bytes (e.g., bigrams). A low-rank FM captures all pairwise position interactions efficiently (O(d*k + C*k) per item) and is more expressive than linear models while cheaper than GBDT.
Failure signal: Dev accuracy ≤ 0.3375 (label-linear) or CPU cost ≥ label-gbdt (131k Ir/item).
```

```python
"""label-fm: Factorization Machine for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Tunes on last 20% of train.in, then trains on all train."""
import json, math, os, struct, sys

import numpy as np

# Fixed seed for reproducibility
SEED = 1
# Hyperparameter grid for tuning
K_CHOICES = [8, 16, 32]
LR_CHOICES = [1e-3, 3e-3, 1e-2]
L2_CHOICES = [1e-4, 1e-3]
# Training budget (same as label-linear)
SAMPLES = 600000
BATCH = 64
MAX_EPOCHS = 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


class FMModel:
    def __init__(self, d, C, k, rng):
        self.d, self.C, self.k = d, C, k
        # Parameters
        self.bias = np.zeros(C, np.float32)
        self.W_lin = np.zeros((d, 256, C), np.float32)
        self.Emb = np.zeros((d, 256, k), np.float32)
        self.U = np.zeros((C, k), np.float32)
        # Xavier initialization for embeddings and U
        scale_emb = math.sqrt(6.0 / (256 + k))
        self.Emb = (rng.uniform(-1, 1, (d, 256, k)) * scale_emb).astype(np.float32)
        scale_U = math.sqrt(6.0 / (k + C))
        self.U = (rng.uniform(-1, 1, (C, k)) * scale_U).astype(np.float32)
        # Linear weights init small
        self.W_lin = (rng.uniform(-0.01, 0.01, (d, 256, C))).astype(np.float32)

    def pack_params(self):
        return [self.bias, self.W_lin, self.Emb, self.U]

    def unpack_params(self, params):
        self.bias, self.W_lin, self.Emb, self.U = params

    def forward(self, X_batch):
        """X_batch: (B, d) uint8 -> logits (B, C) float32"""
        B = X_batch.shape[0]
        # Linear part
        logits = self.bias.copy()
        for j in range(self.d):
            logits += self.W_lin[j, X_batch[:, j], :]
        # Interaction part
        sum_v = np.zeros((B, self.k), np.float32)
        sum_v_sq = np.zeros((B, self.k), np.float32)
        for j in range(self.d):
            emb = self.Emb[j, X_batch[:, j], :]  # (B, k)
            sum_v += emb
            sum_v_sq += emb * emb
        inter = 0.5 * (sum_v * sum_v - sum_v_sq)  # (B, k)
        logits += inter @ self.U.T  # (B, C)
        return logits

    def loss_grad(self, X_batch, y_batch):
        """Returns (loss, grads) where grads is list matching pack_params order."""
        B = X_batch.shape[0]
        logits = self.forward(X_batch)
        # Softmax + cross-entropy
        max_logits = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        # Loss
        correct_probs = probs[np.arange(B), y_batch]
        loss = -np.log(correct_probs + 1e-12).mean()
        # Gradient of logits
        dlogits = probs
        dlogits[np.arange(B), y_batch] -= 1.0
        dlogits /= B  # (B, C)
        # Gradients
        dbias = dlogits.sum(axis=0)  # (C,)
        dW_lin = np.zeros_like(self.W_lin)
        dEmb = np.zeros_like(self.Emb)
        dU = dlogits.T @ (0.5 * (sum_v * sum_v - sum_v_sq))  # Wait, need inter
        # Recompute inter for backward
        sum_v = np.zeros((B, self.k), np.float32)
        sum_v_sq = np.zeros((B, self.k), np.float32)
        for j in range(self.d):
            emb = self.Emb[j, X_batch[:, j], :]
            sum_v += emb
            sum_v_sq += emb * emb
        inter = 0.5 * (sum_v * sum_v - sum_v_sq)  # (B, k)
        dU = dlogits.T @ inter  # (C, k)
        dinter = dlogits @ self.U  # (B, k)
        # Backprop through inter
        # dinter/dEmb_j = sum_v - Emb_j
        for j in range(self.d):
            emb_j = self.Emb[j, X_batch[:, j], :]  # (B, k)
            dEmb_j = dinter * (sum_v - emb_j)  # (B, k)
            # Accumulate gradients for linear weights
            for b in range(B):
                v = X_batch[b, j]
                dW_lin[j, v, :] += dlogits[b]
                dEmb[j, v, :] += dEmb_j[b]
        return loss, [dbias, dW_lin, dEmb, dU]

    def predict(self, X_batch):
        logits = self.forward(X_batch)
        return logits.argmax(axis=1).astype(np.uint8)


def evaluate(model, X, y):
    preds = model.predict(X)
    return (preds == y).mean()


def train_model(model, X_tr, y_tr, X_val, y_val, lr, l2, rng, verbose=False):
    n = X_tr.shape[0]
    steps_per_epoch = math.ceil(n / BATCH)
    max_steps = min(MAX_EPOCHS * steps_per_epoch, math.ceil(SAMPLES / BATCH))
    # Adam state
    params = model.pack_params()
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    best_val_acc = -1.0
    best_params = None
    for step in range(max_steps):
        # Shuffle once per epoch
        if step % steps_per_epoch == 0:
            order = rng.permutation(n)
        batch_idx = order[(step % steps_per_epoch) * BATCH: (step % steps_per_epoch + 1) * BATCH]
        X_batch = X_tr[batch_idx]
        y_batch = y_tr[batch_idx]
        loss, grads = model.loss_grad(X_batch, y_batch)
        # Adam update with L2 regularization (weight decay)
        t += 1
        for i, (p, gr) in enumerate(zip(params, grads)):
            gr = gr + l2 * p  # L2
            m[i] = 0.9 * m[i] + 0.1 * gr
            v[i] = 0.999 * v[i] + 0.001 * gr * gr
            m_hat = m[i] / (1 - 0.9 ** t)
            v_hat = v[i] / (1 - 0.999 ** t)
            p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        model.unpack_params(params)
        # Validate every epoch
        if (step + 1) % steps_per_epoch == 0:
            val_acc = evaluate(model, X_val, y_val)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = [p.copy() for p in params]
            if verbose:
                print(f"  epoch {(step+1)//steps_per_epoch}: loss={loss:.4f}, val_acc={val_acc:.4f}")
    # Restore best
    if best_params is not None:
        model.unpack_params(best_params)
    return best_val_acc


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    # Load all training data
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n_all = X_all.shape[0]
    # Split: last 20% for tuning
    split = int(0.8 * n_all)
    X_tr, X_val = X_all[:split], X_all[split:]
    y_tr, y_val = y_all[:split], y_all[split:]
    rng = np.random.default_rng(SEED)
    # Hyperparameter search
    best_score = -1.0
    best_cfg = None
    best_model = None
    for k in K_CHOICES:
        for lr in LR_CHOICES:
            for l2 in L2_CHOICES:
                model = FMModel(d, C, k, rng)
                val_acc = train_model(model, X_tr, y_tr, X_val, y_val, lr, l2, rng)
                if val_acc > best_score:
                    best_score = val_acc
                    best_cfg = (k, lr, l2)
                    best_model = model
                print(f"k={k}, lr={lr}, l2={l2} -> val_acc={val_acc:.4f}")
    print(f"Best config: k={best_cfg[0]}, lr={best_cfg[1]}, l2={best_cfg[2]} (val_acc={best_score:.4f})")
    # Retrain on full data with best config
    k, lr, l2 = best_cfg
    final_model = FMModel(d, C, k, rng)
    train_model(final_model, X_all, y_all, X_val, y_val, lr, l2, rng, verbose=True)
    # Save weights
    # Format: magic, d, C, k, then bias, W_lin, Emb, U (all float32 little-endian)
    magic = 0x464D4C31  # 'FML1'
    header = struct.pack("<4i", magic, d, C, k)
    bias, W_lin, Emb, U = final_model.pack_params()
    out = header
    out += bias.astype("<f4").tobytes()
    out += W_lin.astype("<f4").tobytes()
    out += Emb.astype("<f4").tobytes()
    out += U.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)
    print(f"Saved weights to {sys.argv[4]} ({len(out)} bytes)")


if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
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
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 16) { fprintf(stderr, "weights too small\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x464D4C31) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], C = hd[2], k = hd[3];
    if (d != 16 || C != 65 || k <= 0 || k > 128) { fprintf(stderr, "bad dims\n"); return 3; }
    // Expected size: 16 + 4*(C + d*256*C + d*256*k + C*k)
    long expected = 16 + 4L * (C + (long)d * 256 * C + (long)d * 256 * k + (long)C * k);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3; }
    const float *p = (const float *)(hd + 4);
    const float *bias = p; p += C;
    const float *W_lin = p; p += (long)d * 256 * C;
    const float *Emb = p; p += (long)d * 256 * k;
    const float *U = p;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(sizeof(float) * k);
    float *sum_v_sq = malloc(sizeof(float) * k);
    float *logits = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t rec_len; const unsigned char *xi = next_rec(x, nx, &pos, &rec_len);
        if (rec_len != (uint32_t)d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, rec_len, d); return 3; }
        // Linear part
        memcpy(logits, bias, C * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *w = W_lin + ((long)j * 256 + xi[j]) * C;
            for (int c = 0; c < C; c++) logits[c] += w[c];
        }
        // Interaction part
        memset(sum_v, 0, k * sizeof(float));
        memset(sum_v_sq, 0, k * sizeof(float));
        for (int j = 0; j < d; j++) {
            const float *e = Emb + ((long)j * 256 + xi[j]) * k;
            for (int f = 0; f < k; f++) {
                float val = e[f];
                sum_v[f] += val;
                sum_v_sq[f] += val * val;
            }
        }
        for (int f = 0; f < k; f++) {
            float inter = 0.5f * (sum_v[f] * sum_v[f] - sum_v_sq[f]);
            const float *u = U + (long)f; // U is (C, k) row-major? Actually U[c*k + f]
            for (int c = 0; c < C; c++) {
                logits[c] += inter * u[c * k];
            }
        }
        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```