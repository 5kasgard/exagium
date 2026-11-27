```hypothesis
Family: label-conv (tiny 2D CNN: 3x3 conv, 8 filters, ReLU, global average pooling, linear).
Why: Optdigits 8x8 images have spatial structure; a tiny CNN captures local patterns better than linear models with similar parameter count.
Failure: If dev accuracy ≤ 0.9613 (label-linear) or inference cost exceeds 2× label-linear.
```

```python
"""label-conv: tiny 2D CNN for 8x8 optdigits.
python3 -s -P train.py card.json train.in train.key weights.bin
Architecture: input 1x8x8 -> conv 3x3, 8 filters, pad=1 -> ReLU -> GAP -> linear 10.
Training: Adam, batch 64, epochs tuned on last 20% of train.in (file order)."""
import json, math, struct, sys

import numpy as np

SEED = 1
BATCH = 64
# Hyperparameter grid for tuning split
LR_GRID = [1e-3, 3e-3, 1e-2]
EPOCH_GRID = [30, 60, 100]
FILTERS = 8  # fixed architecture
KERNEL = 3
PAD = 1
CLASSES = 10
IN_H, IN_W = 8, 8
OUT_H = IN_H  # with pad=1, stride=1
OUT_W = IN_W
N_PATCH = OUT_H * OUT_W  # 64

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def im2col_indices(h, w, kh, kw, pad):
    """Return indices for im2col: (out_h, out_w, kh*kw) -> input indices (with -1 for padding)."""
    oh = h + 2*pad - kh + 1
    ow = w + 2*pad - kw + 1
    idx = np.full((oh, ow, kh*kw), -1, dtype=np.int32)
    for i in range(oh):
        for j in range(ow):
            for di in range(kh):
                for dj in range(kw):
                    ii = i + di - pad
                    jj = j + dj - pad
                    if 0 <= ii < h and 0 <= jj < w:
                        idx[i, j, di*kw + dj] = ii * w + jj
    return idx.reshape(-1, kh*kw)  # (oh*ow, kh*kw)

def forward(X, params, idx_col):
    """X: (N, 64) float32 in [0,1]. Returns logits (N, C), and cache for backward."""
    W_conv, b_conv, W_fc, b_fc = params
    N = X.shape[0]
    # reshape to (N, 1, 8, 8) but we use im2col on flattened 64
    # X_col: (N, 64, 9)
    X_col = X[:, idx_col]  # advanced indexing, pads with 0 where idx==-1
    # but idx_col has -1 for padding; we need to zero those columns.
    # Instead, we can pad X to (N, 64+?) but easier: replace -1 with 0 and multiply by 0 later.
    # We'll create a mask.
    mask = (idx_col >= 0)
    safe_idx = np.where(idx_col >= 0, idx_col, 0)
    X_col = X[:, safe_idx]  # (N, 64, 9)
    X_col = X_col * mask.astype(np.float32)  # zero out padding
    # Conv: (N, 64, 9) @ (9, F) -> (N, 64, F)
    conv_out = X_col @ W_conv.T + b_conv  # W_conv: (F, 9)
    # ReLU
    relu_out = np.maximum(conv_out, 0)
    # GAP: mean over 64 positions -> (N, F)
    gap_out = relu_out.mean(axis=1)
    # Linear
    logits = gap_out @ W_fc.T + b_fc  # W_fc: (C, F)
    cache = (X_col, mask, conv_out, relu_out, gap_out)
    return logits, cache

def loss_and_grad(logits, y, cache, params):
    N = logits.shape[0]
    # Softmax cross-entropy
    max_log = logits.max(axis=1, keepdims=True)
    exp_log = np.exp(logits - max_log)
    probs = exp_log / exp_log.sum(axis=1, keepdims=True)
    loss = -np.log(probs[np.arange(N), y] + 1e-12).mean()
    # dlogits
    dlogits = probs
    dlogits[np.arange(N), y] -= 1
    dlogits /= N
    X_col, mask, conv_out, relu_out, gap_out = cache
    W_conv, b_conv, W_fc, b_fc = params
    # Linear grad
    dW_fc = dlogits.T @ gap_out  # (C, F)
    db_fc = dlogits.sum(axis=0)  # (C,)
    dgap = dlogits @ W_fc  # (N, F)
    # GAP grad: divide by 64
    drelu = dgap[:, None, :] / N_PATCH  # (N, 1, F) -> broadcast to (N, 64, F)
    # ReLU grad
    dconv = drelu * (conv_out > 0)
    # Conv grad
    dW_conv = np.zeros_like(W_conv)
    db_conv = np.zeros_like(b_conv)
    # dW_conv[f, k] = sum_n sum_pos dconv[n, pos, f] * X_col[n, pos, k]
    # We can do: (N, 64, F) * (N, 64, 9) -> sum over N,64 -> (F,9)
    # But careful: X_col has zeros for padding, so it's fine.
    # Transpose: dconv (N,64,F) -> (F,N,64); X_col (N,64,9) -> (9,N,64)
    # dW_conv = (dconv.transpose(2,0,1) @ X_col).transpose(0,2,1)? Let's do explicit.
    # dW_conv[f, k] = sum_{n,pos} dconv[n,pos,f] * X_col[n,pos,k]
    # We can compute by: (dconv.transpose(2,0,1) @ X_col).transpose(0,2,1) is (F,9)?
    # Actually: dconv shape (N,64,F), X_col shape (N,64,9).
    # We want (F,9). So: dconv.swapaxes(0,2) -> (F,64,N) @ X_col.swapaxes(0,1) -> (64,N,9) -> (F,9) after sum over 64,N.
    # Better: use einsum or reshape.
    dW_conv = np.einsum('npf,npk->fk', dconv, X_col)
    db_conv = dconv.sum(axis=(0,1))
    # We don't need dX for earlier layers.
    grads = (dW_conv, db_conv, dW_fc, db_fc)
    return loss, grads

def adam_update(params, grads, state, t, lr=3e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    new_params = []
    new_state = []
    for p, g, (m, v) in zip(params, grads, state):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        p = p - lr * m_hat / (np.sqrt(v_hat) + eps)
        new_params.append(p)
        new_state.append((m, v))
    return new_params, new_state

def accuracy(logits, y):
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def train_model(X_train, y_train, X_val, y_val, lr, epochs, rng):
    n = X_train.shape[0]
    # Initialize params
    # W_conv: (F, 9)
    W_conv = rng.uniform(-1, 1, (FILTERS, KERNEL*KERNEL)).astype(np.float32) * math.sqrt(6.0 / (KERNEL*KERNEL))
    b_conv = np.zeros(FILTERS, np.float32)
    # W_fc: (C, F)
    W_fc = rng.uniform(-1, 1, (CLASSES, FILTERS)).astype(np.float32) * math.sqrt(6.0 / FILTERS)
    b_fc = np.zeros(CLASSES, np.float32)
    params = [W_conv, b_conv, W_fc, b_fc]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    best_val_acc = -1
    best_params = None
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X_train[idx], y_train[idx]
            logits, cache = forward(xb, params, IDX_COL)
            loss, grads = loss_and_grad(logits, yb, cache, params)
            t += 1
            params, state = adam_update(params, grads, state, t, lr=lr)
        # Validation accuracy
        val_logits, _ = forward(X_val, params, IDX_COL)
        val_acc = accuracy(val_logits, y_val)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = [p.copy() for p in params]
    return best_params, best_val_acc

def main():
    card = json.load(open(sys.argv[1]))
    assert card["input"]["bytes"] == 64
    assert card["classes"] == 10
    # Load data
    X_bytes = b"".join(records(sys.argv[2]))
    X = np.frombuffer(X_bytes, dtype=np.uint8).reshape(-1, 64).astype(np.float32) / 16.0  # normalize to [0,1]
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    N = X.shape[0]
    # Split: last 20% for tuning (file order)
    split = int(0.8 * N)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]
    # Precompute im2col indices (with padding zeros)
    global IDX_COL
    idx = im2col_indices(IN_H, IN_W, KERNEL, KERNEL, PAD)  # (64, 9) with -1 for padding
    IDX_COL = idx
    rng = np.random.default_rng(SEED)
    # Grid search on validation set
    best_val_acc = -1
    best_lr = LR_GRID[0]
    best_epochs = EPOCH_GRID[0]
    best_params = None
    for lr in LR_GRID:
        for epochs in EPOCH_GRID:
            params, val_acc = train_model(X_tr, y_tr, X_val, y_val, lr, epochs, rng)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_lr = lr
                best_epochs = epochs
                best_params = params
    # Retrain on full dataset with best hyperparameters
    print(f"Best lr={best_lr}, epochs={best_epochs}, val_acc={best_val_acc:.4f}", file=sys.stderr)
    final_params, _ = train_model(X, y, X_val, y_val, best_lr, best_epochs, rng)  # use val for early stopping? Just train full.
    # Actually we should train on full data without validation. We'll just run the same epochs on full data.
    # But we already have best_params from full training? The function above trains on X_tr only.
    # Let's retrain on full X,y with the same epochs.
    final_params, _ = train_model(X, y, X_val, y_val, best_lr, best_epochs, rng)
    # Save weights
    W_conv, b_conv, W_fc, b_fc = final_params
    # Format: magic, d, C, F, K
    magic = 0x564E4F43  # 'CONV' little-endian
    header = struct.pack("<5i", magic, 64, CLASSES, FILTERS, KERNEL)
    # Flatten weights in order: W_conv (F,9), b_conv (F), W_fc (C,F), b_fc (C)
    # All float32 little-endian
    weight_bytes = b""
    weight_bytes += W_conv.astype("<f4").tobytes()
    weight_bytes += b_conv.astype("<f4").tobytes()
    weight_bytes += W_fc.astype("<f4").tobytes()
    weight_bytes += b_fc.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + weight_bytes)

if __name__ == "__main__":
    main()
```

```c
/* label-conv inference: ./infer weights.bin in.rec N */
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
    if (nw < 20) { fprintf(stderr, "bad weights\n"); return 3; }
    int32_t *hd = (int32_t *)wb;
    if (hd[0] != 0x564E4F43) { fprintf(stderr, "bad magic\n"); return 3; }
    int d = hd[1], C = hd[2], F = hd[3], K = hd[4];
    if (d != 64 || C != 10 || F != 8 || K != 3) { fprintf(stderr, "bad dims\n"); return 3; }
    /* Expected floats: W_conv[F][9], b_conv[F], W_fc[C][F], b_fc[C] */
    long nfloats = (long)F * 9 + F + (long)C * F + C;
    if (nw != 20 + nfloats * 4) { fprintf(stderr, "bad weights size\n"); return 3; }
    float *pf = (float *)(wb + 20);
    float *W_conv = pf; pf += F * 9;
    float *b_conv = pf; pf += F;
    float *W_fc = pf; pf += C * F;
    float *b_fc = pf;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }
        /* Convert to float [0,1] and pad to 10x10 */
        float padded[10][10] = {{0}};
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                padded[r+1][c+1] = xi[r*8 + c] / 16.0f;
            }
        }
        /* Convolution + ReLU + GAP */
        float gap[8] = {0};
        for (int f = 0; f < F; f++) {
            float sum = 0.0f;
            for (int i_h = 0; i_h < 8; i_h++) {
                for (int i_w = 0; i_w < 8; i_w++) {
                    float val = b_conv[f];
                    for (int kh = 0; kh < 3; kh++) {
                        for (int kw = 0; kw < 3; kw++) {
                            val += padded[i_h + kh][i_w + kw] * W_conv[f*9 + kh*3 + kw];
                        }
                    }
                    if (val > 0) sum += val;
                }
            }
            gap[f] = sum / 64.0f;
        }
        /* Linear layer */
        float scores[10];
        for (int c = 0; c < C; c++) {
            float s = b_fc[c];
            for (int f = 0; f < F; f++) {
                s += gap[f] * W_fc[c*F + f];
            }
            scores[c] = s;
        }
        /* Argmax */
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```