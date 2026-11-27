```hypothesis
family: label-conv
A tiny 2D CNN (one 3×3 conv layer with K filters, ReLU, global average pooling, linear classifier) exploits the 8×8 spatial structure of optdigits. It should surpass label-linear (0.961) and approach label-mlp16 (0.977) with fewer ops/item than label-mlp16. Failure: dev accuracy ≤ 0.965 or inference cost ≥ label-mlp16.
```
```python
"""label-conv: tiny 2D CNN for 8x8 digit images.
python3 -s -P train.py card.json train.in train.key weights.bin
Architecture: input (8,8,1) -> Conv3x3 K filters pad=1 -> ReLU -> GlobalAvgPool -> Linear(10) -> logits.
Tunes K on last 20% of train, then retrains on all. Deterministic (seed=1), single thread, numpy only."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1e-3, 1e-4, 64, 300000, 200
K_CHOICES = [4, 8, 12, 16]  # filters to try on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def init_params(K, rng):
    # Conv: K x 3 x 3 x 1, He init for ReLU
    Wc = (rng.normal(0, math.sqrt(2.0 / 9), (K, 3, 3, 1))).astype(np.float32)
    bc = np.zeros(K, np.float32)
    # FC: K -> 10, Xavier
    Wf = (rng.uniform(-1, 1, (K, 10)) * math.sqrt(6.0 / K)).astype(np.float32)
    bf = np.zeros(10, np.float32)
    return Wc, bc, Wf, bf


def forward(X, Wc, bc, Wf, bf):
    # X: (N, 8, 8, 1)
    N = X.shape[0]
    K = Wc.shape[0]
    # Conv 3x3 pad=1 -> (N, 8, 8, K)
    # im2col for simplicity (small sizes)
    Xpad = np.pad(X, ((0,0),(1,1),(1,1),(0,0)), mode='constant')
    col = np.lib.stride_tricks.sliding_window_view(Xpad, (3,3,1), axis=(1,2,3))
    # col shape: (N, 8, 8, 3, 3, 1)
    col = col.reshape(N, 64, 9)
    Wc_row = Wc.reshape(K, 9)
    conv = col @ Wc_row.T + bc  # (N, 64, K)
    conv = conv.reshape(N, 8, 8, K)
    h = np.maximum(conv, 0)  # ReLU
    # Global average pool -> (N, K)
    gap = h.mean(axis=(1,2))
    # FC
    logits = gap @ Wf + bf
    return logits, h, gap


def loss_grad(logits, y):
    # softmax cross-entropy gradient
    N = logits.shape[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    probs[np.arange(N), y] -= 1.0
    return (probs / N).astype(np.float32)


def backward(X, h, gap, dlogits, Wc, Wf, L2):
    N = X.shape[0]
    K = Wf.shape[0]
    # FC grads
    dWf = gap.T @ dlogits + L2 * Wf
    dbf = dlogits.sum(axis=0)
    dgap = dlogits @ Wf.T  # (N, K)
    # GAP grad: dgap / (8*8) broadcast to (N,8,8,K)
    dh = dgap[:, None, None, :] / 64.0
    # ReLU grad
    dh[h == 0] = 0
    # Conv grad: im2col again
    Xpad = np.pad(X, ((0,0),(1,1),(1,1),(0,0)), mode='constant')
    col = np.lib.stride_tricks.sliding_window_view(Xpad, (3,3,1), axis=(1,2,3))
    col = col.reshape(N, 64, 9)
    dh_col = dh.reshape(N, 64, K)
    dWc = dh_col.transpose(0,2,1).reshape(N, K, 64) @ col  # (N, K, 9) -> sum over N
    dWc = dWc.sum(axis=0).reshape(K, 3, 3, 1) + L2 * Wc
    dbc = dh_col.sum(axis=(0,1))
    return dWc, dbc, dWf, dbf


def accuracy(logits, y):
    return (logits.argmax(axis=1) == y).mean()


def train_split(X, y, K, epochs, rng):
    Wc, bc, Wf, bf = init_params(K, rng)
    # Adam state
    params = [Wc, bc, Wf, bf]
    ms = [np.zeros_like(p) for p in params]
    vs = [np.zeros_like(p) for p in params]
    t = 0
    n = X.shape[0]
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X[idx], y[idx]
            logits, h, gap = forward(xb, Wc, bc, Wf, bf)
            dlogits = loss_grad(logits, yb)
            dWc, dbc, dWf, dbf = backward(xb, h, gap, dlogits, Wc, Wf, L2)
            grads = [dWc, dbc, dWf, dbf]
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                ms[i] = 0.9 * ms[i] + 0.1 * g
                vs[i] = 0.999 * vs[i] + 0.001 * g * g
                m_hat = ms[i] / (1 - 0.9**t)
                v_hat = vs[i] / (1 - 0.999**t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    return Wc, bc, Wf, bf


def main():
    card = json.load(open(sys.argv[1]))
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, 64)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    # Normalize to [0,1] (pixel range 0..16)
    X = (X_bytes.astype(np.float32) / 16.0).reshape(-1, 8, 8, 1)

    n = X.shape[0]
    split = int(0.8 * n)
    idx = np.arange(n)
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    train_idx, tune_idx = idx[:split], idx[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_tune, y_tune = X[tune_idx], y[tune_idx]

    # Tune K
    best_K, best_acc = K_CHOICES[0], -1
    tune_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / len(X_train)))
    for K in K_CHOICES:
        Wc, bc, Wf, bf = train_split(X_train, y_train, K, tune_epochs, np.random.default_rng(SEED))
        logits, _, _ = forward(X_tune, Wc, bc, Wf, bf)
        acc = accuracy(logits, y_tune)
        if acc > best_acc:
            best_acc, best_K = acc, K

    # Retrain on full data
    full_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    Wc, bc, Wf, bf = train_split(X, y, best_K, full_epochs, np.random.default_rng(SEED))

    # Save weights: magic, H,W,C_in, K, kH,kW, then arrays
    magic = 0x434F4E56  # 'CONV'
    H, W, C_in = 8, 8, 1
    kH, kW = 3, 3
    K = best_K
    header = struct.pack("<iiiiiii", magic, H, W, C_in, K, kH, kW)
    # All float32 little-endian
    body = (Wc.astype("<f4").tobytes() + bc.astype("<f4").tobytes() +
            Wf.astype("<f4").tobytes() + bf.astype("<f4").tobytes())
    open(sys.argv[4], "wb").write(header + body)


if __name__ == "__main__":
    main()
```
```c
/* label-conv inference: ./infer weights.bin in.rec N
 * Tiny 2D CNN: 8x8x1 -> Conv3x3 K filters pad=1 -> ReLU -> GlobalAvgPool -> Linear(10) -> argmax.
 * Weight format: magic(0x434F4E56), H,W,C_in, K, kH,kW, then float32 arrays:
 *   Wc[K][kH][kW][C_in], bc[K], Wf[K][10], bf[10]. All little-endian. */
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
    unsigned char h[4] = {k&255, k>>8&255, k>>16&255, k>>24};
    fwrite(h,1,4,f); fwrite(r,1,k,f);
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wb = slurp(argv[1], &nw);
    if (nw < 28) { fprintf(stderr, "weights too small\n"); return 3; }
    int32_t *hdr = (int32_t*)wb;
    if (hdr[0] != 0x434F4E56) { fprintf(stderr, "bad magic\n"); return 3; }
    int H = hdr[1], W = hdr[2], C_in = hdr[3], K = hdr[4], kH = hdr[5], kW = hdr[6];
    if (!(H==8 && W==8 && C_in==1 && kH==3 && kW==3 && K>=1 && K<=64)) {
        fprintf(stderr, "bad dims\n"); return 3; }
    size_t expected = 28
        + (size_t)K * kH * kW * C_in * 4
        + (size_t)K * 4
        + (size_t)K * 10 * 4
        + 10 * 4;
    if (nw != (long)expected) { fprintf(stderr, "bad weights size %ld != %zu\n", nw, expected); return 3; }

    const float *p = (const float*)(wb + 28);
    const float *Wc = p; p += K * 9;
    const float *bc = p; p += K;
    const float *Wf = p; p += K * 10;
    const float *bf = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *conv_out = malloc(8*8*K*sizeof(float));
    float *gap = malloc(K*sizeof(float));
    float *logits = malloc(10*sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if (k != 64) { fprintf(stderr, "item %ld is %u bytes, not 64\n", i, k); return 3; }

        // Input: 8x8x1, normalized by 16
        float in[8][8];
        for (int r = 0; r < 8; r++)
            for (int c = 0; c < 8; c++)
                in[r][c] = xi[r*8 + c] / 16.0f;

        // Conv 3x3 pad=1 -> 8x8xK
        for (int fi = 0; fi < K; fi++) {
            for (int r = 0; r < 8; r++) {
                for (int c = 0; c < 8; c++) {
                    float sum = bc[fi];
                    for (int dr = 0; dr < 3; dr++) {
                        int rr = r + dr - 1;
                        for (int dc = 0; dc < 3; dc++) {
                            int cc = c + dc - 1;
                            float val = (rr>=0 && rr<8 && cc>=0 && cc<8) ? in[rr][cc] : 0.0f;
                            sum += val * Wc[((fi*3 + dr)*3 + dc)*C_in];
                        }
                    }
                    conv_out[(r*8 + c)*K + fi] = sum > 0 ? sum : 0; // ReLU
                }
            }
        }

        // Global average pool
        for (int fi = 0; fi < K; fi++) {
            float sum = 0;
            for (int idx = 0; idx < 64; idx++) sum += conv_out[idx*K + fi];
            gap[fi] = sum / 64.0f;
        }

        // FC
        for (int c = 0; c < 10; c++) {
            float sum = bf[c];
            for (int fi = 0; fi < K; fi++) sum += gap[fi] * Wf[fi*10 + c];
            logits[c] = sum;
        }

        // Argmax
        unsigned char best = 0;
        for (int c = 1; c < 10; c++) if (logits[c] > logits[best]) best = c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```