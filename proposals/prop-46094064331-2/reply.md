```hypothesis
Family: label-emb-mlp (per-position byte embeddings + optional ReLU hidden layer).
Why: The 16-byte context has strong positional structure; factorizing the input weight matrix via embeddings (16×256×E) drastically reduces parameters vs. one-hot linear (4096×65), improving generalization on 24k examples. A small hidden layer adds nonlinearity. Grid search on the tuning split (last 20%) picks (E, H); final model trains on all data. Expected to exceed label-linear's 0.3375 accuracy with comparable or lower CPU cost (embedding lookup + small matmul). Failure if dev accuracy ≤ 0.3375 or Ir/item > 7359.
```
```python
import json, math, struct, sys
import numpy as np

# Grid of (embedding_dim, hidden_size). H=0 means no hidden layer (linear on embeddings).
CONFIGS = [(8, 0), (16, 0), (32, 0), (8, 16), (16, 16), (8, 32)]
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100
MAGIC = 0x454D424C  # "EMBL"

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_eval_config(rng, X_tr, y_tr, X_val, y_val, d, C, E, H):
    n_tr = X_tr.shape[0]
    # init
    embeddings = [rng.normal(0, 1/math.sqrt(E), size=(256, E)).astype(np.float32) for _ in range(d)]
    if H > 0:
        W1 = rng.normal(0, math.sqrt(2.0/(d*E)), size=(d*E, H)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = rng.normal(0, math.sqrt(2.0/H), size=(H, C)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        param_arrays = embeddings + [W1, b1, W2, b2]
    else:
        W_out = rng.normal(0, math.sqrt(2.0/(d*E)), size=(d*E, C)).astype(np.float32)
        b_out = np.zeros(C, np.float32)
        param_arrays = embeddings + [W_out, b_out]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in param_arrays]

    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_tr))
    t = 0
    for _ in range(epochs):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X_tr[idx], y_tr[idx]
            emb_concat = np.empty((len(idx), d*E), np.float32)
            for j in range(d):
                emb_concat[:, j*E:(j+1)*E] = embeddings[j][xb[:, j]]
            if H > 0:
                hidden = emb_concat @ W1 + b1
                np.maximum(hidden, 0, out=hidden)
                logits = hidden @ W2 + b2
            else:
                logits = emb_concat @ W_out + b_out
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad_logits = probs / len(yb)
            if H > 0:
                grad_W2 = hidden.T @ grad_logits
                grad_b2 = grad_logits.sum(0)
                grad_hidden = grad_logits @ W2.T
                grad_hidden[hidden <= 0] = 0
                grad_W1 = emb_concat.T @ grad_hidden
                grad_b1 = grad_hidden.sum(0)
                grad_emb_concat = grad_hidden @ W1.T
                grad_embeddings = [np.zeros_like(e) for e in embeddings]
                for j in range(d):
                    grad_embeddings[j][xb[:, j]] += grad_emb_concat[:, j*E:(j+1)*E]
                grads = grad_embeddings + [grad_W1, grad_b1, grad_W2, grad_b2]
            else:
                grad_W_out = emb_concat.T @ grad_logits
                grad_b_out = grad_logits.sum(0)
                grad_emb_concat = grad_logits @ W_out.T
                grad_embeddings = [np.zeros_like(e) for e in embeddings]
                for j in range(d):
                    grad_embeddings[j][xb[:, j]] += grad_emb_concat[:, j*E:(j+1)*E]
                grads = grad_embeddings + [grad_W_out, grad_b_out]
            for i, p in enumerate(param_arrays):
                grads[i] += L2 * p
            t += 1
            for i, (p, gr) in enumerate(zip(param_arrays, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9**t)
                v_hat = v / (1 - 0.999**t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
    # eval
    val_correct = 0
    n_val = X_val.shape[0]
    for s in range(0, n_val, BATCH):
        xb, yb = X_val[s:s+BATCH], y_val[s:s+BATCH]
        emb_concat = np.empty((len(xb), d*E), np.float32)
        for j in range(d):
            emb_concat[:, j*E:(j+1)*E] = embeddings[j][xb[:, j]]
        if H > 0:
            hidden = emb_concat @ W1 + b1
            np.maximum(hidden, 0, out=hidden)
            logits = hidden @ W2 + b2
        else:
            logits = emb_concat @ W_out + b_out
        val_correct += (logits.argmax(axis=1) == yb).sum()
    return val_correct / n_val, param_arrays

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]
    C = card["classes"]
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]
    val_size = n // 5
    tr_size = n - val_size
    X_tr, y_tr = X[:tr_size], y[:tr_size]
    X_val, y_val = X[tr_size:], y[tr_size:]

    base_rng = np.random.default_rng(SEED)
    best_acc, best_cfg, best_params = -1, None, None
    for idx, (E, H) in enumerate(CONFIGS):
        cfg_rng = np.random.default_rng(SEED + idx + 1)
        acc, params = train_eval_config(cfg_rng, X_tr, y_tr, X_val, y_val, d, C, E, H)
        if acc > best_acc:
            best_acc, best_cfg, best_params = acc, (E, H), [p.copy() for p in params]

    E, H = best_cfg
    # Retrain on full data with fresh seed
    final_rng = np.random.default_rng(SEED + 1000 + E*10 + H)
    embeddings = [final_rng.normal(0, 1/math.sqrt(E), size=(256, E)).astype(np.float32) for _ in range(d)]
    if H > 0:
        W1 = final_rng.normal(0, math.sqrt(2.0/(d*E)), size=(d*E, H)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = final_rng.normal(0, math.sqrt(2.0/H), size=(H, C)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        param_arrays = embeddings + [W1, b1, W2, b2]
    else:
        W_out = final_rng.normal(0, math.sqrt(2.0/(d*E)), size=(d*E, C)).astype(np.float32)
        b_out = np.zeros(C, np.float32)
        param_arrays = embeddings + [W_out, b_out]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in param_arrays]
    t = 0
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = final_rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb, yb = X[idx], y[idx]
            emb_concat = np.empty((len(idx), d*E), np.float32)
            for j in range(d):
                emb_concat[:, j*E:(j+1)*E] = embeddings[j][xb[:, j]]
            if H > 0:
                hidden = emb_concat @ W1 + b1
                np.maximum(hidden, 0, out=hidden)
                logits = hidden @ W2 + b2
            else:
                logits = emb_concat @ W_out + b_out
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            grad_logits = probs / len(yb)
            if H > 0:
                grad_W2 = hidden.T @ grad_logits
                grad_b2 = grad_logits.sum(0)
                grad_hidden = grad_logits @ W2.T
                grad_hidden[hidden <= 0] = 0
                grad_W1 = emb_concat.T @ grad_hidden
                grad_b1 = grad_hidden.sum(0)
                grad_emb_concat = grad_hidden @ W1.T
                grad_embeddings = [np.zeros_like(e) for e in embeddings]
                for j in range(d):
                    grad_embeddings[j][xb[:, j]] += grad_emb_concat[:, j*E:(j+1)*E]
                grads = grad_embeddings + [grad_W1, grad_b1, grad_W2, grad_b2]
            else:
                grad_W_out = emb_concat.T @ grad_logits
                grad_b_out = grad_logits.sum(0)
                grad_emb_concat = grad_logits @ W_out.T
                grad_embeddings = [np.zeros_like(e) for e in embeddings]
                for j in range(d):
                    grad_embeddings[j][xb[:, j]] += grad_emb_concat[:, j*E:(j+1)*E]
                grads = grad_embeddings + [grad_W_out, grad_b_out]
            for i, p in enumerate(param_arrays):
                grads[i] += L2 * p
            t += 1
            for i, (p, gr) in enumerate(zip(param_arrays, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9**t)
                v_hat = v / (1 - 0.999**t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Pack weights
    out = struct.pack("<5i", MAGIC, d, E, H, C)
    for emb in embeddings:
        out += emb.astype("<f4").tobytes()
    if H > 0:
        out += W1.astype("<f4").tobytes()
        out += b1.astype("<f4").tobytes()
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()
    else:
        out += W_out.astype("<f4").tobytes()
        out += b_out.astype("<f4").tobytes()
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x454D424C) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (C < 1 || C > 256 || d != 16) { fprintf(stderr, "bad header dims\n"); return 3; }
    long num_floats = (long)d * 256 * E;
    if (H > 0) {
        num_floats += (long)d * E * H + H + (long)H * C + C;
    } else {
        num_floats += (long)d * E * C + C;
    }
    if (nw != 20 + 4 * num_floats) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *emb = (const float *)(hd + 5);
    const float *W1, *b1, *W2, *b2, *W_out, *b_out;
    if (H > 0) {
        W1 = emb + (long)d * 256 * E;
        b1 = W1 + (long)d * E * H;
        W2 = b1 + H;
        b2 = W2 + (long)H * C;
    } else {
        W_out = emb + (long)d * 256 * E;
        b_out = W_out + (long)d * E * C;
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *concat = malloc(sizeof(float) * d * E);
    float *hidden = H ? malloc(sizeof(float) * H) : NULL;
    float *logits = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // concatenate embeddings
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(concat + j * E, e, E * sizeof(float));
        }
        if (H > 0) {
            // hidden = ReLU(concat * W1 + b1)
            for (int h = 0; h < H; h++) {
                float sum = b1[h];
                for (int k = 0; k < d * E; k++) sum += concat[k] * W1[(long)k * H + h];
                hidden[h] = sum > 0 ? sum : 0;
            }
            // logits = hidden * W2 + b2
            for (int c = 0; c < C; c++) {
                float sum = b2[c];
                for (int h = 0; h < H; h++) sum += hidden[h] * W2[(long)h * C + c];
                logits[c] = sum;
            }
        } else {
            // logits = concat * W_out + b_out
            for (int c = 0; c < C; c++) {
                float sum = b_out[c];
                for (int k = 0; k < d * E; k++) sum += concat[k] * W_out[(long)k * C + c];
                logits[c] = sum;
            }
        }
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```