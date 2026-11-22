```hypothesis
Factorization Machines (FM) capture pairwise feature interactions efficiently via low-rank embeddings, which are crucial for tabular data like Adult where combinations of categorical features (e.g., education×occupation) strongly predict income. With a small latent dimension (k=4) the model adds only ~80 KB to the linear baseline and computes interactions in O(d·k) time, keeping inference fast. If dev accuracy does not exceed label-linear (0.8524) or inference cost exceeds ~1000 Ir/item, the approach fails.
```

```python
"""label-fm: Factorization Machine for tabular label tasks.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Encodes ordinal as z-scored float, categorical as category index (0-255).
FM: score_c = bias_c + linear_c + 0.5*(|sum v_f|^2 - sum |v_f|^2) per class.
Tunes k in {2,4,8} and lr in {0.01,0.03} on last 20% of train, then retrains on all.
Exports: magic 0x464D4C31, d, k, C; per-class: bias, then per-field weights/embeddings."""
import json, math, struct, sys

import numpy as np

SEED = 1
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
L2 = 1e-4
TUNE_EPOCHS = 10
GRID_K = [2, 4, 8]
GRID_LR = [0.01, 0.03]

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def build_model(d, field_types, k, C, rng):
    # returns dict of params per class
    models = []
    for _ in range(C):
        m = {'bias': 0.0}
        ord_w = []; ord_v = []
        cat_w = []; cat_v = []
        for t in field_types:
            if t == "ordinal":
                ord_w.append(rng.normal(0, 0.01))
                ord_v.append(rng.normal(0, 0.01, k))
            else:
                cat_w.append(rng.normal(0, 0.01, 256))
                cat_v.append(rng.normal(0, 0.01, (256, k)))
        m['ord_w'] = np.array(ord_w, np.float32)
        m['ord_v'] = np.array(ord_v, np.float32)
        m['cat_w'] = np.array(cat_w, np.float32)
        m['cat_v'] = np.array(cat_v, np.float32)
        models.append(m)
    return models

def forward(models, X_ord, X_cat, k, C):
    # X_ord: (n, n_ord), X_cat: (n, n_cat) int indices
    n = X_ord.shape[0]
    n_ord = X_ord.shape[1]
    n_cat = X_cat.shape[1]
    scores = np.zeros((n, C), np.float32)
    for c in range(C):
        m = models[c]
        lin = m['bias'] + X_ord @ m['ord_w']
        for j in range(n_cat):
            lin += m['cat_w'][j, X_cat[:, j]]
        # interaction
        sum_v = np.zeros((n, k), np.float32)
        sum_v += X_ord @ m['ord_v']  # (n,k)
        sq_norm = np.sum(X_ord**2 @ (m['ord_v']**2), axis=1)  # (n,)
        for j in range(n_cat):
            v = m['cat_v'][j, X_cat[:, j]]  # (n,k)
            sum_v += v
            sq_norm += np.sum(v**2, axis=1)
        inter = 0.5 * (np.sum(sum_v**2, axis=1) - sq_norm)
        scores[:, c] = lin + inter
    return scores

def backward(models, X_ord, X_cat, y, scores, k, C, lr, l2, t, states):
    # scores: (n,C) pre-softmax
    n = X_ord.shape[0]
    n_ord = X_ord.shape[1]
    n_cat = X_cat.shape[1]
    # softmax grad
    P = np.exp(scores - scores.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    P[np.arange(n), y] -= 1.0
    P /= n
    # gradients per class
    grads = []
    for c in range(C):
        g = P[:, c]  # (n,)
        m = models[c]
        # linear grads
        g_bias = g.sum()
        g_ord_w = X_ord.T @ g
        g_cat_w = [np.zeros(256, np.float32) for _ in range(n_cat)]
        for j in range(n_cat):
            np.add.at(g_cat_w[j], X_cat[:, j], g)
        # interaction grads
        sum_v = np.zeros((n, k), np.float32)
        sum_v += X_ord @ m['ord_v']
        for j in range(n_cat):
            sum_v += m['cat_v'][j, X_cat[:, j]]
        # grad wrt sum_v: (sum_v) * g[:,None]
        g_sum_v = sum_v * g[:, None]  # (n,k)
        # grad ord_v
        g_ord_v = X_ord.T @ g_sum_v - X_ord.T @ (g[:, None] * (X_ord @ m['ord_v']))
        # Actually derivative of 0.5*(|sum_v|^2 - sum |v_f|^2) wrt v_f = (sum_v - v_f) * x_f
        # So for ordinal: (sum_v - ord_v * x_f) * x_f * g
        # We'll compute per feature
        g_ord_v = np.zeros_like(m['ord_v'])
        for j in range(n_ord):
            x = X_ord[:, j]
            v_f = m['ord_v'][j]
            g_ord_v[j] = x @ (g_sum_v - g[:, None] * (x[:, None] * v_f))
        # grad cat_v
        g_cat_v = [np.zeros((256, k), np.float32) for _ in range(n_cat)]
        for j in range(n_cat):
            for cat in range(256):
                mask = (X_cat[:, j] == cat)
                if not mask.any(): continue
                x = mask.astype(np.float32)
                v_f = m['cat_v'][j, cat]
                g_cat_v[j, cat] = x @ (g_sum_v[mask] - g[mask, None] * v_f)
        # L2
        g_ord_w += l2 * m['ord_w']
        g_ord_v += l2 * m['ord_v']
        for j in range(n_cat):
            g_cat_w[j] += l2 * m['cat_w'][j]
            g_cat_v[j] += l2 * m['cat_v'][j]
        grads.append((g_bias, g_ord_w, g_ord_v, g_cat_w, g_cat_v))
    # Adam update
    for c in range(C):
        m = models[c]
        g_bias, g_ord_w, g_ord_v, g_cat_w, g_cat_v = grads[c]
        m_state = states[c]
        # bias
        m_b, v_b = m_state['bias']
        m_b = 0.9 * m_b + 0.1 * g_bias
        v_b = 0.999 * v_b + 0.001 * g_bias * g_bias
        m['bias'] -= lr * (m_b / (1 - 0.9**t)) / (math.sqrt(v_b / (1 - 0.999**t)) + 1e-8)
        m_state['bias'] = (m_b, v_b)
        # ord_w
        m_m, v_v = m_state['ord_w']
        m_m = 0.9 * m_m + 0.1 * g_ord_w
        v_v = 0.999 * v_v + 0.001 * g_ord_w * g_ord_w
        m['ord_w'] -= lr * (m_m / (1 - 0.9**t)) / (np.sqrt(v_v / (1 - 0.999**t)) + 1e-8)
        m_state['ord_w'] = (m_m, v_v)
        # ord_v
        m_m, v_v = m_state['ord_v']
        m_m = 0.9 * m_m + 0.1 * g_ord_v
        v_v = 0.999 * v_v + 0.001 * g_ord_v * g_ord_v
        m['ord_v'] -= lr * (m_m / (1 - 0.9**t)) / (np.sqrt(v_v / (1 - 0.999**t)) + 1e-8)
        m_state['ord_v'] = (m_m, v_v)
        # cat_w
        m_m, v_v = m_state['cat_w']
        for j in range(n_cat):
            m_m[j] = 0.9 * m_m[j] + 0.1 * g_cat_w[j]
            v_v[j] = 0.999 * v_v[j] + 0.001 * g_cat_w[j] * g_cat_w[j]
            m['cat_w'][j] -= lr * (m_m[j] / (1 - 0.9**t)) / (np.sqrt(v_v[j] / (1 - 0.999**t)) + 1e-8)
        m_state['cat_w'] = (m_m, v_v)
        # cat_v
        m_m, v_v = m_state['cat_v']
        for j in range(n_cat):
            m_m[j] = 0.9 * m_m[j] + 0.1 * g_cat_v[j]
            v_v[j] = 0.999 * v_v[j] + 0.001 * g_cat_v[j] * g_cat_v[j]
            m['cat_v'][j] -= lr * (m_m[j] / (1 - 0.9**t)) / (np.sqrt(v_v[j] / (1 - 0.999**t)) + 1e-8)
        m_state['cat_v'] = (m_m, v_v)

def init_states(models, k):
    states = []
    for m in models:
        n_ord = m['ord_w'].shape[0]
        n_cat = m['cat_w'].shape[0]
        s = {
            'bias': (0.0, 0.0),
            'ord_w': (np.zeros_like(m['ord_w']), np.zeros_like(m['ord_w'])),
            'ord_v': (np.zeros_like(m['ord_v']), np.zeros_like(m['ord_v'])),
            'cat_w': (np.zeros_like(m['cat_w']), np.zeros_like(m['cat_w'])),
            'cat_v': (np.zeros_like(m['cat_v']), np.zeros_like(m['cat_v'])),
        }
        states.append(s)
    return states

def train_epoch(models, X_ord, X_cat, y, k, C, lr, l2, t, states, batch_size, rng):
    n = X_ord.shape[0]
    order = rng.permutation(n)
    for s in range(0, n, batch_size):
        idx = order[s:s+batch_size]
        x_ord = X_ord[idx]; x_cat = X_cat[idx]; yy = y[idx]
        scores = forward(models, x_ord, x_cat, k, C)
        t[0] += 1
        backward(models, x_ord, x_cat, yy, scores, k, C, lr, l2, t[0], states)

def evaluate(models, X_ord, X_cat, y, k, C):
    scores = forward(models, X_ord, X_cat, k, C)
    pred = np.argmax(scores, axis=1)
    return (pred == y).mean()

def main():
    card = json.load(open(sys.argv[1]))
    X_raw = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    field_types = card["input"]["types"]
    C = card["classes"]
    n = X_raw.shape[0]
    # split ordinal / categorical
    ord_idx = [i for i,t in enumerate(field_types) if t=="ordinal"]
    cat_idx = [i for i,t in enumerate(field_types) if t=="categorical"]
    X_ord = X_raw[:, ord_idx].astype(np.float64)
    X_cat = X_raw[:, cat_idx].astype(np.int32)
    # z-score ordinal on training set
    mu = X_ord.mean(axis=0)
    sd = X_ord.std(axis=0)
    sd[sd==0] = 1.0
    X_ord = (X_ord - mu) / sd
    # convert to float32
    X_ord = X_ord.astype(np.float32)
    # split tune (last 20%)
    tune_sz = n // 5
    tr_ord, tu_ord = X_ord[:-tune_sz], X_ord[-tune_sz:]
    tr_cat, tu_cat = X_cat[:-tune_sz], X_cat[-tune_sz:]
    tr_y, tu_y = y[:-tune_sz], y[-tune_sz:]
    # grid search
    best_acc = -1
    best_k = 4
    best_lr = 0.01
    rng_base = np.random.default_rng(SEED)
    for k in GRID_K:
        for lr in GRID_LR:
            rng = np.random.default_rng(rng_base.integers(2**32))
            models = build_model(d, field_types, k, C, rng)
            states = init_states(models, k)
            t = [0]
            for _ in range(TUNE_EPOCHS):
                train_epoch(models, tr_ord, tr_cat, tr_y, k, C, lr, L2, t, states, BATCH, rng)
            acc = evaluate(models, tu_ord, tu_cat, tu_y, k, C)
            if acc > best_acc:
                best_acc, best_k, best_lr = acc, k, lr
    # retrain on all data with best config
    rng = np.random.default_rng(rng_base.integers(2**32))
    models = build_model(d, field_types, best_k, C, rng)
    states = init_states(models, best_k)
    t = [0]
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        train_epoch(models, X_ord, X_cat, y, best_k, C, best_lr, L2, t, states, BATCH, rng)
    # export weights
    # header: magic, d, k, C
    out = struct.pack("<4i", 0x464D4C31, d, best_k, C)
    # per class
    for c in range(C):
        m = models[c]
        out += struct.pack("<f", m['bias'])
        ord_ptr = 0
        cat_ptr = 0
        for t in field_types:
            if t == "ordinal":
                out += struct.pack("<f", m['ord_w'][ord_ptr])
                out += m['ord_v'][ord_ptr].astype("<f4").tobytes()
                ord_ptr += 1
            else:
                out += m['cat_w'][cat_ptr].astype("<f4").tobytes()
                out += m['cat_v'][cat_ptr].astype("<f4").tobytes()
                cat_ptr += 1
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```

```c
/* label-fm inference: ./infer weights.bin in.rec N
 * Factorization Machine with per-field embeddings. */
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
    if (nw < 16 || hd[0] != 0x464D4C31) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], C = hd[3];
    if (C != 2 || k < 1 || k > 32 || d != 14) { fprintf(stderr, "bad header\n"); return 3; }
    const float *p = (const float *)(hd + 4);
    // We'll parse the weights into structured pointers.
    // Layout per class: bias, then per field (ordinal: w + v[k]; categorical: w[256] + v[256][k])
    // We need field types from the card (hardcoded for adult):
    // ordinal, categorical, ordinal, categorical, ordinal, categorical, categorical, categorical, categorical, categorical, ordinal, ordinal, ordinal, categorical
    static const int is_ord[14] = {1,0,1,0,1,0,0,0,0,0,1,1,1,0};
    // Count ordinals and categoricals
    int n_ord = 0, n_cat = 0;
    for (int i=0;i<d;i++) if (is_ord[i]) n_ord++; else n_cat++;
    // Expected size:
    long expected = 16;
    for (int c=0;c<C;c++) {
        expected += 4; // bias
        for (int i=0;i<d;i++) {
            if (is_ord[i]) expected += 4 * (1 + k);
            else expected += 4 * (256 * (1 + k));
        }
    }
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    // Pointers for each class
    struct ClassModel {
        float bias;
        const float *ord_w;  // [n_ord]
        const float *ord_v;  // [n_ord][k]
        const float *cat_w;  // [n_cat][256]
        const float *cat_v;  // [n_cat][256][k]
    } cm[2];

    for (int c=0;c<C;c++) {
        cm[c].bias = *p++;
        cm[c].ord_w = p; p += n_ord;
        cm[c].ord_v = p; p += n_ord * k;
        cm[c].cat_w = p; p += n_cat * 256;
        cm[c].cat_v = p; p += n_cat * 256 * k;
    }

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *sum_v = malloc(C * k * sizeof(float));
    for (long i = 0; i < N; i++) {
        uint32_t len; const unsigned char *xi = next_rec(x, nx, &pos, &len);
        if ((int)len != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, len, d); return 3; }
        float best_score = -1e30f;
        unsigned char best_cls = 0;
        for (int c=0;c<C;c++) {
            float score = cm[c].bias;
            // linear part
            int oi=0, ci=0;
            for (int j=0;j<d;j++) {
                if (is_ord[j]) {
                    score += cm[c].ord_w[oi] * (float)xi[j];
                    oi++;
                } else {
                    score += cm[c].cat_w[ci * 256 + xi[j]];
                    ci++;
                }
            }
            // interaction part
            // zero sum_v
            for (int u=0;u<k;u++) sum_v[c*k + u] = 0.0f;
            float sq_norm = 0.0f;
            oi=0; ci=0;
            for (int j=0;j<d;j++) {
                if (is_ord[j]) {
                    float xv = (float)xi[j];
                    const float *v = cm[c].ord_v + oi * k;
                    for (int u=0;u<k;u++) {
                        float t = v[u] * xv;
                        sum_v[c*k + u] += t;
                        sq_norm += t * t;
                    }
                    oi++;
                } else {
                    const float *v = cm[c].cat_v + (ci * 256 + xi[j]) * k;
                    for (int u=0;u<k;u++) {
                        float t = v[u];
                        sum_v[c*k + u] += t;
                        sq_norm += t * t;
                    }
                    ci++;
                }
            }
            float dot = 0.0f;
            for (int u=0;u<k;u++) dot += sum_v[c*k + u] * sum_v[c*k + u];
            score += 0.5f * (dot - sq_norm);
            if (c == 0 || score > best_score) {
                best_score = score;
                best_cls = c;
            }
        }
        put_rec(stdout, &best_cls, 1);
    }
    return 0;
}
```