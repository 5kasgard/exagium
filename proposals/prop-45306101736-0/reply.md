```hypothesis
Factorization Machines (FM) capture pairwise feature interactions via low-rank factorization, which suits the Adult dataset's many categorical features (education, occupation, etc.) where interactions are predictive. The model shares a single factor matrix across classes, keeping parameters small. It should beat label-linear (0.8524) on dev accuracy while staying faster than label-mlp16. If dev accuracy ≤ 0.8524 or inference cost exceeds label-linear by >2×, the hypothesis fails.
```

```python
import json, math, struct, sys

import numpy as np

# Fixed hyperparameters chosen after a small tuning on the last 20% of train.in
SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100
FACTOR_DIM = 8

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
    C = card["classes"]
    d = card["input"]["bytes"]
    types = card["input"]["types"]
    n = X.shape[0]

    # Build per-column encoding matrices (256 x k_j)
    encodings = []
    offsets = [0]
    for j, t in enumerate(types):
        col = X[:, j]
        if t == "ordinal":
            v = col.astype(np.float64)
            mu, sd = v.mean(), v.std() or 1.0
            E = ((np.arange(256, dtype=np.float64) - mu) / sd).reshape(-1, 1)
        else:
            vals = np.unique(col)
            E = np.zeros((256, len(vals)), dtype=np.float64)
            E[vals, np.arange(len(vals))] = 1.0
        encodings.append(E.astype(np.float32))
        offsets.append(offsets[-1] + E.shape[1])
    k = offsets[-1]

    # Full encoded matrix
    Z = np.zeros((n, k), dtype=np.float32)
    for j in range(d):
        off, sz = offsets[j], offsets[j+1] - offsets[j]
        Z[:, off:off+sz] = encodings[j][X[:, j]]

    # Split: last 20% for validation (tuning)
    val_start = int(n * 0.8)
    idx_train = np.arange(val_start)
    idx_val = np.arange(val_start, n)
    Z_train, y_train = Z[idx_train], y[idx_train]
    Z_val, y_val = Z[idx_val], y[idx_val]

    rng = np.random.default_rng(SEED)

    def train_fm(Z_tr, y_tr, Z_va, y_va, epochs, lr, l2, fdim):
        n_tr, k = Z_tr.shape
        # Parameters
        w0 = np.zeros(C, np.float32)
        w = (rng.uniform(-1, 1, (k, C)) * math.sqrt(6.0 / k)).astype(np.float32)
        V = (rng.uniform(-1, 1, (k, fdim)) * math.sqrt(6.0 / fdim)).astype(np.float32)
        params = [w0, w, V]
        state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
        t = 0

        def loss_acc(Zb, yb):
            S = Zb @ V
            inter = 0.5 * (np.sum(S**2, axis=1) - np.sum((Zb**2) @ (V**2), axis=1))
            logits = w0 + Zb @ w + inter[:, None]
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            loss = -np.log(probs[np.arange(len(yb)), yb] + 1e-12).mean()
            acc = (probs.argmax(axis=1) == yb).mean()
            return loss, acc, probs, S

        best_val_acc = -1.0
        best_params = None
        for ep in range(epochs):
            order = rng.permutation(n_tr)
            for s in range(0, n_tr, BATCH):
                idx = order[s:s+BATCH]
                Zb, yb = Z_tr[idx], y_tr[idx]
                loss, acc, probs, S = loss_acc(Zb, yb)
                # Gradients
                dlogits = probs
                dlogits[np.arange(len(yb)), yb] -= 1.0
                dlogits /= len(yb)
                dw0 = dlogits.sum(axis=0)
                dw = Zb.T @ dlogits + l2 * w
                dinter = dlogits.sum(axis=1)
                dS = dinter[:, None] * S
                dV = Zb.T @ dS - (Zb**2).T @ (dinter[:, None] * V) + l2 * V
                grads = [dw0, dw, dV]
                t += 1
                for i, (p, gr) in enumerate(zip(params, grads)):
                    m, v = state[i]
                    m *= 0.9; m += 0.1 * gr
                    v *= 0.999; v += 0.001 * gr * gr
                    p -= lr * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
            # Validation
            _, val_acc, _, _ = loss_acc(Z_va, y_va)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = [p.copy() for p in params]
        return best_params, best_val_acc

    # Quick tuning on validation split (fixed grid, few epochs)
    best_config = None
    best_acc = -1.0
    for fdim in [4, 8, 16]:
        for lr in [1e-3, 3e-3]:
            params, acc = train_fm(Z_train, y_train, Z_val, y_val, epochs=20, lr=lr, l2=L2, fdim=fdim)
            if acc > best_acc:
                best_acc = acc
                best_config = (fdim, lr)
                best_params = params
    fdim, lr = best_config
    w0, w, V = best_params

    # Retrain on full data with best config
    params_full, _ = train_fm(Z, y, Z_val, y_val, epochs=MAX_EPOCHS, lr=lr, l2=L2, fdim=fdim)
    w0, w, V = params_full

    # Precompute per-column lookup tables
    # For each column j, for each byte 0..255:
    #   linear[C], S[fdim], sq[1]
    tables = []
    for j in range(d):
        off, sz = offsets[j], offsets[j+1] - offsets[j]
        Ej = encodings[j]  # (256, sz)
        wj = w[off:off+sz]          # (sz, C)
        Vj = V[off:off+sz]          # (sz, fdim)
        Vj_sq = np.sum(Vj**2, axis=1)  # (sz,)
        linear_table = Ej @ wj        # (256, C)
        S_table = Ej @ Vj             # (256, fdim)
        sq_table = np.sum(Ej**2 * Vj_sq[None, :], axis=1)  # (256,)
        tables.append((linear_table, S_table, sq_table))

    # Write weights.bin
    magic = 0x464D3130  # "FM10"
    header = struct.pack("<5i", magic, d, k, fdim, C)
    out = header + w0.astype("<f4").tobytes()
    for j in range(d):
        lin, S, sq = tables[j]
        out += lin.astype("<f4").tobytes()
        out += S.astype("<f4").tobytes()
        out += sq.astype("<f4").tobytes()
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
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x464D3130) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], k = hd[2], fdim = hd[3], C = hd[4];
    if (C < 1 || C > 256) { fprintf(stderr, "bad classes\n"); return 3; }

    // Expected size: header(20) + w0(C*4) + d * 256 * (C + fdim + 1) * 4
    long expected = 20L + 4L * C + 4L * d * 256 * (C + fdim + 1);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *ptr = (const float *)(hd + 5); // after header (5 int32)
    const float *w0 = ptr; ptr += C;
    // tables: for each column, 256 entries of (linear[C], S[fdim], sq[1])
    const float *tables = ptr;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *linear_sum = calloc(C, sizeof(float));
    float *S_sum = calloc(fdim, sizeof(float));
    float sq_sum;
    unsigned char best;

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(linear_sum, 0, C * sizeof(float));
        memset(S_sum, 0, fdim * sizeof(float));
        sq_sum = 0.0f;

        for (int j = 0; j < d; j++) {
            const float *col_table = tables + (long)j * 256 * (C + fdim + 1);
            const float *entry = col_table + xi[j] * (C + fdim + 1);
            const float *lin = entry;
            const float *S = entry + C;
            const float *sq = entry + C + fdim;
            for (int c = 0; c < C; c++) linear_sum[c] += lin[c];
            for (int f = 0; f < fdim; f++) S_sum[f] += S[f];
            sq_sum += *sq;
        }

        float inter = 0.0f;
        for (int f = 0; f < fdim; f++) inter += S_sum[f] * S_sum[f];
        inter = 0.5f * (inter - sq_sum);

        best = 0;
        float best_logit = w0[0] + linear_sum[0] + inter;
        for (int c = 1; c < C; c++) {
            float logit = w0[c] + linear_sum[c] + inter;
            if (logit > best_logit) { best_logit = logit; best = (unsigned char)c; }
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```