```hypothesis
Family: label-minigru — a tiny GRU (32-dim embeddings, 32 hidden units) processing the 16-byte sequence step-by-step.
Why: Character-level language modeling benefits from sequential state; a GRU captures n-gram statistics and longer dependencies better than bag-of-positions linear/MLP models, while staying small enough for fast inference.
Failure signal: If dev accuracy ≤ 0.34 (label-linear) or instructions/item ≥ 150k (label-gbdt territory), the complexity isn't justified.
```
```python
"""label-minigru: tiny GRU for next-byte prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single-thread, stdlib+numpy only.
Exports: magic 0x47525531, d, E, H, C; embeddings[256][E]; GRU weights; output weights."""
import json, math, struct, sys

import numpy as np

E = 32          # embedding dim
H = 32          # GRU hidden dim
SEED = 1
LR = 1e-3
L2 = 1e-5
BATCH = 64
EPOCHS = 20
CLIP = 1.0

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def gru_forward(x, h, Wz, Wr, Wh, bz, br, bh):
    # x: (B, E), h: (B, H)
    # concat = [x, h] -> (B, E+H)
    concat = np.concatenate([x, h], axis=1)
    z = 1.0 / (1.0 + np.exp(-(concat @ Wz + bz)))
    r = 1.0 / (1.0 + np.exp(-(concat @ Wr + br)))
    h_tilde = np.tanh(np.concatenate([x, r * h], axis=1) @ Wh + bh)
    h_new = (1 - z) * h + z * h_tilde
    return h_new, (concat, z, r, h_tilde)

def gru_backward(dh_new, cache, Wz, Wr, Wh):
    concat, z, r, h_tilde = cache
    B = dh_new.shape[0]
    dh = dh_new * (1 - z)
    dz = dh_new * (h_tilde - cache[3])  # h_tilde - h_prev
    dh_tilde = dh_new * z
    dr = dh_tilde * cache[3] * (1 - r) * r  # wait, need h_prev
    # Let's recompute carefully using saved values
    x = concat[:, :E]
    h_prev = concat[:, E:]
    # gradients
    dz_raw = dz * z * (1 - z)          # (B, H)
    dr_raw = (dh_tilde * (1 - h_tilde**2) @ Wh[E:, :].T) * r * (1 - r)  # (B, H)
    dh_tilde_raw = dh_tilde * (1 - h_tilde**2)                          # (B, H)
    # dconcat for z and r gates
    dconcat_zr = np.concatenate([dz_raw @ Wz[:E, :].T + dr_raw @ Wr[:E, :].T,
                                 dz_raw @ Wz[E:, :].T + dr_raw @ Wr[E:, :].T], axis=1)
    # dconcat for h_tilde
    concat_h = np.concatenate([x, r * h_prev], axis=1)
    dconcat_h = np.concatenate([dh_tilde_raw @ Wh[:E, :].T,
                                dh_tilde_raw @ Wh[E:, :].T * r], axis=1)
    dconcat = dconcat_zr + dconcat_h
    dx = dconcat[:, :E]
    dh_prev = dconcat[:, E:] + dh * (1 - z)  # from h_new = (1-z)*h_prev + ...
    # param grads
    dWz = concat.T @ dz_raw
    dbz = dz_raw.sum(0)
    dWr = concat.T @ dr_raw
    dbr = dr_raw.sum(0)
    dWh = concat_h.T @ dh_tilde_raw
    dbh = dh_tilde_raw.sum(0)
    return dx, dh_prev, (dWz, dbz, dWr, dbr, dWh, dbh)

def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]   # 16
    C = card["classes"]          # 65
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X.shape[0]

    rng = np.random.default_rng(SEED)
    # embeddings: 256 x E
    emb = (rng.normal(0, 0.01, (256, E))).astype(np.float32)
    # GRU weights: concat_dim = E+H
    concat_dim = E + H
    Wz = (rng.normal(0, math.sqrt(2.0/concat_dim), (concat_dim, H))).astype(np.float32)
    Wr = (rng.normal(0, math.sqrt(2.0/concat_dim), (concat_dim, H))).astype(np.float32)
    Wh = (rng.normal(0, math.sqrt(2.0/concat_dim), (concat_dim, H))).astype(np.float32)
    bz = np.zeros(H, np.float32)
    br = np.zeros(H, np.float32)
    bh = np.zeros(H, np.float32)
    # Output layer
    W_out = (rng.normal(0, math.sqrt(2.0/H), (H, C))).astype(np.float32)
    b_out = np.zeros(C, np.float32)

    params = [emb, Wz, Wr, Wh, bz, br, bh, W_out, b_out]
    # Adam state
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            xb = X[idx]          # (B, d)
            yb = y[idx]          # (B,)
            B = xb.shape[0]

            # Forward
            h = np.zeros((B, H), np.float32)
            caches = []
            for step in range(d):
                x_emb = emb[xb[:, step]]          # (B, E)
                h, cache = gru_forward(x_emb, h, Wz, Wr, Wh, bz, br, bh)
                caches.append((x_emb, h, cache))
            # Output
            logits = h @ W_out + b_out            # (B, C)
            # Softmax + CE
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), yb] -= 1.0
            probs /= B
            # Backward output
            dW_out = h.T @ probs
            db_out = probs.sum(0)
            dh = probs @ W_out.T                  # (B, H)

            # Backward through time
            demb = np.zeros_like(emb)
            dWz = dWr = dWh = np.zeros_like(Wz)
            dbz = dbr = dbh = np.zeros_like(bz)
            for step in reversed(range(d)):
                x_emb, h_new, cache = caches[step]
                x_emb_prev, h_prev, _ = caches[step-1] if step > 0 else (None, np.zeros((B,H),np.float32), None)
                # Need to recompute cache for backward
                concat = np.concatenate([x_emb, h_prev], axis=1)
                z = 1.0 / (1.0 + np.exp(-(concat @ Wz + bz)))
                r = 1.0 / (1.0 + np.exp(-(concat @ Wr + br)))
                h_tilde = np.tanh(np.concatenate([x_emb, r * h_prev], axis=1) @ Wh + bh)
                # gradients
                dz = dh * (h_tilde - h_prev)
                dh_tilde = dh * z
                dh_prev = dh * (1 - z)

                dz_raw = dz * z * (1 - z)
                dr_raw = (dh_tilde * (1 - h_tilde**2) @ Wh[E:, :].T) * r * (1 - r)
                dh_tilde_raw = dh_tilde * (1 - h_tilde**2)

                dconcat_zr = np.concatenate([dz_raw @ Wz[:E, :].T + dr_raw @ Wr[:E, :].T,
                                             dz_raw @ Wz[E:, :].T + dr_raw @ Wr[E:, :].T], axis=1)
                concat_h = np.concatenate([x_emb, r * h_prev], axis=1)
                dconcat_h = np.concatenate([dh_tilde_raw @ Wh[:E, :].T,
                                            dh_tilde_raw @ Wh[E:, :].T * r], axis=1)
                dconcat = dconcat_zr + dconcat_h

                dx_emb = dconcat[:, :E]
                dh_prev += dconcat[:, E:]

                demb[xb[:, step]] += dx_emb
                dWz += concat.T @ dz_raw
                dbz += dz_raw.sum(0)
                dWr += concat.T @ dr_raw
                dbr += dr_raw.sum(0)
                dWh += concat_h.T @ dh_tilde_raw
                dbh += dh_tilde_raw.sum(0)

                dh = dh_prev

            # L2 regularization
            for p, dp in zip(params, [demb, dWz, dWr, dWh, dbz, dbr, dbh, dW_out, db_out]):
                dp += L2 * p

            # Gradient clipping
            grads = [demb, dWz, dWr, dWh, dbz, dbr, dbh, dW_out, db_out]
            norm = math.sqrt(sum(np.sum(g*g) for g in grads))
            if norm > CLIP:
                scale = CLIP / norm
                grads = [g * scale for g in grads]

            # Adam update
            t += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g * g
                m_hat = m[i] / (1 - 0.9**t)
                v_hat = v[i] / (1 - 0.999**t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Export weights
    # magic, d, E, H, C
    out = struct.pack("<5i", 0x47525531, d, E, H, C)
    out += emb.astype("<f4").tobytes()
    out += Wz.astype("<f4").tobytes()
    out += Wr.astype("<f4").tobytes()
    out += Wh.astype("<f4").tobytes()
    out += bz.astype("<f4").tobytes()
    out += br.astype("<f4").tobytes()
    out += bh.astype("<f4").tobytes()
    out += W_out.astype("<f4").tobytes()
    out += b_out.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)

if __name__ == "__main__":
    main()
```
```c
/* label-minigru: ./infer weights.bin in.rec N
 * GRU with 32-dim embeddings, 32 hidden units, 16 steps, 65-class output. */
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
    fwrite(h, 1, 4, f); fwrite(r, 1, k, f);
}

static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static float tanh_f(float x) { return tanhf(x); }

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 20 || hd[0] != 0x47525531) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], E = hd[2], H = hd[3], C = hd[4];
    if (d != 16 || E != 32 || H != 32 || C != 65) { fprintf(stderr, "unexpected dims\n"); return 3; }

    // Expected weight size:
    // emb[256][E] + Wz[E+H][H] + Wr[E+H][H] + Wh[E+H][H] + bz[H] + br[H] + bh[H] + W_out[H][C] + b_out[C]
    long expected = 20
        + 256L * E * 4
        + (E+H) * H * 4 * 3
        + H * 4 * 3
        + H * C * 4
        + C * 4;
    if (nw != expected) { fprintf(stderr, "bad weights size %ld != %ld\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 5);
    const float *emb = p; p += 256 * E;
    const float *Wz = p; p += (E+H) * H;
    const float *Wr = p; p += (E+H) * H;
    const float *Wh = p; p += (E+H) * H;
    const float *bz = p; p += H;
    const float *br = p; p += H;
    const float *bh = p; p += H;
    const float *W_out = p; p += H * C;
    const float *b_out = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;

    // Buffers
    float h[32] = {0};
    float x_emb[32];
    float concat[64];
    float z[32], r[32], h_tilde[32];
    float logits[65];

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }

        // Reset hidden state
        memset(h, 0, H * sizeof(float));

        for (int step = 0; step < d; step++) {
            // Embedding lookup
            const float *e = emb + xi[step] * E;
            memcpy(x_emb, e, E * sizeof(float));

            // Concat [x_emb, h]
            memcpy(concat, x_emb, E * sizeof(float));
            memcpy(concat + E, h, H * sizeof(float));

            // z gate
            for (int j = 0; j < H; j++) {
                float sum = bz[j];
                for (int k = 0; k < E+H; k++) sum += concat[k] * Wz[k * H + j];
                z[j] = sigmoid(sum);
            }
            // r gate
            for (int j = 0; j < H; j++) {
                float sum = br[j];
                for (int k = 0; k < E+H; k++) sum += concat[k] * Wr[k * H + j];
                r[j] = sigmoid(sum);
            }
            // h_tilde: concat [x_emb, r * h]
            float rh[32];
            for (int j = 0; j < H; j++) rh[j] = r[j] * h[j];
            for (int j = 0; j < H; j++) {
                float sum = bh[j];
                for (int k = 0; k < E; k++) sum += x_emb[k] * Wh[k * H + j];
                for (int k = 0; k < H; k++) sum += rh[k] * Wh[(E + k) * H + j];
                h_tilde[j] = tanh_f(sum);
            }
            // New h
            for (int j = 0; j < H; j++) {
                h[j] = (1.0f - z[j]) * h[j] + z[j] * h_tilde[j];
            }
        }

        // Output layer
        memcpy(logits, b_out, C * sizeof(float));
        for (int j = 0; j < H; j++) {
            float hj = h[j];
            if (hj != 0.0f) {
                const float *w = W_out + j * C;
                for (int c = 0; c < C; c++) logits[c] += hj * w[c];
            }
        }

        // Argmax
        unsigned char best = 0;
        float best_score = logits[0];
        for (int c = 1; c < C; c++) {
            if (logits[c] > best_score) {
                best_score = logits[c];
                best = (unsigned char)c;
            }
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```