```hypothesis
Family: label-gru (tiny GRU with byte embeddings).  
Why: The task is sequential character prediction; a GRU captures order and context better than bag-of-bytes linear models.  
Failure sign: If dev accuracy ≤ 0.3375 (label-linear) or inference cost > 2× label-linear, the family does not improve the Pareto frontier.
```

```python
"""label-gru: tiny GRU with byte embeddings for next-char prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Exports: magic 0x47525531, d, H, C, E; float32 arrays in order:
  emb[256*E], Wz[H*E], Wr[H*E], Wh[H*E], Uz[H*H], Ur[H*H], Uh[H*H],
  bz[H], br[H], bh[H], Wout[C*H], bout[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
H_CANDIDATES = [16, 32, 64]  # hidden/embedding size

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def gru_forward(x, h, params):
    # x: (B, E), h: (B, H), params: dict of (H, E) or (H, H) or (H,)
    Wz, Wr, Wh, Uz, Ur, Uh, bz, br, bh = params
    z = 1.0 / (1.0 + np.exp(-(x @ Wz.T + h @ Uz.T + bz)))
    r = 1.0 / (1.0 + np.exp(-(x @ Wr.T + h @ Ur.T + br)))
    h_tilde = np.tanh(x @ Wh.T + (r * h) @ Uh.T + bh)
    h_new = (1 - z) * h + z * h_tilde
    return h_new

def softmax_cross_entropy(logits, y):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    loss = -np.log(probs[np.arange(len(y)), y] + 1e-12).mean()
    grad = probs
    grad[np.arange(len(y)), y] -= 1.0
    grad /= len(y)
    return loss, grad

def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]  # 16
    n_all = len(X_all)
    split = int(0.8 * n_all)
    X_tr, X_va = X_all[:split], X_all[split:]
    y_tr, y_va = y_all[:split], y_all[split:]

    best_H, best_val_acc = None, -1.0
    rng = np.random.default_rng(SEED)

    for H in H_CANDIDATES:
        E = H
        # init
        scale = math.sqrt(6.0 / (E + H))
        Wz = rng.uniform(-scale, scale, (H, E)).astype(np.float32)
        Wr = rng.uniform(-scale, scale, (H, E)).astype(np.float32)
        Wh = rng.uniform(-scale, scale, (H, E)).astype(np.float32)
        scale_h = math.sqrt(6.0 / (H + H))
        Uz = rng.uniform(-scale_h, scale_h, (H, H)).astype(np.float32)
        Ur = rng.uniform(-scale_h, scale_h, (H, H)).astype(np.float32)
        Uh = rng.uniform(-scale_h, scale_h, (H, H)).astype(np.float32)
        bz = np.zeros(H, np.float32); br = np.zeros(H, np.float32); bh = np.zeros(H, np.float32)
        Wout = rng.uniform(-scale, scale, (C, H)).astype(np.float32)
        bout = np.zeros(C, np.float32)
        emb = rng.uniform(-0.1, 0.1, (256, E)).astype(np.float32)

        params = [Wz, Wr, Wh, Uz, Ur, Uh, bz, br, bh, Wout, bout, emb]
        state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
        t = 0

        n_tr = len(X_tr)
        epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_tr))
        for _ in range(epochs):
            order = rng.permutation(n_tr)
            for s in range(0, n_tr, BATCH):
                idx = order[s:s + BATCH]
                x = X_tr[idx]  # (B, d)
                y = y_tr[idx]
                B = len(x)
                # embed
                x_emb = emb[x]  # (B, d, E)
                h = np.zeros((B, H), np.float32)
                # store for backward
                hs = [h.copy()]
                zs, rs, h_tildes = [], [], []
                for step in range(d):
                    x_step = x_emb[:, step, :]
                    z = 1.0 / (1.0 + np.exp(-(x_step @ Wz.T + h @ Uz.T + bz)))
                    r = 1.0 / (1.0 + np.exp(-(x_step @ Wr.T + h @ Ur.T + br)))
                    h_tilde = np.tanh(x_step @ Wh.T + (r * h) @ Uh.T + bh)
                    h = (1 - z) * h + z * h_tilde
                    hs.append(h.copy())
                    zs.append(z); rs.append(r); h_tildes.append(h_tilde)
                logits = h @ Wout.T + bout
                loss, grad_logits = softmax_cross_entropy(logits, y)
                # backward
                dWout = grad_logits.T @ h
                dbout = grad_logits.sum(0)
                dh = grad_logits @ Wout
                dWz = dWr = dWh = dUz = dUr = dUh = np.zeros_like(Wz)
                dbz = dbr = dbh = np.zeros(H, np.float32)
                demb = np.zeros_like(emb)
                for step in reversed(range(d)):
                    z, r, h_tilde = zs[step], rs[step], h_tildes[step]
                    h_prev = hs[step]
                    x_step = x_emb[:, step, :]
                    dz = dh * (h_tilde - h_prev) * z * (1 - z)
                    dr = dh * z * (1 - np.tanh(x_step @ Wh.T + (r * h_prev) @ Uh.T + bh)**2) * (h_prev @ Uh.T)
                    dh_tilde = dh * z * (1 - h_tilde**2)
                    dWz += dz.T @ x_step
                    dUz += dz.T @ h_prev
                    dbz += dz.sum(0)
                    dWr += dr.T @ x_step
                    dUr += dr.T @ h_prev
                    dbr += dr.sum(0)
                    dWh += dh_tilde.T @ x_step
                    dUh += dh_tilde.T @ (r * h_prev)
                    dbh += dh_tilde.sum(0)
                    dx_step = dz @ Wz + dr @ Wr + dh_tilde @ Wh
                    dh = dh * (1 - z) + dz @ Uz + dr @ Ur + dh_tilde @ (r * Uh)
                    # accumulate embedding grad
                    for b in range(B):
                        demb[x[b, step]] += dx_step[b]
                grads = [dWz, dWr, dWh, dUz, dUr, dUh, dbz, dbr, dbh, dWout, dbout, demb]
                # Adam update
                t += 1
                for i, (p, gr) in enumerate(zip(params, grads)):
                    gr = gr.astype(np.float32)
                    if p.ndim == 2:
                        gr += L2 * p
                    m, v = state[i]
                    m *= 0.9; m += 0.1 * gr
                    v *= 0.999; v += 0.001 * gr * gr
                    m_hat = m / (1 - 0.9 ** t)
                    v_hat = v / (1 - 0.999 ** t)
                    p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

        # validation accuracy
        n_va = len(X_va)
        correct = 0
        for s in range(0, n_va, 256):
            x = X_va[s:s+256]
            y = y_va[s:s+256]
            B = len(x)
            x_emb = emb[x]
            h = np.zeros((B, H), np.float32)
            for step in range(d):
                x_step = x_emb[:, step, :]
                z = 1.0 / (1.0 + np.exp(-(x_step @ Wz.T + h @ Uz.T + bz)))
                r = 1.0 / (1.0 + np.exp(-(x_step @ Wr.T + h @ Ur.T + br)))
                h_tilde = np.tanh(x_step @ Wh.T + (r * h) @ Uh.T + bh)
                h = (1 - z) * h + z * h_tilde
            logits = h @ Wout.T + bout
            pred = logits.argmax(axis=1)
            correct += (pred == y).sum()
        val_acc = correct / n_va
        if val_acc > best_val_acc:
            best_val_acc, best_H = val_acc, H
            best_params = params

    # retrain on all data with best_H
    H = best_H
    E = H
    Wz, Wr, Wh, Uz, Ur, Uh, bz, br, bh, Wout, bout, emb = best_params
    params = [Wz, Wr, Wh, Uz, Ur, Uh, bz, br, bh, Wout, bout, emb]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    n_all = len(X_all)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_all))
    for _ in range(epochs):
        order = rng.permutation(n_all)
        for s in range(0, n_all, BATCH):
            idx = order[s:s + BATCH]
            x = X_all[idx]
            y = y_all[idx]
            B = len(x)
            x_emb = emb[x]
            h = np.zeros((B, H), np.float32)
            hs = [h.copy()]
            zs, rs, h_tildes = [], [], []
            for step in range(d):
                x_step = x_emb[:, step, :]
                z = 1.0 / (1.0 + np.exp(-(x_step @ Wz.T + h @ Uz.T + bz)))
                r = 1.0 / (1.0 + np.exp(-(x_step @ Wr.T + h @ Ur.T + br)))
                h_tilde = np.tanh(x_step @ Wh.T + (r * h) @ Uh.T + bh)
                h = (1 - z) * h + z * h_tilde
                hs.append(h.copy())
                zs.append(z); rs.append(r); h_tildes.append(h_tilde)
            logits = h @ Wout.T + bout
            loss, grad_logits = softmax_cross_entropy(logits, y)
            dWout = grad_logits.T @ h
            dbout = grad_logits.sum(0)
            dh = grad_logits @ Wout
            dWz = dWr = dWh = dUz = dUr = dUh = np.zeros_like(Wz)
            dbz = dbr = dbh = np.zeros(H, np.float32)
            demb = np.zeros_like(emb)
            for step in reversed(range(d)):
                z, r, h_tilde = zs[step], rs[step], h_tildes[step]
                h_prev = hs[step]
                x_step = x_emb[:, step, :]
                dz = dh * (h_tilde - h_prev) * z * (1 - z)
                dr = dh * z * (1 - np.tanh(x_step @ Wh.T + (r * h_prev) @ Uh.T + bh)**2) * (h_prev @ Uh.T)
                dh_tilde = dh * z * (1 - h_tilde**2)
                dWz += dz.T @ x_step
                dUz += dz.T @ h_prev
                dbz += dz.sum(0)
                dWr += dr.T @ x_step
                dUr += dr.T @ h_prev
                dbr += dr.sum(0)
                dWh += dh_tilde.T @ x_step
                dUh += dh_tilde.T @ (r * h_prev)
                dbh += dh_tilde.sum(0)
                dx_step = dz @ Wz + dr @ Wr + dh_tilde @ Wh
                dh = dh * (1 - z) + dz @ Uz + dr @ Ur + dh_tilde @ (r * Uh)
                for b in range(B):
                    demb[x[b, step]] += dx_step[b]
            grads = [dWz, dWr, dWh, dUz, dUr, dUh, dbz, dbr, dbh, dWout, dbout, demb]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                gr = gr.astype(np.float32)
                if p.ndim == 2:
                    gr += L2 * p
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # pack weights
    Wz, Wr, Wh, Uz, Ur, Uh, bz, br, bh, Wout, bout, emb = params
    magic = 0x47525531  # GRU1
    header = struct.pack("<5i", magic, d, H, C, E)
    body = (
        emb.astype("<f4").tobytes() +
        Wz.astype("<f4").tobytes() + Wr.astype("<f4").tobytes() + Wh.astype("<f4").tobytes() +
        Uz.astype("<f4").tobytes() + Ur.astype("<f4").tobytes() + Uh.astype("<f4").tobytes() +
        bz.astype("<f4").tobytes() + br.astype("<f4").tobytes() + bh.astype("<f4").tobytes() +
        Wout.astype("<f4").tobytes() + bout.astype("<f4").tobytes()
    )
    open(sys.argv[4], "wb").write(header + body)

if __name__ == "__main__":
    main()
```

```c
/* label-gru inference: ./infer weights.bin in.rec N
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

static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static float tanh_f(float x) { return tanhf(x); }

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: infer weights.bin in.rec N\n"); return 2; }
    long nw, nx = 0, N = atol(argv[3]), pos = 0;
    unsigned char *wbuf = slurp(argv[1], &nw);
    int32_t *hd = (int32_t *)wbuf;
    if (nw < 20 || hd[0] != 0x47525531) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], H = hd[2], C = hd[3], E = hd[4];
    if (C < 1 || C > 256 || d != 16 || H != E) { fprintf(stderr, "bad weights dims\n"); return 3; }
    size_t expected = 20 + 4L * (256L*E + 3L*H*E + 3L*H*H + 3L*H + C*H + C);
    if (nw != (long)expected) { fprintf(stderr, "bad weights size %ld != %zu\n", nw, expected); return 3; }

    const float *p = (const float *)(hd + 5);
    const float *emb = p; p += 256 * E;
    const float *Wz = p; p += H * E;
    const float *Wr = p; p += H * E;
    const float *Wh = p; p += H * E;
    const float *Uz = p; p += H * H;
    const float *Ur = p; p += H * H;
    const float *Uh = p; p += H * H;
    const float *bz = p; p += H;
    const float *br = p; p += H;
    const float *bh = p; p += H;
    const float *Wout = p; p += C * H;
    const float *bout = p;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *h = calloc(H, sizeof(float));
    float *x_emb = malloc(E * sizeof(float));
    float *z = malloc(H * sizeof(float));
    float *r = malloc(H * sizeof(float));
    float *h_tilde = malloc(H * sizeof(float));
    float *logits = malloc(C * sizeof(float));

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        memset(h, 0, H * sizeof(float));
        for (int step = 0; step < d; step++) {
            const float *e = emb + xi[step] * E;
            memcpy(x_emb, e, E * sizeof(float));
            // z = sigmoid(x_emb @ Wz.T + h @ Uz.T + bz)
            for (int j = 0; j < H; j++) {
                float sum = bz[j];
                for (int k = 0; k < E; k++) sum += x_emb[k] * Wz[j * E + k];
                for (int k = 0; k < H; k++) sum += h[k] * Uz[j * H + k];
                z[j] = sigmoid(sum);
            }
            // r = sigmoid(x_emb @ Wr.T + h @ Ur.T + br)
            for (int j = 0; j < H; j++) {
                float sum = br[j];
                for (int k = 0; k < E; k++) sum += x_emb[k] * Wr[j * E + k];
                for (int k = 0; k < H; k++) sum += h[k] * Ur[j * H + k];
                r[j] = sigmoid(sum);
            }
            // h_tilde = tanh(x_emb @ Wh.T + (r * h) @ Uh.T + bh)
            for (int j = 0; j < H; j++) {
                float sum = bh[j];
                for (int k = 0; k < E; k++) sum += x_emb[k] * Wh[j * E + k];
                for (int k = 0; k < H; k++) sum += r[k] * h[k] * Uh[j * H + k];
                h_tilde[j] = tanh_f(sum);
            }
            // h = (1 - z) * h + z * h_tilde
            for (int j = 0; j < H; j++) {
                h[j] = (1.0f - z[j]) * h[j] + z[j] * h_tilde[j];
            }
        }
        // logits = h @ Wout.T + bout
        for (int c = 0; c < C; c++) {
            float sum = bout[c];
            for (int j = 0; j < H; j++) sum += h[j] * Wout[c * H + j];
            logits[c] = sum;
        }
        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (logits[c] > logits[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```