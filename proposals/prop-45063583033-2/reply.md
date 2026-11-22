```hypothesis
Family: label-emblinear (per-position byte embeddings + linear classifier).
Why: Character prediction benefits from positional context; embeddings reduce the 4096 one-hot features to a dense 16*E representation, forcing sharing across positions and acting as a regularized bottleneck. This should improve accuracy over label-linear (33.8%) while keeping inference fast (one embedding lookup per byte + small matrix multiply).
Failure: If dev accuracy ≤ 0.3375 or CPU cost per item exceeds label-linear significantly, the hypothesis is disproven.
```
```python
"""label-emblinear: per-position byte embeddings followed by a linear layer.
train.py: python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Chooses embedding dim E on the last 20% tuning split, then trains on all data.
Weights format: magic 0x4D45424C, int32 d, E, C; float32 emb[d][256][E], W_lin[d*E][C], b_lin[C]."""
import json, math, struct, sys

import numpy as np

SEED = 1
LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 3e-3, 1e-4, 64, 600000, 100
CANDIDATE_E = [8, 16, 32, 64]  # embedding dimensions to try on tuning split


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def train_eval(X_tr, y_tr, X_val, y_val, E, rng, epochs):
    """Train on X_tr,y_tr, return validation accuracy."""
    n_tr, d = X_tr.shape
    C = 65
    # parameters
    W_emb = (rng.uniform(-1, 1, (d, 256, E)) * math.sqrt(6.0 / 256)).astype(np.float32)
    W_lin = (rng.uniform(-1, 1, (d * E, C)) * math.sqrt(6.0 / (d * E))).astype(np.float32)
    b_lin = np.zeros(C, np.float32)
    params = [W_emb, W_lin, b_lin]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward(X_batch):
        # X_batch: (B, d) uint8
        B = X_batch.shape[0]
        # gather embeddings: for each position j, take W_emb[j, X_batch[:, j], :]
        emb = np.empty((B, d, E), np.float32)
        for j in range(d):
            emb[:, j, :] = W_emb[j, X_batch[:, j], :]
        flat = emb.reshape(B, d * E)
        logits = flat @ W_lin + b_lin
        return emb, flat, logits

    def loss_grad(logits, y_batch):
        B = logits.shape[0]
        max_log = logits.max(axis=1, keepdims=True)
        exp_log = np.exp(logits - max_log)
        probs = exp_log / exp_log.sum(axis=1, keepdims=True)
        probs[np.arange(B), y_batch] -= 1.0
        return probs / B  # dlogits

    for _ in range(epochs):
        order = rng.permutation(n_tr)
        for s in range(0, n_tr, BATCH):
            idx = order[s:s + BATCH]
            X_batch, y_batch = X_tr[idx], y_tr[idx]
            emb, flat, logits = forward(X_batch)
            dlogits = loss_grad(logits, y_batch).astype(np.float32)
            # gradients
            dW_lin = flat.T @ dlogits + L2 * W_lin
            db_lin = dlogits.sum(axis=0) + L2 * b_lin
            dflat = dlogits @ W_lin.T  # (B, d*E)
            demb = dflat.reshape(X_batch.shape[0], d, E)
            dW_emb = np.zeros_like(W_emb)
            for j in range(d):
                np.add.at(dW_emb[j], X_batch[:, j], demb[:, j, :])
            dW_emb += L2 * W_emb
            grads = [dW_emb, dW_lin, db_lin]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # validation accuracy
    emb_val, flat_val, logits_val = forward(X_val)
    preds = logits_val.argmax(axis=1)
    return (preds == y_val).mean()


def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    N = len(X_all)
    # split: last 20% for tuning
    n_tune = N // 5
    n_train = N - n_tune
    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_tune, y_tune = X_all[n_train:], y_all[n_train:]

    rng = np.random.default_rng(SEED)
    # determine epochs per candidate (roughly same total samples)
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n_train))

    best_E, best_acc = CANDIDATE_E[0], -1.0
    for E in CANDIDATE_E:
        # fresh rng for each candidate to keep deterministic across runs
        cand_rng = np.random.default_rng(SEED + E)
        acc = train_eval(X_train, y_train, X_tune, y_tune, E, cand_rng, epochs)
        if acc > best_acc:
            best_acc, best_E = acc, E

    # final training on all data with best_E
    final_rng = np.random.default_rng(SEED + best_E + 1000)
    final_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / N))
    # reuse train_eval but on full data (no validation needed, just train)
    # We'll copy the training loop here for final model
    W_emb = (final_rng.uniform(-1, 1, (d, 256, best_E)) * math.sqrt(6.0 / 256)).astype(np.float32)
    W_lin = (final_rng.uniform(-1, 1, (d * best_E, C)) * math.sqrt(6.0 / (d * best_E))).astype(np.float32)
    b_lin = np.zeros(C, np.float32)
    params = [W_emb, W_lin, b_lin]
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0

    def forward_batch(X_batch):
        B = X_batch.shape[0]
        emb = np.empty((B, d, best_E), np.float32)
        for j in range(d):
            emb[:, j, :] = W_emb[j, X_batch[:, j], :]
        flat = emb.reshape(B, d * best_E)
        logits = flat @ W_lin + b_lin
        return emb, flat, logits

    for _ in range(final_epochs):
        order = final_rng.permutation(N)
        for s in range(0, N, BATCH):
            idx = order[s:s + BATCH]
            X_batch, y_batch = X_all[idx], y_all[idx]
            emb, flat, logits = forward_batch(X_batch)
            B = X_batch.shape[0]
            max_log = logits.max(axis=1, keepdims=True)
            exp_log = np.exp(logits - max_log)
            probs = exp_log / exp_log.sum(axis=1, keepdims=True)
            probs[np.arange(B), y_batch] -= 1.0
            dlogits = (probs / B).astype(np.float32)
            dW_lin = flat.T @ dlogits + L2 * W_lin
            db_lin = dlogits.sum(axis=0) + L2 * b_lin
            dflat = dlogits @ W_lin.T
            demb = dflat.reshape(B, d, best_E)
            dW_emb = np.zeros_like(W_emb)
            for j in range(d):
                np.add.at(dW_emb[j], X_batch[:, j], demb[:, j, :])
            dW_emb += L2 * W_emb
            grads = [dW_emb, dW_lin, db_lin]
            t += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    # write weights.bin
    magic = 0x4D45424C  # "MEBL"
    header = struct.pack("<4i", magic, d, best_E, C)
    # emb: d * 256 * E floats, row-major (position, byte, embedding)
    emb_bytes = W_emb.astype("<f4").tobytes()
    # W_lin: (d*E) * C floats
    lin_bytes = W_lin.astype("<f4").tobytes()
    # b_lin: C floats
    bias_bytes = b_lin.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(header + emb_bytes + lin_bytes + bias_bytes)


if __name__ == "__main__":
    main()
```
```c
/* label-emblinear inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output record: 1 byte class index.
 * Weights format: magic 0x4D45424C, int32 d, E, C; float32 emb[d][256][E], W_lin[d*E][C], b_lin[C]. */
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
    if (nw < 16 || hd[0] != 0x4D45424C) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (d != 16 || C != 65 || E <= 0 || E > 256) { fprintf(stderr, "bad weights dims\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * E + (long)d * E * C + C);
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }
    const float *emb = (const float *)(hd + 4);
    const float *W_lin = emb + (long)d * 256 * E;
    const float *b_lin = W_lin + (long)d * E * C;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *flat = malloc(sizeof(float) * d * E);
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        // gather embeddings into flat
        for (int j = 0; j < d; j++) {
            const float *e = emb + ((long)j * 256 + xi[j]) * E;
            memcpy(flat + j * E, e, E * sizeof(float));
        }
        // linear layer: scores = flat @ W_lin + b_lin
        for (int c = 0; c < C; c++) scores[c] = b_lin[c];
        for (int c = 0; c < C; c++) {
            float sum = 0.0f;
            for (int f = 0; f < d * E; f++) sum += flat[f] * W_lin[(long)f * C + c];
            scores[c] += sum;
        }
        // argmax
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    free(flat); free(scores);
    return 0;
}
```