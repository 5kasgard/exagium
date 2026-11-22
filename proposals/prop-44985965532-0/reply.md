```hypothesis
Family: label-fm (Factorization Machine with rank-4 pairwise interactions).
Why: Adult income depends strongly on feature interactions (education×occupation, marital-status×relationship) which linear models miss. FM adds O(d·R) parameters (d=14, R=4) and ~200 ops/item, capturing pairwise interactions via low-rank embeddings while staying near linear speed.
Failure signal: Dev accuracy ≤ label-linear (0.8524) or Ir/item > 2× label-linear (>1500) showing interactions not worth the cost.
```
```python
import json, math, struct, sys

import numpy as np

R = 4
H = 0
SEED, LR, L2, BATCH, SAMPLES, MAX_EPOCHS = 1, 3e-3, 1e-4, 64, 600000, 100


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
    rng = np.random.default_rng(SEED)

    enc = []
    for j, t in enumerate(types):
        v = X[:, j].astype(np.float64)
        if t == "ordinal":
            mu, sd = v.mean(), v.std() or 1.0
            enc.append(((np.arange(256) - mu) / sd)[:, None])
        else:
            vals = np.unique(X[:, j])
            E = np.zeros((256, len(vals)))
            E[vals, np.arange(len(vals))] = 1.0
            enc.append(E)

    k = sum(E.shape[1] for E in enc)
    W = H if H else C

    W1 = (rng.uniform(-1, 1, (k, W)) * math.sqrt(6.0 / k)).astype(np.float32)
    b1 = np.zeros(W, np.float32)

    V = [rng.normal(0, 0.01, (256, R)).astype(np.float32) for _ in range(d)]
    w_fm = np.zeros(C, np.float32)

    if H:
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [W1, b1] + V + [w_fm, W2, b2]
    else:
        params = [W1, b1] + V + [w_fm]

    state = [(np.zeros_like(p), np.zeros_like(p)) for p in params]

    Z = np.concatenate([E[X[:, j]] for j, E in enumerate(enc)], axis=1).astype(np.float32)
    n = Z.shape[0]

    def softmax_grad(logits, yy):
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        probs[np.arange(len(yy)), yy] -= 1.0
        return (probs / len(yy)).astype(np.float32)

    t = 0
    max_epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(max_epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            z = Z[idx]
            yy = y[idx]
            B = len(yy)

            linear_out = z @ W1 + b1

            sum_r = np.zeros((B, R), np.float32)
            sum_sq_r = np.zeros((B, R), np.float32)
            for j in range(d):
                v_j = V[j][X[idx, j]]
                sum_r += v_j
                sum_sq_r += v_j * v_j
            interaction = 0.5 * np.sum(sum_r * sum_r - sum_sq_r, axis=1)

            if H:
                hidden = np.maximum(linear_out, 0)
                logits = hidden @ W2 + b2
            else:
                logits = linear_out
            logits += interaction[:, None] * w_fm[None, :]

            g = softmax_grad(logits, yy)

            if H:
                grad_W2 = hidden.T @ g + L2 * W2
                grad_b2 = g.sum(0) + L2 * b2
                gh = g @ W2.T
                gh *= (linear_out > 0)
                grad_W1 = z.T @ gh + L2 * W1
                grad_b1 = gh.sum(0) + L2 * b1
                grads = [grad_W1, grad_b1]
            else:
                grad_W1 = z.T @ g + L2 * W1
                grad_b1 = g.sum(0) + L2 * b1
                grads = [grad_W1, grad_b1]

            gw = g @ w_fm
            grad_V = []
            for j in range(d):
                v_j = V[j][X[idx, j]]
                grad_vj = gw[:, None] * (sum_r - v_j)
                grad_vj += L2 * v_j
                grad_accum = np.zeros((256, R), np.float32)
                vals = X[idx, j]
                np.add.at(grad_accum, vals, grad_vj)
                grad_V.append(grad_accum)

            grad_w_fm = np.sum(g * interaction[:, None], axis=0) + L2 * w_fm

            if H:
                all_grads = grads + grad_V + [grad_w_fm, grad_W2, grad_b2]
            else:
                all_grads = grads + grad_V + [grad_w_fm]

            t += 1
            for i, (p, gr) in enumerate(zip(params, all_grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * gr
                v *= 0.999; v += 0.001 * gr * gr
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    T = []
    off = 0
    for E in enc:
        T.append((E.astype(np.float32) @ W1[off:off + E.shape[1]]).astype(np.float32))
        off += E.shape[1]

    out = struct.pack("<5i", 0x464D3130, d, R, C, H)
    for t_arr in T:
        out += t_arr.astype("<f4").tobytes()
    out += b1.astype("<f4").tobytes()
    for v_arr in V:
        out += v_arr.astype("<f4").tobytes()
    out += w_fm.astype("<f4").tobytes()
    if H:
        out += W2.astype("<f4").tobytes()
        out += b2.astype("<f4").tobytes()

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
    int d = hd[1], R = hd[2], C = hd[3], H = hd[4];
    int W = H ? H : C;
    long expected = 20 + 4L * ((long)d * 256 * W + W + (long)d * 256 * R + C + (H ? (long)H * C + C : 0));
    if (C < 1 || C > 256 || nw != expected) {
        fprintf(stderr, "bad weights size: got %ld, expected %ld\n", nw, expected); return 3;
    }
    const float *ptr = (const float *)(hd + 5);
    const float *T = ptr; ptr += (long)d * 256 * W;
    const float *b1 = ptr; ptr += W;
    const float *V = ptr; ptr += (long)d * 256 * R;
    const float *w_fm = ptr; ptr += C;
    const float *W2 = H ? ptr : NULL; if (H) ptr += (long)H * C;
    const float *b2 = H ? ptr : NULL;

    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *acc = malloc(sizeof(float) * W);
    float *sum_r = malloc(sizeof(float) * R);
    float *sum_sq_r = malloc(sizeof(float) * R);
    float *sc = malloc(sizeof(float) * C);

    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int w = 0; w < W; w++) acc[w] = b1[w];
        for (int j = 0; j < d; j++) {
            const float *t = T + ((long)j * 256 + xi[j]) * W;
            for (int w = 0; w < W; w++) acc[w] += t[w];
        }
        for (int r = 0; r < R; r++) {
            sum_r[r] = 0.0f;
            sum_sq_r[r] = 0.0f;
        }
        for (int j = 0; j < d; j++) {
            const float *v = V + ((long)j * 256 + xi[j]) * R;
            for (int r = 0; r < R; r++) {
                float val = v[r];
                sum_r[r] += val;
                sum_sq_r[r] += val * val;
            }
        }
        float interaction = 0.0f;
        for (int r = 0; r < R; r++) {
            interaction += sum_r[r] * sum_r[r] - sum_sq_r[r];
        }
        interaction *= 0.5f;

        if (H) {
            for (int c = 0; c < C; c++) sc[c] = b2[c];
            for (int u = 0; u < H; u++) {
                float h = acc[u] > 0.0f ? acc[u] : 0.0f;
                for (int c = 0; c < C; c++) sc[c] += h * W2[(long)u * C + c];
            }
        } else {
            for (int c = 0; c < C; c++) sc[c] = acc[c];
        }
        for (int c = 0; c < C; c++) sc[c] += interaction * w_fm[c];

        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (sc[c] > sc[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```