```hypothesis
Factorized linear model (embedding sum + class vectors). Each of the 16 context positions has a 256×E embedding table; each of the 65 classes has an E‑dim vector. Score for class c = bias_c + Σ_{pos} embed[pos, byte] · class_vec[c]. This shares statistical strength across positions and classes, reducing parameters vs. full linear (16·256·65) while capturing similarities. Trained with Adam on cross‑entropy. Expect to beat label‑linear (0.3375) on dev accuracy at similar or lower inference cost (E=16 → ~1300 multiply‑adds/item). Failure if dev accuracy ≤ label‑linear or instructions/item exceed label‑linear by >20%.
```
```python
"""Factorized linear model for next‑character prediction.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib + numpy only.
Uses last 20% of train.in as tuning split for hyperparameter search,
then retrains on all data. Exports weights as:
  int32 magic=0x46414354, d, E, C
  float32 embed[d][256][E], class_vec[C][E], bias[C] (all little‑endian)."""
import json, math, struct, sys

import numpy as np

SEED = 1
RNG = np.random.default_rng(SEED)

# Hyperparameter grid (chosen on tuning split)
E_GRID = [8, 16, 32]
LR_GRID = [0.01, 0.03]
EPOCHS_GRID = [10, 20]
BATCH = 64
L2 = 1e-4
ADAM_BETA1, ADAM_BETA2, ADAM_EPS = 0.9, 0.999, 1e-8


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def softmax_cross_entropy_grad(logits, targets):
    """Return gradient of loss w.r.t. logits (softmax - onehot)."""
    max_logits = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probs[np.arange(len(targets)), targets] -= 1.0
    return probs / len(targets)


def evaluate(X, y, embed, class_vec, bias):
    """Return accuracy on given data."""
    n = X.shape[0]
    correct = 0
    for s in range(0, n, BATCH):
        Xb, yb = X[s:s+BATCH], y[s:s+BATCH]
        # sum embeddings
        h = np.zeros((len(Xb), E), dtype=np.float32)
        for pos in range(d):
            h += embed[pos, Xb[:, pos]]
        logits = h @ class_vec.T + bias
        pred = logits.argmax(axis=1)
        correct += (pred == yb).sum()
    return correct / n


def train_loop(X, y, embed, class_vec, bias, lr, epochs):
    """In‑place Adam update on given data."""
    n = X.shape[0]
    # Adam state
    m_embed = np.zeros_like(embed)
    v_embed = np.zeros_like(embed)
    m_class = np.zeros_like(class_vec)
    v_class = np.zeros_like(class_vec)
    m_bias = np.zeros_like(bias)
    v_bias = np.zeros_like(bias)
    t = 0
    for _ in range(epochs):
        order = RNG.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X[idx], y[idx]
            # forward
            h = np.zeros((len(Xb), E), dtype=np.float32)
            for pos in range(d):
                h += embed[pos, Xb[:, pos]]
            logits = h @ class_vec.T + bias
            # gradient
            g = softmax_cross_entropy_grad(logits, yb).astype(np.float32)
            # grads
            grad_class = g.T @ h + L2 * class_vec
            grad_bias = g.sum(axis=0)
            grad_h = g @ class_vec  # (batch, E)
            grad_embed = np.zeros_like(embed)
            for pos in range(d):
                # accumulate gradient for each byte index
                np.add.at(grad_embed[pos], Xb[:, pos], grad_h)
            grad_embed += L2 * embed
            # Adam update
            t += 1
            for param, grad, m, v in [
                (embed, grad_embed, m_embed, v_embed),
                (class_vec, grad_class, m_class, v_class),
                (bias, grad_bias, m_bias, v_bias),
            ]:
                m *= ADAM_BETA1
                m += (1 - ADAM_BETA1) * grad
                v *= ADAM_BETA2
                v += (1 - ADAM_BETA2) * grad * grad
                m_hat = m / (1 - ADAM_BETA1 ** t)
                v_hat = v / (1 - ADAM_BETA2 ** t)
                param -= lr * m_hat / (np.sqrt(v_hat) + ADAM_EPS)


def main():
    card = json.load(open(sys.argv[1]))
    d = card["input"]["bytes"]  # 16
    C = card["classes"]         # 65
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, d)
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)

    # Split: last 20% for tuning
    n = len(X)
    val_size = n // 5
    train_X, train_y = X[:-val_size], y[:-val_size]
    val_X, val_y = X[-val_size:], y[-val_size:]

    best_acc = -1.0
    best_cfg = None
    best_params = None

    for E in E_GRID:
        for lr in LR_GRID:
            for epochs in EPOCHS_GRID:
                # Initialize
                embed = RNG.normal(0, 0.01, (d, 256, E)).astype(np.float32)
                class_vec = RNG.normal(0, 0.01, (C, E)).astype(np.float32)
                bias = np.zeros(C, dtype=np.float32)
                # Train on train split
                train_loop(train_X, train_y, embed, class_vec, bias, lr, epochs)
                # Evaluate on validation
                acc = evaluate(val_X, val_y, embed, class_vec, bias)
                if acc > best_acc:
                    best_acc = acc
                    best_cfg = (E, lr, epochs)
                    best_params = (embed.copy(), class_vec.copy(), bias.copy())
    print(f"Best config: E={best_cfg[0]}, lr={best_cfg[1]}, epochs={best_cfg[2]}, val_acc={best_acc:.4f}", file=sys.stderr)

    # Retrain on full data with best config
    E, lr, epochs = best_cfg
    embed, class_vec, bias = best_params
    train_loop(X, y, embed, class_vec, bias, lr, epochs)

    # Write weights.bin
    magic = 0x46414354  # 'FACT'
    header = struct.pack("<4i", magic, d, E, C)
    # embed[d][256][E] in row-major (d, 256, E)
    embed_bytes = embed.astype("<f4").tobytes()
    class_bytes = class_vec.astype("<f4").tobytes()
    bias_bytes = bias.astype("<f4").tobytes()
    with open(sys.argv[4], "wb") as f:
        f.write(header + embed_bytes + class_bytes + bias_bytes)


if __name__ == "__main__":
    main()
```
```c
/* Factorized linear inference: ./infer weights.bin in.rec N
 * Input record: d bytes (d=16). Output record: 1 byte (class index 0..C-1).
 * Weights format: int32 magic=0x46414354, d, E, C; float32 embed[d][256][E], class_vec[C][E], bias[C]. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static unsigned char *slurp(const char *path, long *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(3); }
    fseek(f, 0, SEEK_END);
    *len = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc(*len ? *len : 1);
    if (fread(buf, 1, *len, f) != (size_t)*len) {
        fprintf(stderr, "short read %s\n", path);
        exit(3);
    }
    fclose(f);
    return buf;
}

static const unsigned char *next_rec(const unsigned char *buf, long n, long *pos, uint32_t *len) {
    if (*pos + 4 > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    uint32_t k = (uint32_t)buf[*pos] | (uint32_t)buf[*pos+1]<<8 | (uint32_t)buf[*pos+2]<<16 | (uint32_t)buf[*pos+3]<<24;
    if (*pos + 4 + (long)k > n) { fprintf(stderr, "truncated input\n"); exit(3); }
    *len = k;
    const unsigned char *rec = buf + *pos + 4;
    *pos += 4 + k;
    return rec;
}

static void put_rec(FILE *f, const unsigned char *rec, uint32_t k) {
    unsigned char hdr[4] = {k & 255, (k>>8)&255, (k>>16)&255, (k>>24)&255};
    fwrite(hdr, 1, 4, f);
    fwrite(rec, 1, k, f);
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: infer weights.bin in.rec N\n");
        return 2;
    }
    long nw, nx = 0;
    int32_t *hd = (int32_t *)slurp(argv[1], &nw);
    if (nw < 16 || hd[0] != 0x46414354) { fprintf(stderr, "bad weights magic\n"); return 3; }
    int d = hd[1], E = hd[2], C = hd[3];
    if (d != 16 || C != 65 || E <= 0 || E > 256) { fprintf(stderr, "bad dimensions\n"); return 3; }
    long expected = 16L + 4L * ((long)d * 256 * E + (long)C * E + C);
    if (nw != expected) { fprintf(stderr, "bad weights size\n"); return 3; }

    const float *embed = (const float *)(hd + 4);
    const float *class_vec = embed + (long)d * 256 * E;
    const float *bias = class_vec + (long)C * E;

    long N = atol(argv[3]);
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    long pos = 0;

    float *h = malloc(sizeof(float) * E);
    float *scores = malloc(sizeof(float) * C);
    if (!h || !scores) { fprintf(stderr, "oom\n"); exit(3); }

    for (long i = 0; i < N; i++) {
        uint32_t k;
        const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld has %u bytes, expected %d\n", i, k, d); return 3; }

        // h = sum_{pos} embed[pos, xi[pos]]
        memset(h, 0, E * sizeof(float));
        for (int p = 0; p < d; p++) {
            const float *e = embed + ((long)p * 256 + xi[p]) * E;
            for (int j = 0; j < E; j++) h[j] += e[j];
        }

        // scores[c] = bias[c] + h · class_vec[c]
        for (int c = 0; c < C; c++) {
            float sum = bias[c];
            const float *cv = class_vec + (long)c * E;
            for (int j = 0; j < E; j++) sum += h[j] * cv[j];
            scores[c] = sum;
        }

        // argmax (first max wins)
        unsigned char best = 0;
        for (int c = 1; c < C; c++) if (scores[c] > scores[best]) best = (unsigned char)c;
        put_rec(stdout, &best, 1);
    }

    free(h);
    free(scores);
    if (x) free(x);
    free(hd);
    return 0;
}
```