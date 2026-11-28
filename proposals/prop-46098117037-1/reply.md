```hypothesis
Family: label-local (locally connected 2-layer ReLU network). The 8x8 digit image is split into four 4x4 non-overlapping patches; each patch feeds a small linear layer (8 ReLU units), then a global linear layer combines the 32 features. This captures local stroke patterns (loops, line endings) that global linear models miss, while adding only ~30% more inference ops (832 vs 640 FMA). If dev accuracy ≤ 0.965 (≤0.4% above label-linear) the extra complexity is not justified.
```
```python
import json, math, struct, sys
import numpy as np

SEED = 1
LR = 3e-3
L2 = 1e-4
BATCH = 64
MAX_EPOCHS_TUNE = 30
MAX_EPOCHS_FULL = 50

def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out

def train_model(X, y, patches, P, patch_size, H, C, rng, epochs):
    n = X.shape[0]
    local_W = []
    local_bias = []
    for p in range(P):
        scale = math.sqrt(6.0 / 256)
        W = (rng.uniform(-1, 1, (patch_size, 256, H)) * scale).astype(np.float32)
        b = np.zeros(H, np.float32)
        local_W.append(W)
        local_bias.append(b)
    scale2 = math.sqrt(6.0 / (P * H))
    global_W = (rng.uniform(-1, 1, (P * H, C)) * scale2).astype(np.float32)
    global_bias = np.zeros(C, np.float32)

    all_params = []
    for p in range(P):
        all_params.append(local_W[p])
        all_params.append(local_bias[p])
    all_params.append(global_W)
    all_params.append(global_bias)
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in all_params]

    t = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            x_batch = X[idx]
            y_batch = y[idx]
            B = len(idx)

            local_pres = []
            local_acts = []
            for p in range(P):
                x_patch = x_batch[:, patches[p]]
                emb = local_W[p][np.arange(patch_size)[None, :], x_patch]
                pre = emb.sum(axis=1) + local_bias[p]
                act = np.maximum(pre, 0)
                local_pres.append(pre)
                local_acts.append(act)

            concat_act = np.concatenate(local_acts, axis=1)
            logits = concat_act @ global_W + global_bias

            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs[np.arange(B), y_batch] -= 1.0
            grad_logits = probs / B

            grad_global_W = concat_act.T @ grad_logits + L2 * global_W
            grad_global_bias = grad_logits.sum(axis=0)
            grad_concat_act = grad_logits @ global_W.T

            grad_local_bias = [np.zeros(H, np.float32) for _ in range(P)]
            grad_local_W = [np.zeros((patch_size, 256, H), np.float32) for _ in range(P)]

            for p in range(P):
                grad_act = grad_concat_act[:, p*H:(p+1)*H]
                pre = local_pres[p]
                grad_pre = grad_act * (pre > 0)
                grad_local_bias[p] = grad_pre.sum(axis=0)
                x_patch = x_batch[:, patches[p]]
                for j in range(patch_size):
                    idx_vals = x_patch[:, j]
                    np.add.at(grad_local_W[p][j], idx_vals, grad_pre)
                grad_local_W[p] += L2 * local_W[p]

            t += 1
            grads = []
            for p in range(P):
                grads.append(grad_local_W[p])
                grads.append(grad_local_bias[p])
            grads.append(grad_global_W)
            grads.append(grad_global_bias)

            for i, (param, grad) in enumerate(zip(all_params, grads)):
                m, v = state[i]
                m *= 0.9; m += 0.1 * grad
                v *= 0.999; v += 0.001 * grad * grad
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                param -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)

    return (local_W, local_bias, global_W, global_bias)

def evaluate(X, y, patches, P, patch_size, H, C, params):
    local_W, local_bias, global_W, global_bias = params
    n = X.shape[0]
    correct = 0
    for s in range(0, n, BATCH):
        idx = slice(s, min(s+BATCH, n))
        x_batch = X[idx]
        y_batch = y[idx]
        B = x_batch.shape[0]
        local_acts = []
        for p in range(P):
            x_patch = x_batch[:, patches[p]]
            emb = local_W[p][np.arange(patch_size)[None, :], x_patch]
            pre = emb.sum(axis=1) + local_bias[p]
            act = np.maximum(pre, 0)
            local_acts.append(act)
        concat_act = np.concatenate(local_acts, axis=1)
        logits = concat_act @ global_W + global_bias
        preds = logits.argmax(axis=1)
        correct += (preds == y_batch).sum()
    return correct / n

def export_weights(path, patches, P, patch_size, H, C, params):
    local_W, local_bias, global_W, global_bias = params
    magic = 0x4C4F434C
    header = struct.pack("<6i", magic, 64, P, patch_size, H, C)
    out = bytearray()
    out += header
    for p in range(P):
        out += local_W[p].astype("<f4").tobytes()
    for p in range(P):
        out += local_bias[p].astype("<f4").tobytes()
    out += global_W.astype("<f4").tobytes()
    out += global_bias.astype("<f4").tobytes()
    open(path, "wb").write(out)

def main():
    card = json.load(open(sys.argv[1]))
    X_all = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y_all = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]

    patches = []
    for by in (0, 4):
        for bx in (0, 4):
            idx = []
            for y in range(4):
                for x in range(4):
                    idx.append((by+y)*8 + (bx+x))
            patches.append(idx)
    P = len(patches)
    patch_size = len(patches[0])

    H_local_candidates = [4, 8, 12]
    n = X_all.shape[0]
    tune_size = n // 5
    train_size = n - tune_size
    X_train, y_train = X_all[:train_size], y_all[:train_size]
    X_tune, y_tune = X_all[train_size:], y_all[train_size:]

    best_acc = -1
    best_H = H_local_candidates[0]
    for i, H in enumerate(H_local_candidates):
        rng = np.random.default_rng(SEED + i)
        params = train_model(X_train, y_train, patches, P, patch_size, H, C, rng, MAX_EPOCHS_TUNE)
        acc = evaluate(X_tune, y_tune, patches, P, patch_size, H, C, params)
        if acc > best_acc:
            best_acc = acc
            best_H = H

    rng = np.random.default_rng(SEED + 100)
    final_params = train_model(X_all, y_all, patches, P, patch_size, best_H, C, rng, MAX_EPOCHS_FULL)
    export_weights(sys.argv[4], patches, P, patch_size, best_H, C, final_params)

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
    return r