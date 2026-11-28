```hypothesis
Factorization Machine (FM) with learned embeddings per column-value pair, capturing pairwise feature interactions efficiently. The adult dataset has 14 columns with mixed ordinal/categorical types; FMs excel at such tabular data by modeling interactions without exploding parameters. Using the identity sum_{i<j} <v_i,v_j> = (||sum v_i||^2 - sum ||v_i||^2)/2, inference is O(d*k) not O(d^2*k). With k=4, 2 classes, 14 columns, 256 values each, weights ~140KB. Expect dev accuracy >0.855 (beating label-linear 0.8524) at similar or lower CPU cost. Failure: accuracy <= label-linear or inference cost > label-mlp16.
```
```python
"""Factorization Machine for tabular label tasks with fixed-size byte items.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, stdlib+numpy only.
Encodes: ordinal as z-scored single feature, categorical as one-hot over training values.
But we directly learn per-column per-value embeddings and linear weights.
Model: for each class c: score_c = w0_c + sum_j w_c[j, x_j] + sum_{i<j} <v_c[i,x_i], v_c[j,x_j]>
Interaction computed via (||sum v||^2 - sum ||v||^2)/2.
Hyperparameters (k, lr, l2, epochs) chosen on last 20% of train.in (tuning split), then retrain on all.
Exported weights: magic 0x464D3132, d, k, C; float32 w0[C], w[C][d][256], v[C][d][256][k] (row-major)."""
import json, math, struct, sys

import numpy as np

SEED = 1
# Hyperparameter grid for tuning
K_CHOICES = [2, 4, 8]
LR_CHOICES = [1e-3, 3e-3, 1e-2]
L2 = 1e-4
BATCH = 64
SAMPLES = 600000
MAX_EPOCHS = 100


def records(path):
    data, out, i = open(path, "rb").read(), [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i + 4:i + 4 + k])
        i += 4 + k
    return out


def build_encoding(card, X):
    """Return list of (type, mapping) per column.
    For ordinal: (mu, sd) for z-scoring.
    For categorical: list of unique values in training data (for one-hot index)."""
    enc = []
    for j, t in enumerate(card["input"]["types"]):
        col = X[:, j]
        if t == "ordinal":
            mu = col.mean()
            sd = col.std()
            if sd == 0:
                sd = 1.0
            enc.append(("ordinal", (mu, sd)))
        else:
            vals = np.unique(col)
            # map value -> index in vals
            mapping = {v: i for i, v in enumerate(vals)}
            enc.append(("categorical", (vals, mapping)))
    return enc


def prepare_data(card, X, enc):
    """Transform X into:
    - ordinal_cols: indices of ordinal columns
    - cat_cols: list of (col_idx, mapping, num_categories) for categorical
    But for FM we don't one-hot expand; we keep original byte values and use lookup tables of size 256.
    However, for categorical we need to map unseen values to a special index (255).
    We'll create a lookup table for each column that maps byte value -> feature index.
    For ordinal: we z-score and treat as a single continuous feature? But FM expects sparse binary features.
    The label-linear encodes ordinal as a single continuous feature (z-scored) and categorical as one-hot.
    We can do the same: ordinal becomes one feature (continuous), categorical becomes many binary features.
    But then the number of features varies. Simpler: treat all columns as categorical with 256 possible values,
    and for ordinal we use the z-scored value as a continuous feature? Actually, the FM formulation with embeddings
    works for categorical features. For ordinal, we can either discretize (already done by the card into 255 bins)
    or treat as continuous with a single weight and embedding per column? The card says numeric columns as 255 training-quantile bins.
    So they are already discretized into 255 bins (0-254) with 255=unseen. So we can treat all 14 columns as categorical with 256 categories.
    That's what label-linear does: for ordinal it z-scores the bin index? Wait, label-linear does:
        if ordinal: mu, sd = v.mean(), v.std(); enc.append(((np.arange(256) - mu) / sd)[:, None])
    So it maps each bin to a scalar (z-scored) and then uses that as a single feature (not one-hot).
    For categorical: one-hot over training values.
    We'll follow the same: ordinal -> continuous feature (z-scored), categorical -> one-hot over training values.
    But then the FM would have mixed continuous and binary features. We can handle continuous features by having a single embedding vector per column (not per value) and multiply by the continuous value.
    However, to keep it simple and consistent with the card's encoding, we'll use the same encoding as label-linear:
    Build a design matrix Z where each ordinal column contributes 1 feature (z-scored), each categorical column contributes len(unique) features (one-hot).
    Then we can learn a factorization machine on this binary/continuous feature vector.
    But the number of features can be large (sum of categories). The FM would need embeddings per feature.
    Alternatively, we can use the "field-aware" FM where each column is a field, and we have embeddings per field per value.
    Since the card gives column types, we can do field-aware FM: each column is a field, each value in that column has an embedding.
    For ordinal, we have 256 values (bins) but we can still assign an embedding per bin. That's 256 embeddings per ordinal column.
    That's fine (14*256=3584 embeddings per class). We'll do that: treat all columns as categorical with 256 values.
    For ordinal, we don't z-score; we just use the bin index as categorical. The card says numeric columns as 255 training-quantile bins.
    So we have 255 bins + unseen. We'll learn an embedding for each bin.
    This matches the input format (each column is a byte 0-255).
    So we don't need any encoding transformation; we just use the raw byte values as indices into embedding tables.
    The linear term also uses a weight per column per value.
    This is exactly what label-linear does for the linear part (with H=0 it's logistic regression on one-hot).
    But label-linear for ordinal uses z-scoring and a single weight per column? Wait, label-linear's encoding:
        For ordinal: it creates a 256x1 matrix (z-scored) and then multiplies by W1 (which has 1 column per ordinal feature).
        So it's a single weight per ordinal column? Actually, W1 has shape (k, H) where k = sum of encoded dims.
        For ordinal, encoded dim = 1. So it's a single weight per ordinal column (times the z-scored value).
    We'll do the same: for ordinal columns, we have a single linear weight and a single embedding vector per column (not per value).
    But the input is already binned. We can either use the bin index to index into a table of 256 weights/embeddings,
    or we can compute z-scored value and multiply by a single weight. The latter is more efficient and matches label-linear.
    Let's follow label-linear exactly: we'll build the same encoding matrices (enc) as label-linear.
    Then we have a feature matrix Z of shape (n, k) where k = d_ordinal + sum_categorical uniques.
    Then we can learn an FM on this feature matrix Z. However, Z is not binary for ordinal (it's continuous).
    FM with continuous features: the interaction term is sum_{f<g} <v_f, v_g> * z_f * z_g.
    We can still use the efficient computation: let S = sum_f z_f * v_f, then interaction = (||S||^2 - sum_f z_f^2 * ||v_f||^2)/2.
    So we need embeddings per feature (not per column-value). The number of features k can be up to 14*255? Actually categorical uniques <=255 each.
    In worst case, k could be large. But label-linear's k is the sum of unique categories per column + ordinal count.
    For adult, let's check: the card says 14 columns. We don't know exact unique counts. But label-linear works with it.
    We'll proceed with the same encoding as label-linear, then train FM on the encoded features.
    This way we reuse the encoding logic and the feature dimension k is manageable.
    We'll learn per-feature linear weight and per-feature embedding vector (size K).
    Then at inference, we need to apply the same encoding to the input bytes to get the feature vector z (size k).
    But the inference C code would need to replicate the encoding (z-scoring for ordinal, one-hot for categorical).
    That's more complex. The label-linear C code uses precomputed lookup tables T[j][v] which already incorporate the encoding and the linear weights.
    They compute T = E @ W1 where E is the encoding matrix (256 x encoded_dim) for each column.
    Then at inference, they just sum T[j][x_j] over j. That's very fast.
    We can do the same: precompute for each column j and value v:
        - the encoded feature vector e_jv (size k_j) where k_j is 1 for ordinal, num_categories for categorical.
        - the linear weight contribution: e_jv @ w_lin (where w_lin is per-feature linear weight)
        - the embedding contribution: e_jv @ V (where V is per-feature embedding matrix of size (k_j, K))
    But e_jv is a one-hot vector for categorical, or a scalar for ordinal.
    So we can precompute:
        For each column j, value v:
            linear_term[j][v] = e_jv @ w_lin   (scalar)
            embedding_vector[j][v] = e_jv @ V   (vector of size K)
    Then at inference, for a sample we sum linear_term[j][x_j] over j to get linear part,
    and sum embedding_vector[j][x_j] over j to get the summed embedding S, then compute interaction.
    This matches the label-linear's T tables but with an extra embedding vector per column-value.
    The total storage: for each class, we have linear_term (d*256) and embedding_vector (d*256*K).
    That's exactly what we proposed earlier (w and v per column-value per class).
    So we don't need to change the encoding; we can just learn w and v directly per column-value.
    And we can treat all columns uniformly: each column has 256 values, each value has a linear weight and an embedding vector.
    For ordinal columns, the model can learn to approximate the z-scoring if needed.
    This is simpler and more flexible.
    Let's do that: no explicit encoding in training; we directly learn per-column per-value parameters.
    The input X is (n, d) uint8. We'll use X as indices.
    We'll have parameters:
        w0: (C,)
        w: (C, d, 256)
        v: (C, d, 256, K)
    Training: for each sample, we gather w_c[:, j, x_j] and v_c[:, j, x_j, :].
    We'll implement this.
    """

    # We'll just return the raw X (uint8) and the column types for validation.
    return X


def compute_scores_and_grads(X_batch, y_batch, w0, w, v, K, C):
    """X_batch: (B, d) uint8, y_batch: (B,) int64.
    Returns loss, grads for w0, w, v."""
    B, d = X_batch.shape
    # For each class, compute linear sum and summed embedding
    # w0: (C,), w: (C, d, 256), v: (C, d, 256, K)
    # We'll vectorize over batch and classes.
    # Linear term: (B, C)
    lin = w0 + np.sum(w[:, X_batch.T, np.arange(d)], axis=1).T  # (B, C)  # careful with indexing
    # Actually w[:, X_batch.T, np.arange(d)] gives (C, d, B) -> sum over d -> (C, B) -> transpose -> (B, C)
    # Let's do it step by step for clarity and to avoid transpose confusion.
    # We'll loop over classes? C=2, small. Loop over classes for readability.
    scores = np.zeros((B, C), dtype=np.float32)
    S = np.zeros((B, C, K), dtype=np.float32)  # summed embeddings
    norm_sum = np.zeros((B, C), dtype=np.float32)  # sum of squared norms of embeddings
    for c in range(C):
        # linear
        lin_c = w0[c]
        for j in range(d):
            lin_c += w[c, j, X_batch[:, j]]
        scores[:, c] = lin_c
        # embeddings
        S_c = np.zeros((B, K), dtype=np.float32)
        norm_c = np.zeros(B, dtype=np.float32)
        for j in range(d):
            v_vals = v[c, j, X_batch[:, j], :]  # (B, K)
            S_c += v_vals
            norm_c += np.sum(v_vals * v_vals, axis=1)
        S[:, c, :] = S_c
        norm_sum[:, c] = norm_c
    # interaction term
    for c in range(C):
        dot_SS = np.sum(S[:, c, :] * S[:, c, :], axis=1)
        inter = 0.5 * (dot_SS - norm_sum[:, c])
        scores[:, c] += inter
    # softmax
    scores_max = np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # loss
    log_probs = np.log(probs[np.arange(B), y_batch] + 1e-12)
    loss = -np.mean(log_probs)
    # gradients
    # dL/dscores = probs - one_hot
    dscores = probs.copy()
    dscores[np.arange(B), y_batch] -= 1.0
    dscores /= B
    # grad w0
    grad_w0 = np.sum(dscores, axis=0)  # (C,)
    # grad w
    grad_w = np.zeros_like(w)
    # grad v
    grad_v = np.zeros_like(v)
    # For each class
    for c in range(C):
        dc = dscores[:, c]  # (B,)
        # linear grad: for each column j, value x_j, add dc * 1
        for j in range(d):
            idx = X_batch[:, j]
            np.add.at(grad_w[c, j], idx, dc)
        # interaction grad: d(inter)/dS = S (since inter = 0.5*(S·S - sum ||v||^2))
        # d(inter)/dv = S - v (because d(sum ||v||^2)/dv = 2v, so 0.5*(2S - 2v) = S - v)
        # Actually, inter = 0.5 * (sum_k S_k^2 - sum_i sum_k v_{i,k}^2)
        # d inter / d v_{i,k} = 0.5 * (2 S_k * 1 - 2 v_{i,k}) = S_k - v_{i,k}
        # So grad_v for each column j, value x_j, is dc * (S_c - v_val)
        for j in range(d):
            idx = X_batch[:, j]
            v_vals = v[c, j, idx, :]  # (B, K)
            # S_c is (B, K)
            g = dc[:, np.newaxis] * (S[:, c, :] - v_vals)  # (B, K)
            # accumulate to grad_v[c, j, idx, :]
            for k in range(K):
                np.add.at(grad_v[c, j, idx, k], np.arange(B), g[:, k])
    # L2 regularization
    grad_w += L2 * w
    grad_v += L2 * v
    return loss, grad_w0, grad_w, grad_v


def train_fm(X, y, K, lr, epochs, rng):
    n, d = X.shape
    C = 2
    # init
    w0 = np.zeros(C, dtype=np.float32)
    w = rng.uniform(-0.01, 0.01, (C, d, 256)).astype(np.float32)
    v = rng.uniform(-0.01, 0.01, (C, d, 256, K)).astype(np.float32)
    # Adam state
    m_w0, v_w0 = np.zeros_like(w0), np.zeros_like(w0)
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_v, v_v = np.zeros_like(v), np.zeros_like(v)
    t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s+BATCH]
            Xb, yb = X[idx], y[idx]
            loss, gw0, gw, gv = compute_scores_and_grads(Xb, yb, w0, w, v, K, C)
            t += 1
            # Adam update
            for param, grad, m, vv in [(w0, gw0, m_w0, v_w0), (w, gw, m_w, v_w), (v, gv, m_v, v_v)]:
                m *= beta1; m += (1-beta1)*grad
                vv *= beta2; vv += (1-beta2)*grad*grad
                m_hat = m / (1 - beta1**t)
                v_hat = vv / (1 - beta2**t)
                param -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return w0, w, v


def evaluate(X, y, w0, w, v, K, C):
    n = X.shape[0]
    scores = np.zeros((n, C), dtype=np.float32)
    for c in range(C):
        lin = w0[c]
        for j in range(d):
            lin += w[c, j, X[:, j]]
        scores[:, c] = lin
        S = np.zeros((n, K), dtype=np.float32)
        norm = np.zeros(n, dtype=np.float32)
        for j in range(d):
            v_vals = v[c, j, X[:, j], :]
            S += v_vals
            norm += np.sum(v_vals * v_vals, axis=1)
        dot_SS = np.sum(S * S, axis=1)
        inter = 0.5 * (dot_SS - norm)
        scores[:, c] += inter
    preds = np.argmax(scores, axis=1)
    return np.mean(preds == y)


def main():
    card = json.load(open(sys.argv[1]))
    X = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    d = card["input"]["bytes"]
    C = card["classes"]
    n = X.shape[0]
    # Tuning split: last 20%
    split = int(0.8 * n)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    rng = np.random.default_rng(SEED)
    # Hyperparameter search
    best_acc = -1
    best_params = None
    best_model = None
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / split))
    for K in K_CHOICES:
        for lr in LR_CHOICES:
            w0, w, v = train_fm(X_train, y_train, K, lr, epochs, rng)
            acc = evaluate(X_val, y_val, w0, w, v, K, C)
            if acc > best_acc:
                best_acc = acc
                best_params = (K, lr)
                best_model = (w0, w, v)
    print(f"Best K={best_params[0]}, lr={best_params[1]}, val_acc={best_acc:.4f}", file=sys.stderr)
    # Retrain on all data with best params
    K, lr = best_params
    epochs_full = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    w0, w, v = train_fm(X, y, K, lr, epochs_full, rng)
    # Write weights
    # magic 0x464D3132 (FM12)
    out = struct.pack("<4i", 0x464D3132, d, K, C)
    out += w0.astype("<f4").tobytes()
    out += w.astype("<f4").tobytes()
    out += v.astype("<f4").tobytes()
    open(sys.argv[4], "wb").write(out)


if __name__ == "__main__":
    main()
```
```c
/* Factorization Machine inference for tabular label tasks.
 * ./infer weights.bin in.rec N
 * Weights format: magic 0x464D3132, int32 d, K, C; float32 w0[C], w[C][d][256], v[C][d][256][K] (row-major).
 * Input record: d bytes (one per column). Output: one byte class index (argmax). */
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
    if (nw < 16 || hd[0] != 0x464D3132) { fprintf(stderr, "bad weights\n"); return 3; }
    int d = hd[1], K = hd[2], C = hd[3];
    if (C < 1 || C > 256 || K < 1 || K > 64 || d < 1 || d > 256) {
        fprintf(stderr, "bad dimensions\n"); return 3;
    }
    // Expected weight size: 16 + 4*C + 4*C*d*256 + 4*C*d*256*K
    long expected = 16L + 4L*C + 4L*C*d*256 + 4L*C*d*256*K;
    if (nw != expected) { fprintf(stderr, "bad weights size: got %ld expected %ld\n", nw, expected); return 3; }
    const float *w0 = (const float *)(hd + 4);
    const float *w = w0 + C;
    const float *v = w + (long)C * d * 256;
    unsigned char *x = N ? slurp(argv[2], &nx) : NULL;
    float *S = malloc(sizeof(float) * C * K);
    float *scores = malloc(sizeof(float) * C);
    for (long i = 0; i < N; i++) {
        uint32_t k; const unsigned char *xi = next_rec(x, nx, &pos, &k);
        if ((int)k != d) { fprintf(stderr, "item %ld is %u bytes, not %d\n", i, k, d); return 3; }
        for (int c = 0; c < C; c++) {
            scores[c] = w0[c];
            for (int f = 0; f < K; f++) S[c*K + f] = 0.0f;
        }
        // accumulate linear and embeddings
        for (int j = 0; j < d; j++) {
            unsigned char val = xi[j];
            for (int c = 0; c < C; c++) {
                scores[c] += w[((long)c * d + j) * 256 + val];
                const float *v_ptr = v + (((long)c * d + j) * 256 + val) * K;
                for (int f = 0; f < K; f++) S[c*K + f] += v_ptr[f];
            }
        }
        // interaction and argmax
        unsigned char best = 0;
        for (int c = 0; c < C; c++) {
            float dot_SS = 0.0f, norm_sum = 0.0f;
            for (int f = 0; f < K; f++) {
                float sf = S[c*K + f];
                dot_SS += sf * sf;
            }
            // norm_sum = sum_j ||v_j||^2
            for (int j = 0; j < d; j++) {
                unsigned char val = xi[j];
                const float *v_ptr = v + (((long)c * d + j) * 256 + val) * K;
                for (int f = 0; f < K; f++) norm_sum += v_ptr[f] * v_ptr[f];
            }
            scores[c] += 0.5f * (dot_SS - norm_sum);
            if (c == 0 || scores[c] > scores[best]) best = (unsigned char)c;
        }
        put_rec(stdout, &best, 1);
    }
    return 0;
}
```