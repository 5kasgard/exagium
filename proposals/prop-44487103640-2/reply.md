```hypothesis
We propose a factorization machine (FM) of degree 2 with embedding size 4 (family: label-fm). FM captures pairwise feature interactions with factorized parameters, which is known to work well on tabular data like Adult. It should beat linear logistic regression (label-linear) by modeling interactions, while remaining fast at inference (O(d*k) per item). If dev accuracy does not exceed label-linear by at least 0.002, or if inference cost exceeds label-linear by more than 20%, the hypothesis fails.
```

```python
import json, math, struct, sys
import numpy as np

# ------------------------------------------------------------
# Record I/O (same as example)
# ------------------------------------------------------------
def records(path):
    data = open(path, "rb").read()
    out, i = [], 0
    while i < len(data):
        k = struct.unpack_from("<I", data, i)[0]
        out.append(data[i+4:i+4+k])
        i += 4 + k
    return out

# ------------------------------------------------------------
# Preprocessing: build feature matrix from raw bytes
# ------------------------------------------------------------
def build_mappings(X_bytes, types, fit=True, mappings=None):
    """Return (X, mappings) where X is (n, n_features) float32.
    mappings is a dict with keys:
      'ordinal': list of (mean, std, feat_idx) per ordinal feature
      'categorical': list of (map_array, unseen_idx) per cat feature
    """
    n, d = X_bytes.shape
    if fit:
        ordinal_info = []
        cat_info = []
        feat_idx = 0
        for j, t in enumerate(types):
            col = X_bytes[:, j]
            if t == "ordinal":
                mu = col.mean()
                sd = col.std() or 1.0
                ordinal_info.append((mu, sd, feat_idx))
                feat_idx += 1
            else:
                # categories seen in training
                cats = np.unique(col)
                # mapping from byte value to feature index
                map_arr = np.full(256, -1, dtype=np.int32)
                for c in cats:
                    map_arr[c] = feat_idx
                    feat_idx += 1
                # unseen category (value 255)
                unseen_idx = feat_idx
                map_arr[255] = unseen_idx
                feat_idx += 1
                cat_info.append((map_arr, unseen_idx))
        n_features = feat_idx
        mappings = {'ordinal': ordinal_info, 'categorical': cat_info, 'n_features': n_features}
    else:
        ordinal_info = mappings['ordinal']
        cat_info = mappings['categorical']
        n_features = mappings['n_features']

    X = np.zeros((n, n_features), dtype=np.float32)
    for j, t in enumerate(types):
        col = X_bytes[:, j]
        if t == "ordinal":
            mu, sd, fidx = ordinal_info[j]
            X[:, fidx] = (col.astype(np.float32) - mu) / sd
        else:
            map_arr, unseen_idx = cat_info[j]
            fidx = map_arr[col]
            # set one-hot
            X[np.arange(n), fidx] = 1.0
    return X, mappings

# ------------------------------------------------------------
# Factorization Machine with Adam
# ------------------------------------------------------------
class FM:
    def __init__(self, n_features, k, seed=1):
        self.n_features = n_features
        self.k = k
        rng = np.random.default_rng(seed)
        self.w0 = 0.0
        self.w = rng.normal(0, 0.01, n_features).astype(np.float32)
        self.V = rng.normal(0, 0.01, (n_features, k)).astype(np.float32)
        # Adam state
        self.m_w0, self.v_w0 = 0.0, 0.0
        self.m_w = np.zeros_like(self.w)
        self.v_w = np.zeros_like(self.w)
        self.m_V = np.zeros_like(self.V)
        self.v_V = np.zeros_like(self.V)
        self.t = 0

    def forward(self, X):
        """X: (n, n_features) -> scores (n,)"""
        linear = X @ self.w + self.w0
        inter = np.zeros(X.shape[0], dtype=np.float32)
        for f in range(self.k):
            xv = X @ self.V[:, f]
            xv2 = (X**2) @ (self.V[:, f]**2)
            inter += 0.5 * (xv**2 - xv2)
        return linear + inter

    def loss_grad(self, X, y, l2):
        """Compute loss and gradients for a batch."""
        n = X.shape[0]
        scores = self.forward(X)
        p = 1.0 / (1.0 + np.exp(-scores))
        # avoid log(0)
        eps = 1e-12
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        loss += l2 * (np.sum(self.w**2) + np.sum(self.V**2))

        dscore = (p - y) / n  # (n,)
        # gradients
        gw0 = np.sum(dscore)
        gw = X.T @ dscore + 2 * l2 * self.w
        gV = np.zeros_like(self.V)
        for f in range(self.k):
            xv = X @ self.V[:, f]
            # (n, n_features) * (n,1) -> (n, n_features)
            gV[:, f] = X.T @ (dscore * (xv[:, None] - self.V[:, f] * X)) + 2 * l2 * self.V[:, f]
        return loss, gw0, gw, gV

    def adam_step(self, gw0, gw, gV, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        # w0
        self.m_w0 = beta1 * self.m_w0 + (1 - beta1) * gw0
        self.v_w0 = beta2 * self.v_w0 + (1 - beta2) * (gw0 * gw0)
        m_hat = self.m_w0 / (1 - beta1 ** self.t)
        v_hat = self.v_w0 / (1 - beta2 ** self.t)
        self.w0 -= lr * m_hat / (np.sqrt(v_hat) + eps)
        # w
        self.m_w = beta1 * self.m_w + (1 - beta1) * gw
        self.v_w = beta2 * self.v_w + (1 - beta2) * (gw * gw)
        m_hat = self.m_w / (1 - beta1 ** self.t)
        v_hat = self.v_w / (1 - beta2 ** self.t)
        self.w -= lr * m_hat / (np.sqrt(v_hat) + eps)
        # V
        self.m_V = beta1 * self.m_V + (1 - beta1) * gV
        self.v_V = beta2 * self.v_V + (1 - beta2) * (gV * gV)
        m_hat = self.m_V / (1 - beta1 ** self.t)
        v_hat = self.v_V / (1 - beta2 ** self.t)
        self.V -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def predict(self, X):
        scores = self.forward(X)
        return (scores > 0).astype(np.int64)

# ------------------------------------------------------------
# Training loop with hyperparameter search on tuning split
# ------------------------------------------------------------
def train_fm(X_train, y_train, X_tune, y_tune, n_features, k, lr, l2, epochs, batch_size, seed):
    fm = FM(n_features, k, seed)
    n = X_train.shape[0]
    steps_per_epoch = max(1, n // batch_size)
    for epoch in range(epochs):
        order = np.random.default_rng(seed + epoch).permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s+batch_size]
            loss, gw0, gw, gV = fm.loss_grad(X_train[idx], y_train[idx], l2)
            fm.adam_step(gw0, gw, gV, lr)
    # evaluate on tune
    pred = fm.predict(X_tune)
    acc = np.mean(pred == y_tune)
    return fm, acc

def main():
    card = json.load(open(sys.argv[1]))
    types = card["input"]["types"]
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    n = X_bytes.shape[0]
    split = int(n * 0.8)

    # Hyperparameter grid (small for time)
    grid = [
        {"k": 4, "lr": 0.03, "l2": 1e-4, "epochs": 30, "batch": 64},
        {"k": 4, "lr": 0.01, "l2": 1e-4, "epochs": 40, "batch": 64},
        {"k": 8, "lr": 0.03, "l2": 1e-4, "epochs": 30, "batch": 64},
        {"k": 4, "lr": 0.03, "l2": 1e-3, "epochs": 30, "batch": 64},
    ]
    best_acc = -1
    best_params = None
    best_mappings = None

    for params in grid:
        # Preprocess on training portion only
        X_tr, mappings = build_mappings(X_bytes[:split], types, fit=True)
        y_tr = y[:split]
        X_tu, _ = build_mappings(X_bytes[split:], types, fit=False, mappings=mappings)
        y_tu = y[split:]
        fm, acc = train_fm(X_tr, y_tr, X_tu, y_tu, mappings['n_features'], **params, seed=1)
        if acc > best_acc:
            best_acc = acc
            best_params = params
            best_mappings = mappings
        print(f"params {params} -> tune acc {acc:.4f}")

    print(f"Best: {best_params} with acc {best_acc:.4f}")

    # Retrain on full data with best params
    X_full, full_mappings = build_mappings(X_bytes, types, fit=True)
    y_full = y
    # We don't have a separate validation now; just train for the same number of epochs
    fm = FM(full_mappings['n_features'], best_params['k'], seed=1)
    n_full = X_full.shape[0]
    for epoch in range(best_params['epochs']):
        order = np.random.default_rng(1 + epoch).permutation(n_full)
        for s in range(0, n_full, best_params['batch']):
            idx = order[s:s+best_params['batch']]
            loss, gw0, gw, gV = fm.loss_grad(X_full[idx], y_full[idx], best_params['l2'])
            fm.adam_step(gw0, gw, gV, best_params['lr'])

    # --------------------------------------------------------
    # Write weights.bin
    # Format:
    #   magic (uint32) = 0x464D3101
    #   k (int32)
    #   n_features (int32)
    #   d (int32) = 14
    #   w0 (float32)
    #   w (n_features * float32)
    #   V (n_features * k * float32) row-major
    #   for each of d features:
    #       type (uint8) 0=ordinal, 1=categorical
    #       if ordinal: mean (float32), std (float32), feat_idx (int32)
    #       if categorical: mapping[256] (int32 each)
    # --------------------------------------------------------
    d = card["input"]["bytes"]
    magic = 0x464D3101
    k = best_params['k']
    n_features = full_mappings['n_features']
    ordinal_info = full_mappings['ordinal']
    cat_info = full_mappings['categorical']

    # We need to write feature info in the original feature order (0..13).
    # ordinal_info and cat_info are already in that order because we iterated j=0..d-1.
    # But note: ordinal_info has entries only for ordinal features, cat_info only for categorical.
    # We'll create a list of feature descriptors in order.
    feat_descriptors = []
    ord_ptr = 0
    cat_ptr = 0
    for t in types:
        if t == "ordinal":
            feat_descriptors.append(('ordinal', ordinal_info[ord_ptr]))
            ord_ptr += 1
        else:
            feat_descriptors.append(('categorical', cat_info[cat_ptr]))
            cat_ptr += 1

    # Pack
    out = bytearray()
    out += struct.pack("<I", magic)
    out += struct.pack("<iii", k, n_features, d)
    out += struct.pack("<f", fm.w0)
    out += fm.w.astype("<f4").tobytes()
    out += fm.V.astype("<f4").tobytes()  # row-major: (n_features, k) flattened

    for desc in feat_descriptors:
        if desc[0] == 'ordinal':
            _, (mu, sd, fidx) = desc
            out += struct.pack("<Bfff", 0, mu, sd, fidx)  # type, mean, std, feat_idx
        else:
            _, (map_arr, unseen_idx) = desc
            out += struct.pack("<B", 1)  # type
            out += map_arr.astype("<i").tobytes()

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

static const unsigned char *next_rec(const unsigned char *buf, long n, long *pos, uint32_t *