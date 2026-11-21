```hypothesis
label-emb-mlp: Dense embeddings for categorical features + 1-hidden-layer MLP (H=32, E=8). Beats label-linear by learning category similarities and feature interactions; aims to match label-mlp16 accuracy with lower inference cost. Fails if dev accuracy ≤ 0.8524 (label-linear) or instructions/item ≥ 2325 (label-mlp16).
```
```python
"""label-emb-mlp: Embeddings for categorical features + 1-hidden-layer MLP.
python3 -s -P train.py card.json train.in train.key weights.bin
Deterministic (seed 1), single thread, numpy only."""
import json, math, struct, sys
import numpy as np

H = 32          # hidden units
E = 8           # embedding dimension
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
    X_bytes = np.frombuffer(b"".join(records(sys.argv[2])), dtype=np.uint8).reshape(-1, card["input"]["bytes"])
    y = np.array([r[0] for r in records(sys.argv[3])], dtype=np.int64)
    C = card["classes"]
    d = card["input"]["bytes"]
    types = card["input"]["types"]
    
    rng = np.random.default_rng(SEED)
    
    cat_cols = [j for j, t in enumerate(types) if t == "categorical"]
    ord_cols = [j for j, t in enumerate(types) if t == "ordinal"]
    num_cat = len(cat_cols)
    num_ord = len(ord_cols)
    
    # Build vocabularies and mappings for categorical features
    cat_vocabs = []
    cat_mappings = []
    cat_embeds = []
    
    for j in cat_cols:
        vals = np.unique(X_bytes[:, j])
        vals = vals[vals != 255]
        vocab_size = len(vals) + 1
        mapping = np.full(256, vocab_size - 1, dtype=np.uint16)
        mapping[vals] = np.arange(len(vals), dtype=np.uint16)
        mapping[255] = vocab_size - 1
        embed = rng.uniform(-1, 1, (vocab_size, E)).astype(np.float32) * math.sqrt(6.0 / (vocab_size + E))
        cat_vocabs.append(vocab_size)
        cat_mappings.append(mapping)
        cat_embeds.append(embed)
    
    # Ordinal normalization stats
    ord_means = []
    ord_stds = []
    for j in ord_cols:
        v = X_bytes[:, j].astype(np.float64)
        mu, sd = v.mean(), v.std() or 1.0
        ord_means.append(mu)
        ord_stds.append(sd)
    
    input_dim = num_ord + num_cat * E
    n = X_bytes.shape[0]
    
    # Initialize MLP weights
    if H:
        W1 = (rng.uniform(-1, 1, (input_dim, H)) * math.sqrt(6.0 / input_dim)).astype(np.float32)
        b1 = np.zeros(H, np.float32)
        W2 = (rng.uniform(-1, 1, (H, C)) * math.sqrt(6.0 / H)).astype(np.float32)
        b2 = np.zeros(C, np.float32)
        params = [W1, b1, W2, b2]
    else:
        W1 = np.zeros((input_dim, C), np.float32)
        b1 = np.zeros(C, np.float32)
        params = [W1, b1]
    
    all_params = cat_embeds + params
    state = [(np.zeros_like(p), np.zeros_like(p)) for p in all_params]
    t = 0
    
    def softmax_grad(logits, yy):
        logits = logits - logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        P[np.arange(len(yy)), yy] -= 1.0
        return (P / len(yy)).astype(np.float32)
    
    epochs = min(MAX_EPOCHS, math.ceil(SAMPLES / n))
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            yy = y[idx]
            
            # Build batch input
            z = np.zeros((len(idx), input_dim), dtype=np.float32)
            for i, j in enumerate(ord_cols):
                z[:, i] = (X_bytes[idx, j].astype(np.float32) - ord_means[i]) / ord_stds[i]
            for i, j in enumerate(cat_cols):
                mapping = cat_mappings[i]
                embed = cat_embeds[i]
                z[:, num_ord + i*E : num_ord + (i+1)*E] = embed[mapping[X_bytes[idx, j]]]
            
            # Forward
            if H:
                a = z @ W1 + b1
                h = np.maximum(a, 0)
                logits = h @ W2 + b2
                g = softmax_grad(logits, yy)
                gW2 = h.T @ g + L2 * W2
                gb2 = g.sum(0)
                gh = g @ W2.T
                gh *= (a > 0)
                gW1 = z.T @ gh + L2 * W1
                gb1 = gh.sum(0)
                grads = [gW1, gb1, gW2, gb2]
                dz = gh @ W1.T
            else:
                logits = z @ W1 + b1
                g = softmax_grad(logits, yy)
                gW1 = z.T @ g + L2 * W1
                gb1 = g.sum(0)
                grads = [gW1, gb1]
                dz = g @ W1.T
            
            # Embedding gradients
            embed_grads = [np.zeros_like(e) for e in cat_embeds]
            for i in range(num_cat):
                start = num_ord + i * E
                end = start + E
                dz_emb = dz[:, start:end]
                mapping = cat_mappings[i]
                col = cat_cols[i]
                embed_idx = mapping[X_bytes[idx, col]]
                np.add.at(embed_grads[i], embed_idx, dz_emb)
                embed_grads[i] += L2 * cat_embeds[i]
            
            all_grads = embed_grads + grads
            
            t += 1
            for p, gr, (m, v) in zip(all_params, all_grads, state):
                m