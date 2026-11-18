<!-- cloudflare @cf/openai/gpt-oss-120b status=200 latency=59.1s -->

**Adversarial review of the README**  

Below each problem is shown the exact line (or table entry) from the README, why it does **not** follow the evidence that you supplied, a concrete fix, and a severity grade (Critical / Major / Minor).  

---

### 1. Over‑stated immutability claim  
**README line** – “Every claim is recorded, so none can be quietly changed later.”  

**What’s wrong** – The ledger is append‑only, but the evidence only shows that entries are written and anchored; it does **not** prove that a claim cannot be altered *before* anchoring or that the anchoring itself is tamper‑proof.  

**Fix** – Change to something verifiable, e.g.:  
> “Every claim is recorded in an append‑only, hash‑chained ledger that is anchored with Sigstore/OpenTimestamps, making post‑anchor modifications detectable.”  

**Grade** – **Major** (misleads readers about a core security guarantee).

---

### 2. Accuracy‑difference wording  
**README line** – “… at 0.7 points lower dev accuracy.”  

**What’s wrong** – Evidence shows lgn_d dev = 0.9728 and mlp16 dev = 0.9800, a **0.0072 absolute difference** (≈ 0.73 %‑points). “0.7 points” suggests 0.7 on
