"""Take a proposal's files out of a model reply (queue P5). Runs with no secrets; the reply is untrusted text.
python3 -I extract.py REPLY.md CAND_DIR   -> writes CAND_DIR/{train.py, infer.c, README.md}; exit 1 with a reason if
the reply does not hold exactly what is asked: a ```hypothesis block, one ```python block for train.py and one ```c
block for infer.c (the first of each kind is taken), each within its size limit. Nothing in the reply is executed here."""
import os, re, sys

LIMITS = {"hypothesis": 3000, "python": 40000, "c": 40000}
FENCE = re.compile(r"^```[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n(.*?)^```[ \t]*$", re.S | re.M)
KINDS = {"hypothesis": "hypothesis", "text": "hypothesis", "python": "python", "py": "python", "c": "c"}


def blocks(reply):
    out = {}
    for lang, body in FENCE.findall(reply):
        kind = KINDS.get(lang.lower())
        if kind and kind not in out:
            out[kind] = body
    return out


def extract(reply, dst):
    b = blocks(reply)
    missing = [k for k in LIMITS if k not in b]
    if missing:
        return f"missing block(s): {', '.join(missing)}"
    for k, lim in LIMITS.items():
        if len(b[k].encode()) > lim:
            return f"{k} block is {len(b[k].encode())} bytes, over {lim}"
        if "\x00" in b[k]:
            return f"{k} block holds a NUL byte"
    if "def " not in b["python"] or "main" not in b["c"]:
        return "train.py has no function or infer.c has no main"
    os.makedirs(dst, exist_ok=True)
    for f, text in (("train.py", b["python"]), ("infer.c", b["c"]),
                    ("README.md", "# Proposal (model-written, untrusted)\n\n" + b["hypothesis"].strip() + "\n")):
        with open(os.path.join(dst, f), "w") as fh:
            fh.write(text)
    return None


if __name__ == "__main__":
    why = extract(open(sys.argv[1], encoding="utf-8", errors="replace").read(), sys.argv[2])
    if why:
        print(f"not extracted: {why}")
        sys.exit(1)
    print("extracted")
