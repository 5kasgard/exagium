"""Make a confirmation copy of a built task: train = 80% of train, dev = the held-out 20%, no test (queue P5b).
python3 -I split.py BUILT TASK OUT_BUILT SEED"""
import json, os, random, shutil, struct, sys


def records(path):
    b, i, out = open(path, "rb").read(), 0, []
    while i < len(b):
        n = struct.unpack("<I", b[i:i + 4])[0]
        out.append(b[i:i + 4 + n]); i += 4 + n
    return out


def main(built, task, out, seed):
    src, dst = os.path.join(built, task), os.path.join(out, task)
    xs, ys = records(os.path.join(src, "train.in")), records(os.path.join(src, "train.key"))
    assert len(xs) == len(ys)
    idx = list(range(len(xs)))
    random.Random(int(seed)).shuffle(idx)
    n_conf = len(idx) // 5
    conf, keep = sorted(idx[:n_conf]), sorted(idx[n_conf:])
    os.makedirs(dst, exist_ok=True)
    for name, part in (("train", keep), ("dev", conf)):
        open(os.path.join(dst, name + ".in"), "wb").write(b"".join(xs[i] for i in part))
        open(os.path.join(dst, name + ".key"), "wb").write(b"".join(ys[i] for i in part))
    json.dump({"task": task, "derived_from": "train", "seed": int(seed), "items": {"train": len(keep), "dev": len(conf)},
               "dev_is": "confirmation split (20% of original train), queue P5b"}, open(os.path.join(dst, "manifest.json"), "w"), indent=1)
    print(f"{task}: train {len(keep)}, confirmation {len(conf)}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
