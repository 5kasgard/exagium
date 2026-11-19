"""Make a confirmation copy of a built task: train = 80% of train, dev = the held-out 20%, no test (queue P5b).
python3 -I split.py BUILT TASK OUT_BUILT SEED [TASKS_DIR]
It writes OUT_BUILT/TASK-confirm/ and, under TASKS_DIR (a workspace copy, never committed), TASK-confirm/card.json: the
task's card with the new files' digests, since runner/run.py refuses data that does not match its card (first run, 2026-11-20)."""
import hashlib, json, os, random, shutil, struct, sys


def records(path):
    b, i, out = open(path, "rb").read(), 0, []
    while i < len(b):
        n = struct.unpack("<I", b[i:i + 4])[0]
        out.append(b[i:i + 4 + n]); i += 4 + n
    return out


def main(built, task, out, seed, tasks_dir=None):
    name = task + "-confirm"
    src, dst = os.path.join(built, task), os.path.join(out, name)
    xs, ys = records(os.path.join(src, "train.in")), records(os.path.join(src, "train.key"))
    assert len(xs) == len(ys)
    idx = list(range(len(xs)))
    random.Random(int(seed)).shuffle(idx)
    n_conf = len(idx) // 5
    conf, keep = sorted(idx[:n_conf]), sorted(idx[n_conf:])
    os.makedirs(dst, exist_ok=True)
    for split_name, part in (("train", keep), ("dev", conf)):
        open(os.path.join(dst, split_name + ".in"), "wb").write(b"".join(xs[i] for i in part))
        open(os.path.join(dst, split_name + ".key"), "wb").write(b"".join(ys[i] for i in part))
    json.dump({"task": task, "derived_from": "train", "seed": int(seed), "items": {"train": len(keep), "dev": len(conf)},
               "dev_is": "confirmation split (20% of original train), queue P5b"}, open(os.path.join(dst, "manifest.json"), "w"), indent=1)
    if tasks_dir:
        card = json.load(open(os.path.join(tasks_dir, task, "card.json")))
        card["task"] = name
        card["data"] = {f: hashlib.sha256(open(os.path.join(dst, f), "rb").read()).hexdigest()
                        for f in ("train.in", "train.key", "dev.in", "dev.key")}
        card["items"] = {"train": len(keep), "dev": len(conf)}
        card["description"] = card.get("description", "") + " | P5b confirmation copy: dev is a held-out 20% of train"
        card.pop("test_gate", None)
        os.makedirs(os.path.join(tasks_dir, name), exist_ok=True)
        json.dump(card, open(os.path.join(tasks_dir, name, "card.json"), "w"), indent=1)
    print(f"{name}: train {len(keep)}, confirmation {len(conf)}")


if __name__ == "__main__":
    main(*sys.argv[1:6])
