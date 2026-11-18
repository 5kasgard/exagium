"""Queue Q55: do trained weights differ across hosts because OpenBLAS picks its kernels per CPU?

python3 runner/q55.py BUILT TASK OUT_JSON [--cores Prescott Haswell SkylakeX Zen]

Trains label-linear and label-mlp16 on TASK once with OpenBLAS choosing its kernels itself ("auto") and once per
forced OPENBLAS_CORETYPE, each in a fresh process (no jail: this is about the trainer's numerics, not isolation),
and records the weights' sha256 with the host's CPU and versions. The test (fixed before running, 2026-11-18):
  - mechanism: on one host, weights differ between forced core types;
  - cause: with the same forced core type, weights are identical on two hosts whose "auto" weights differ.
Supported if both hold for some core type; rejected if forcing one core type leaves the hosts' weights different.
"""
import argparse, hashlib, json, os, platform, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CANDS = ("label-linear", "label-mlp16")


def train(cand, task_dir, card, core):
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONHASHSEED="0")
    env.pop("OPENBLAS_CORETYPE", None)
    if core != "auto":
        env["OPENBLAS_CORETYPE"] = core
    with tempfile.TemporaryDirectory() as d:
        w = os.path.join(d, "w.bin")
        subprocess.run([sys.executable, "-s", "-P", os.path.join(HERE, "candidates", cand, "train.py"), card,
                        os.path.join(task_dir, "train.in"), os.path.join(task_dir, "train.key"), w],
                       check=True, env=env, capture_output=True)
        probe = subprocess.run([sys.executable, "-c", "import numpy; c = numpy.show_config(mode='dicts'); "
                                "print(c['Build Dependencies']['blas'].get('version'))"], env=env,
                               capture_output=True, text=True)
        return hashlib.sha256(open(w, "rb").read()).hexdigest(), probe.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("built"); ap.add_argument("task"); ap.add_argument("out")
    ap.add_argument("--cores", nargs="+", default=["Prescott", "Haswell", "SkylakeX", "Zen"])
    a = ap.parse_args()
    card = os.path.join(os.path.dirname(HERE), "tasks", a.task, "card.json")
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "")
    import numpy
    res = dict(task=a.task, cpu=cpu, python=platform.python_version(), numpy=numpy.__version__,
               run_id=os.environ.get("GITHUB_RUN_ID", ""), weights={})
    for core in ["auto"] + a.cores:
        for c in CANDS:
            h, blas = train(c, os.path.join(a.built, a.task), card, core)
            res["weights"][f"{c}/{core}"] = h
            res["blas"] = blas
            print(f"{c:13s} {core:9s} {h[:16]}")
    json.dump(res, open(a.out, "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
