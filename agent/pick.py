"""Turn the owner's /oc comment into a run plan, by script (no model). P36 issue agent.

  /oc <question>                     default model (models.json "default")
  /oc model=<alias|provider/model> <question>
  /oc models                         list the models opencode can reach (no inference)

python3 -I agent/pick.py MODELS_JSON OUT_DIR   reads COMMENT, ISSUE_TITLE, ISSUE_BODY from the environment
Writes OUT_DIR/prompt.md and prints GitHub output lines: mode, model, paid. Exit 2 with a reason on a refused request."""
import json, os, re, sys

TRIGGER = re.compile(r"^/(oc|opencode)\b[ \t]*", re.I)
MODEL_ARG = re.compile(r"^model=(\S+)[ \t]*")
MODEL_ID = re.compile(r"^[a-z0-9][a-z0-9-]*/[A-Za-z0-9][A-Za-z0-9._:-]{0,120}$")
MAX_PROMPT = 20000


def plan(cfg, comment, title="", body=""):
    m = TRIGGER.match(comment)
    if not m:
        raise ValueError("comment does not start with /oc or /opencode")
    rest = comment[m.end():].strip()
    if rest.split("\n", 1)[0].strip().lower() == "models":
        return {"mode": "models", "model": "", "paid": cfg.get("allow_paid", False), "prompt": ""}
    want = cfg.get("default", "free")
    a = MODEL_ARG.match(rest)
    if a:
        want, rest = a.group(1), rest[a.end():].strip()
    model = cfg.get("aliases", {}).get(want, want)
    if not MODEL_ID.match(model):
        raise ValueError(f"not a model id: {model!r} (use provider/model or an alias in agent/models.json)")
    paid = model not in cfg.get("free_ids", [])
    if paid and not cfg.get("allow_paid", False):
        raise ValueError(f"{model} is paid and agent/models.json has allow_paid false")
    if not rest:
        raise ValueError("no question after /oc")
    prompt = ""
    if title or body:
        prompt += f"# Issue: {title}\n\n{body}\n\n"
    prompt += f"# Request from the repository owner\n\n{rest}\n"
    return {"mode": "ask", "model": model, "paid": paid, "prompt": prompt[:MAX_PROMPT]}


def main(a):
    cfg = json.load(open(a[0]))
    try:
        p = plan(cfg, os.environ.get("COMMENT", ""), os.environ.get("ISSUE_TITLE", ""), os.environ.get("ISSUE_BODY", ""))
    except ValueError as e:
        os.makedirs(a[1], exist_ok=True)
        open(os.path.join(a[1], "answer.md"), "w").write(f"Refused: {e}\n")
        print(f"refused={e}"[:300].replace("\n", " "))
        return 2
    os.makedirs(a[1], exist_ok=True)
    open(os.path.join(a[1], "prompt.md"), "w").write(p["prompt"])
    print(f"mode={p['mode']}\nmodel={p['model']}\npaid={str(p['paid']).lower()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
