"""Remove any provider key value from a file before it leaves the job (blind review 45512119627: a crash or debug
output could print one; GitHub masks secrets in logs, not in artifacts or comments).

python3 -I agent/scrub.py FILE   reads the key values from the environment variables named in agent/models.json"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
names = {p["env"] for p in json.load(open(os.path.join(HERE, "models.json")))["providers"].values() if p.get("env")}
values = [os.environ[n] for n in names if len(os.environ.get(n, "")) >= 8]
path = sys.argv[1]
text = open(path, encoding="utf-8", errors="replace").read()
n = 0
for v in values:
    n += text.count(v)
    text = text.replace(v, "[redacted key]")
open(path, "w", encoding="utf-8").write(text)
print(f"scrub: {n} key value(s) removed")
