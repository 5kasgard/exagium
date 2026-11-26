"""Post one message to the owner's Discord as a persona (fabric/discord/personas.json). Standard library only.

  python3 -I post.py agent KEY < text       e.g. agent free: a scheduled reply, as "Scout"
  python3 -I post.py workflow NAME < text   the voice personas.json gives that workflow (else "Forge")

Needs DISCORD_WEBHOOK_URL; without it, does nothing and exits 0. Mentions are disabled (no @everyone from a model's
text). Discord caps a message at 2,000 characters; longer text is sent in parts (at most 4), cut at line ends.
Prints only the HTTP status, never the text (Actions logs are public)."""
import json, os, re, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MAX, PARTS = 1900, 4


def persona(kind, key):
    p = json.load(open(os.path.join(HERE, "personas.json")))
    if kind == "agent":
        return p["agents"].get(key) or {**p["system"]["default"], "name": f"Agent {key}"}
    return p["system"].get(p["workflow_voice"].get(key, ""), p["system"]["default"])


def parts(text):
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text).strip() or "(empty reply)"
    out = []
    while text and len(out) < PARTS:
        cut = text.rfind("\n", 0, MAX) if len(text) > MAX else len(text)
        cut = cut if cut > MAX // 2 else min(MAX, len(text))
        out.append(text[:cut]); text = text[cut:].lstrip("\n")
    if text:
        out[-1] = out[-1][: MAX - 40] + "\n… (cut; the full reply is in LobeHub)"
    return out


def post(kind, key, text):
    hook = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if not hook:
        return []
    who = persona(kind, key)
    codes = []
    for part in parts(text):
        body = json.dumps({"content": part, "username": who["name"], "avatar_url": who["avatar"],
                           "allowed_mentions": {"parse": []}}).encode()
        req = urllib.request.Request(hook, data=body, method="POST",
                                     headers={"content-type": "application/json", "user-agent": "forge-discord"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                codes.append(r.status)
        except urllib.error.HTTPError as e:
            codes.append(e.code)
        except Exception:
            codes.append(-1)
    return codes


if __name__ == "__main__":
    print("discord:", post(sys.argv[1], sys.argv[2], sys.stdin.read()) or "DISCORD_WEBHOOK_URL not set")
