"""Set up the owner's LobeHub workbench on Vercel + Neon (queue P36/P40, owner 2026-11-25). Standard library only.

  python3 -I fabric/workbench/setup.py project            # Vercel project (created if missing); prints nothing secret
  python3 -I fabric/workbench/setup.py neon               # Neon project + DATABASE_URL on Vercel (only if missing)
  python3 -I fabric/workbench/setup.py env NAME FROM_VAR  # upsert a Vercel env var from an env var (skipped if empty)
  python3 -I fabric/workbench/setup.py gen NAME KIND      # generate once and keep (KIND: secret32 | jwks)
  python3 -I fabric/workbench/setup.py set NAME VALUE     # a plain, non-secret value

Kinds: provider keys and generated secrets are "sensitive" (runtime only: `vercel pull` cannot read them, so the
build and its third-party scripts never see them); DATABASE_URL is "encrypted" because the build runs migrations.
Reads VERCEL_API_KEY (and NEON_API_KEY for `neon`). Values are never printed: only names and what was done.
"""
import base64, json, os, secrets, subprocess, sys, urllib.error, urllib.request

PROJECT = "forge-workbench"
NEON_REGION = "aws-ap-southeast-2"   # Sydney, nearest to the owner
VERCEL = "https://api.vercel.com"
NEON = "https://console.neon.tech/api/v2"


def call(base, path, token, method="GET", body=None):
    req = urllib.request.Request(base + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except ValueError:
            return e.code, {}


def vtoken():
    t = os.environ.get("VERCEL_API_KEY")
    if not t:
        sys.exit("VERCEL_API_KEY is not set")
    return t


def project_id(token):
    st, j = call(VERCEL, f"/v9/projects/{PROJECT}", token)
    if st == 200:
        return j["id"], False
    if st != 404:
        sys.exit(f"vercel project lookup: HTTP {st} {str(j.get('error', {}).get('message', ''))[:120]}")
    st, j = call(VERCEL, "/v11/projects", token, "POST", {"name": PROJECT, "framework": "nextjs"})
    if st not in (200, 201):
        sys.exit(f"vercel project create: HTTP {st} {str(j.get('error', {}).get('message', ''))[:120]}")
    return j["id"], True


def env_keys(token, pid):
    st, j = call(VERCEL, f"/v10/projects/{pid}/env", token)
    if st != 200:
        sys.exit(f"vercel env list: HTTP {st}")
    return {e["key"] for e in j.get("envs", []) if "production" in (e.get("target") or [])}


def put_env(token, pid, key, value, kind="sensitive"):
    st, j = call(VERCEL, f"/v10/projects/{pid}/env?upsert=true", token, "POST",
                 {"key": key, "value": value, "type": kind, "target": ["production"]})
    if st not in (200, 201):
        sys.exit(f"vercel env {key}: HTTP {st} {str(j.get('error', {}).get('message', ''))[:120]}")


def jwks():
    """An RS256 key pair as a JWKS JSON string (LobeHub's JWKS_KEY), made by Node's crypto (on every runner)."""
    js = ("const c=require('crypto');const {privateKey}=c.generateKeyPairSync('rsa',{modulusLength:2048});"
          "const k=privateKey.export({format:'jwk'});k.alg='RS256';k.use='sig';k.kid=c.randomUUID();"
          "process.stdout.write(JSON.stringify({keys:[k]}))")
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout
    keys = json.loads(out)["keys"]
    assert keys[0]["kty"] == "RSA" and keys[0].get("d"), "not a private RSA JWK"
    return out


def neon_url(ntoken):
    st, j = call(NEON, "/projects", ntoken)
    if st != 200:
        sys.exit(f"neon list: HTTP {st}")
    proj = next((p for p in j.get("projects", []) if p.get("name") == PROJECT), None)
    if proj is None:
        st, j = call(NEON, "/projects", ntoken, "POST", {"project": {"name": PROJECT, "region_id": NEON_REGION}})
        if st not in (200, 201):
            sys.exit(f"neon create: HTTP {st} {str(j.get('message', ''))[:120]}")
        return j["connection_uris"][0]["connection_uri"], True
    st, j = call(NEON, f"/projects/{proj['id']}/connection_uri?database_name=neondb&role_name=neondb_owner", ntoken)
    if st != 200:
        sys.exit(f"neon connection uri: HTTP {st}")
    return j["uri"], False


def main(a):
    token = vtoken()
    pid, made = project_id(token)
    if a[0] == "project":
        print(f"vercel project {PROJECT}: {'created' if made else 'exists'}")
        return 0
    have = env_keys(token, pid)
    if a[0] == "neon":
        if "DATABASE_URL" in have:
            print("DATABASE_URL: already set, kept"); return 0
        uri, created = neon_url(os.environ.get("NEON_API_KEY") or sys.exit("NEON_API_KEY is not set"))
        put_env(token, pid, "DATABASE_URL", uri, kind="encrypted")   # the build's migration step needs it (vercel pull)
        print(f"DATABASE_URL: set (neon project {'created' if created else 'existing'})")
        return 0
    if a[0] == "env":
        v = os.environ.get(a[2], "")
        if not v:
            print(f"{a[1]}: {a[2]} not set, skipped"); return 0
        put_env(token, pid, a[1], v)
        print(f"{a[1]}: set from {a[2]}")
        return 0
    if a[0] == "gen":
        if a[1] in have:
            print(f"{a[1]}: already set, kept (changing it would sign everyone out or lose stored keys)"); return 0
        v = base64.b64encode(secrets.token_bytes(32)).decode() if a[2] == "secret32" else jwks()
        put_env(token, pid, a[1], v)
        print(f"{a[1]}: generated and set")
        return 0
    if a[0] == "url":   # only the project-named address: other aliases carry the account's name (public logs)
        st, j = call(VERCEL, f"/v9/projects/{pid}", token)
        al = [x for x in ((j.get("targets") or {}).get("production") or {}).get("alias", []) if x == f"{PROJECT}.vercel.app"]
        print(f"https://{al[0]}" if al else "address: see the Vercel dashboard (the project-named alias is not assigned)")
        return 0
    if a[0] == "set":
        put_env(token, pid, a[1], a[2], kind="plain")
        print(f"{a[1]}: set")
        return 0
    sys.exit("usage: see the docstring")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
