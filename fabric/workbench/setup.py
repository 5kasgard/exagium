"""Set up the owner's LobeHub workbench on Vercel + Neon (queue P36/P40, owner 2026-11-25). Standard library only.

  python3 -I fabric/workbench/setup.py project            # Vercel project (created if missing); prints nothing secret
  python3 -I fabric/workbench/setup.py neon               # Neon project + DATABASE_URL on Vercel (only if missing)
  python3 -I fabric/workbench/setup.py env NAME FROM_VAR [--build]  # upsert a Vercel env var from an env var (skipped if empty)
  python3 -I fabric/workbench/setup.py gen NAME KIND      # generate once and keep (KIND: secret32 | jwks)
  python3 -I fabric/workbench/setup.py set NAME VALUE     # a plain, non-secret value
  python3 -I fabric/workbench/setup.py domain HOST        # attach a custom domain to the project; prints its status
  python3 -I fabric/workbench/setup.py url                # the address, and whether the build is what serves it
  python3 -I fabric/workbench/setup.py search             # SEARCH_PROVIDERS / CRAWLER_IMPLS from HAS_<NAME>=true flags
  python3 -I fabric/workbench/setup.py logs [SECONDS]      # recent warnings, errors and 4xx/5xx of the live deployment

Kinds: provider keys and generated secrets are "sensitive" (runtime only: `vercel pull` cannot read them, so the
build and its third-party scripts never see them); DATABASE_URL is "encrypted" because the build runs migrations.
Reads VERCEL_API_KEY (and NEON_API_KEY for `neon`). Values are never printed: only names and what was done.
"""
import base64, json, os, re, secrets, socket, subprocess, sys, time, urllib.error, urllib.request

PROJECT = "forge-workbench"
DOMAIN = "workbench.exagium.xyz"   # forge-workbench.vercel.app served another site (owner, 2026-11-25: 404s)
NEON_REGION = "aws-ap-southeast-2"   # Sydney, nearest to the owner
VERCEL = "https://api.vercel.com"
NEON = "https://console.neon.tech/api/v2"


TEAM = None   # Vercel Hobby accounts live in a personal team; API calls must name it (first run: 403 on create)


def call(base, path, token, method="GET", body=None):
    if base == VERCEL and TEAM:
        path += ("&" if "?" in path else "?") + "teamId=" + TEAM
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
        sys.exit(f"vercel project create: HTTP {st} {str(j.get('error', {}).get('message', ''))[:120]}"
                 + (" (team resolved)" if TEAM else " (no team found: check the token's scope covers the account)"))
    return j["id"], True


def env_keys(token, pid):
    st, j = call(VERCEL, f"/v10/projects/{pid}/env", token)
    if st != 200:
        sys.exit(f"vercel env list: HTTP {st}")
    return {e["key"] for e in j.get("envs", []) if "production" in (e.get("target") or [])}


def put_env(token, pid, key, value, kind="sensitive"):
    if kind != "sensitive":   # an existing sensitive entry cannot change kind in place: remove it first
        st, j = call(VERCEL, f"/v10/projects/{pid}/env", token)
        for e in (j.get("envs", []) if st == 200 else []):
            if e.get("key") == key and e.get("type") != kind and "production" in (e.get("target") or []):
                call(VERCEL, f"/v9/projects/{pid}/env/{e['id']}", token, "DELETE")
    st, j = call(VERCEL, f"/v10/projects/{pid}/env?upsert=true", token, "POST",
                 {"key": key, "value": value, "type": kind, "target": ["production"]})
    if st not in (200, 201):
        sys.exit(f"vercel env {key}: HTTP {st} {str(j.get('error', {}).get('message', ''))[:120]}")


REDACT = [(re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+"), "<email>"), (re.compile(r"[\w.-]+\.vercel\.app"), "<host>"),
          (re.compile(r"https?://\S+"), "<url>"), (re.compile(r"\b(\d{1,3}\.){3}\d{1,3}\b"), "<ip>"),
          (re.compile(r"[A-Za-z0-9_-]{32,}"), "<long>")]


def redact(text):
    """Public logs: no emails, account-named hosts, addresses, IPs or long tokens."""
    for rx, sub in REDACT:
        text = rx.sub(sub, text)
    return text


def log_row(r):
    """One line for a log record worth seeing (warning or worse, or an HTTP error), else None."""
    st = r.get("responseStatusCode") or 0
    if r.get("level") not in ("warning", "error", "fatal") and st < 400:
        return None
    path = (r.get("requestPath") or "").split("?")[0]
    when = time.strftime("%H:%M:%S", time.gmtime((r.get("timestampInMs") or 0) / 1000))
    return redact(f"{when} {r.get('level')} {r.get('requestMethod', '')} {path} {st} {(r.get('message') or '')[:400]}")


SEARCH_ORDER = ["tavily", "exa", "brave", "jina", "firecrawl"]   # first listed is tried first
CRAWL_ORDER = ["firecrawl", "exa", "jina", "naive"]              # jina's reader also works without a key


def search_settings(env):
    """Only providers whose key exists, so LobeHub never offers a search that cannot work."""
    has = {n for n in SEARCH_ORDER if env.get(f"HAS_{n.upper()}") == "true"}
    search = ",".join(n for n in SEARCH_ORDER if n in has)
    crawl = ",".join(n for n in CRAWL_ORDER if n in has or n in ("jina", "naive"))
    return search, crawl


def jwks():
    """An RS256 key pair as a JWKS JSON string (LobeHub's JWKS_KEY), made by Node's crypto (on every runner)."""
    js = ("const c=require('crypto');const {privateKey}=c.generateKeyPairSync('rsa',{modulusLength:2048});"
          "const k=privateKey.export({format:'jwk'});k.alg='RS256';k.use='sig';k.kid=c.randomUUID();"
          "process.stdout.write(JSON.stringify({keys:[k]}))")
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout
    keys = json.loads(out)["keys"]
    assert keys[0]["kty"] == "RSA" and keys[0].get("d"), "not a private RSA JWK"
    return out


def neon_org(ntoken):
    """Neon accounts now belong to an organization, and listing or creating projects must name it (2nd run: HTTP 400
    on GET /projects). The account's one organization, or NEON_ORG_ID; its id is never printed."""
    if os.environ.get("NEON_ORG_ID"):
        return os.environ["NEON_ORG_ID"]
    st, j = call(NEON, "/users/me/organizations", ntoken)
    orgs = j.get("organizations", []) if st == 200 else []
    if len(orgs) > 1:
        sys.exit(f"{len(orgs)} Neon organizations: set NEON_ORG_ID to choose one")
    return orgs[0]["id"] if orgs else None


def neon_url(ntoken):
    org = neon_org(ntoken)
    st, j = call(NEON, "/projects" + (f"?org_id={org}" if org else ""), ntoken)
    if st != 200:
        sys.exit(f"neon list: HTTP {st} {str(j.get('message', ''))[:160]}")
    proj = next((p for p in j.get("projects", []) if p.get("name") == PROJECT), None)
    if proj is None:
        body = {"name": PROJECT, "region_id": NEON_REGION, **({"org_id": org} if org else {})}
        st, j = call(NEON, "/projects", ntoken, "POST", {"project": body})
        if st not in (200, 201):
            sys.exit(f"neon create: HTTP {st} {str(j.get('message', ''))[:120]}")
        return j["connection_uris"][0]["connection_uri"], True
    st, j = call(NEON, f"/projects/{proj['id']}/connection_uri?database_name=neondb&role_name=neondb_owner", ntoken)
    if st != 200:
        sys.exit(f"neon connection uri: HTTP {st}")
    return j["uri"], False


def resolve_team(token):
    """The account's one team, if it has exactly one; its id and slug are never printed (they can carry a name)."""
    global TEAM
    st, j = call(VERCEL, "/v2/teams", token)
    teams = j.get("teams", []) if st == 200 else []
    if len(teams) == 1:
        TEAM = teams[0]["id"]
    elif len(teams) > 1:
        sys.exit(f"{len(teams)} Vercel teams: set VERCEL_TEAM_ID to choose one")
    TEAM = os.environ.get("VERCEL_TEAM_ID") or TEAM


def main(a):
    token = vtoken()
    resolve_team(token)
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
    if a[0] == "env":   # `--build`: the build must read it too (3rd run: S3_ENDPOINT empty at build, "Invalid URL")
        v = os.environ.get(a[2], "")
        if not v:
            print(f"{a[1]}: {a[2]} not set, skipped"); return 0
        put_env(token, pid, a[1], v, kind="encrypted" if "--build" in a[3:] else "sensitive")
        print(f"{a[1]}: set from {a[2]}")
        return 0
    if a[0] == "gen":
        if a[1] in have:
            print(f"{a[1]}: already set, kept (changing it would sign everyone out or lose stored keys)"); return 0
        v = base64.b64encode(secrets.token_bytes(32)).decode() if a[2] == "secret32" else jwks()
        put_env(token, pid, a[1], v)
        print(f"{a[1]}: generated and set")
        return 0
    if a[0] == "domain":
        st, j = call(VERCEL, f"/v10/projects/{pid}/domains", token, "POST", {"name": a[1]})
        if st not in (200, 201, 409):
            sys.exit(f"vercel domain {a[1]}: HTTP {st} {str(j.get('error', {}).get('message', ''))[:160]}")
        st, j = call(VERCEL, f"/v9/projects/{pid}/domains/{a[1]}", token)
        print(f"{a[1]}: attached, verified={j.get('verified')}")
        return 0
    if a[0] == "url":   # never prints other aliases: they carry the account's name (public logs)
        st, j = call(VERCEL, f"/v9/projects/{pid}", token)
        prod = (j.get("targets") or {}).get("production") or {}
        al = prod.get("alias") or []
        print(f"latest production deployment: {prod.get('readyState')}")
        print(f"{PROJECT}.vercel.app points at this project: {'yes' if f'{PROJECT}.vercel.app' in al else 'no'}")
        print(f"https://{DOMAIN}" if DOMAIN in al else f"{DOMAIN} is not yet on the latest deployment")
        return 0
    if a[0] == "search":
        search, crawl = search_settings(os.environ)
        if search:
            put_env(token, pid, "SEARCH_PROVIDERS", search, kind="plain")
        put_env(token, pid, "CRAWLER_IMPLS", crawl, kind="plain")
        print(f"SEARCH_PROVIDERS: {search or 'none (no search key yet)'}; CRAWLER_IMPLS: {crawl}")
        return 0
    if a[0] == "logs":   # the stream stays open: read what arrives within the window
        st, j = call(VERCEL, f"/v6/deployments?projectId={pid}&target=production&limit=1", token)
        deps = j.get("deployments") or []
        if not deps:
            sys.exit(f"no production deployment (HTTP {st})")
        path = f"/v1/projects/{pid}/deployments/{deps[0]['uid']}/runtime-logs" + (f"?teamId={TEAM}" if TEAM else "")
        req = urllib.request.Request(VERCEL + path, headers={"Authorization": f"Bearer {token}"})
        rows, end, seen, paths = [], time.time() + float(a[1] if len(a) > 1 else 25), 0, {}
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                while time.time() < end:
                    try:
                        line = r.readline()
                    except (socket.timeout, TimeoutError):
                        continue   # a quiet stream is not the end: the read timeout fired between records (0 records twice)
                    if not line:
                        break
                    try:
                        rec = json.loads(line)
                        row = log_row(rec)
                    except ValueError:
                        continue
                    seen += 1   # every request, by path and status: shows whether the stream sees traffic at all
                    k = f"{(rec.get('requestPath') or '').split('?')[0][:60]} {rec.get('responseStatusCode') or ''}"
                    paths[k] = paths.get(k, 0) + 1
                    if row:
                        rows.append(row)
        except (socket.timeout, TimeoutError, urllib.error.URLError) as e:
            if not rows and not isinstance(e, (socket.timeout, TimeoutError)):
                sys.exit(f"runtime logs: {redact(str(e))[:160]}")
        print(f"deployment state {deps[0].get('state')}; {seen} log records; {len(rows)} warning/error rows")
        for k, n in sorted(paths.items(), key=lambda kv: -kv[1])[:30]:
            print(f"  {n:4d}  {redact(k)}")
        print("\n".join(rows[-80:]))
        return 0
    if a[0] == "set":
        put_env(token, pid, a[1], a[2], kind="plain")
        print(f"{a[1]}: set")
        return 0
    sys.exit("usage: see the docstring")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
