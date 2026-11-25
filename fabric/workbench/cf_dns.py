"""Point a hostname under the owner's Cloudflare zone at Vercel: a DNS-only CNAME (Vercel issues the certificate, so
Cloudflare must not proxy it). Upserts one record; prints only the host and what was done. Standard library only.

  python3 -I fabric/workbench/cf_dns.py HOST        # reads CLOUDFLARE_API_TOKEN
"""
import json, os, sys, urllib.error, urllib.request

CF = "https://api.cloudflare.com/client/v4"
TARGET = "cname.vercel-dns.com"


def call(path, token, method="GET", body=None):
    req = urllib.request.Request(CF + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except ValueError:
            return e.code, {}


def upsert(host, token):
    zone_name = ".".join(host.split(".")[-2:])
    st, j = call(f"/zones?name={zone_name}", token)
    zones = j.get("result") or []
    if st != 200 or not zones:
        sys.exit(f"zone {zone_name}: HTTP {st}, {len(zones)} found (the token needs Zone:Read and DNS:Edit)")
    zid = zones[0]["id"]
    st, j = call(f"/zones/{zid}/dns_records?name={host}", token)
    recs = j.get("result") or []
    body = {"type": "CNAME", "name": host, "content": TARGET, "proxied": False, "ttl": 1,
            "comment": "forge workbench on Vercel (fabric/workbench/cf_dns.py)"}
    other = [r for r in recs if r.get("type") != "CNAME"]
    if other:
        sys.exit(f"{host}: has a {other[0].get('type')} record already; not replaced")
    if recs:
        r = recs[0]
        if r.get("content") == TARGET and not r.get("proxied"):
            return f"{host}: CNAME already correct"
        st, j = call(f"/zones/{zid}/dns_records/{r['id']}", token, "PUT", body)
        done = "updated"
    else:
        st, j = call(f"/zones/{zid}/dns_records", token, "POST", body)
        done = "created"
    if st not in (200, 201):
        sys.exit(f"{host}: HTTP {st} {[e.get('message') for e in j.get('errors', [])][:2]}")
    return f"{host}: CNAME {done} (DNS only)"


if __name__ == "__main__":
    print(upsert(sys.argv[1], os.environ.get("CLOUDFLARE_API_TOKEN") or sys.exit("CLOUDFLARE_API_TOKEN is not set")))
