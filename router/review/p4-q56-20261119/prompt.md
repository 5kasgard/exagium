You are reviewing two security changes in a public repository. Be adversarial and concrete; you see only the diff.

A) Queue P4 and Q22 in router/engines.py: paid model pools (cerebras, xai, gemini) hold the owner's money. A call to one must be refused before any network request unless the workflow lists the pool in ROUTER_PAID_POOLS, the call is within per-call caps, and ROUTER_USAGE_LOG names a log in which today's (UTC) calls to that pool are under the daily cap. Every call (all pools) is logged without prompt or reply text. router.yml seeds the log from the history on the router-runs branch.
B) Queue Q56 in runner/jail.py and runner/run.py: candidate C code (untrusted, later model-written) used to be compiled with gcc on the host, so `#include "/host/secret"` could copy a host file into the binary and its output. Now gcc runs in the jail (bubblewrap namespaces or Landlock, rlimits, no_new_privs) without the seccomp filter (gcc forks; the filter refuses fork). A fail-closed probe in check() compiles a plain program and one that #includes a host file, which must fail.

Questions, each with a verdict (holds / fails / cannot tell) and the diff line:
1. Can any code path reach a paid pool's network request without passing paid_guard (other functions in engines.py, gemini_models, compat_models, review.py, probe.py)? Does a listing call cost money?
2. Can the daily cap be bypassed or miscounted: timestamps, UTC day boundary, malformed log lines, a log that is missing or unwritable, concurrent runs, a failed call that raises before logging?
3. Does the usage log leak anything secret (keys, account ids, prompt text)?
4. For Q56: what can gcc still read or write inside the jail? Can an #include, #embed, __has_include, a linker script, -specs, or `#pragma GCC dependency` reach a host file? Is running gcc without the seccomp filter a real gap, and is the probe strong enough to catch a regression?
5. Anything that could break the existing probe's pass rule or the runner.

Output: numbered findings with severity (critical / major / minor), line, evidence, fix. End with one verdict: ready / ready with fixes / not ready. Do not praise.

DIFF:
diff --git a/.github/workflows/router.yml b/.github/workflows/router.yml
index 49c5d4f..81330c9 100644
--- a/.github/workflows/router.yml
+++ b/.github/workflows/router.yml
@@ -19,9 +19,17 @@ jobs:
   probe:
     runs-on: ubuntu-24.04
     steps:
-      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262  # v4.4.0
+      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262  # v4.4.0  # zizmor: ignore[artipacked] this job pushes with the checkout token and uploads no workspace artifact
+      - name: "today's paid calls so far (queue P4): the daily cap counts router-runs' call log, not just this run"
+        run: |
+          git fetch -q origin router-runs && git show FETCH_HEAD:router/log/calls.jsonl > "$RUNNER_TEMP/usage.jsonl" 2>/dev/null || : > "$RUNNER_TEMP/usage.jsonl"
+          wc -l < "$RUNNER_TEMP/usage.jsonl"
       - name: probe every engine
         env:
+          # queue P4: paid pools this workflow may call, under engines.PAID's per-call and daily caps. Two probe runs at
+          # once can each see the same count, so a day can exceed a cap by one run's calls (at most 4 per pool).
+          ROUTER_PAID_POOLS: cerebras,xai,gemini
+          ROUTER_USAGE_LOG: ${{ runner.temp }}/usage.jsonl
           OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
           CLOUDFLARE_API_TOKEN: ${{ secrets.CLOUDFLARE_API_TOKEN }}
           CLOUDFLARE_ACCOUNT_ID: ${{ secrets.CLOUDFLARE_ACCOUNT_ID }}
diff --git a/router/engines.py b/router/engines.py
index dc42b79..8f476c9 100644
--- a/router/engines.py
+++ b/router/engines.py
@@ -59,6 +59,62 @@ SECRET_ENVS = ("OPENROUTER_API_KEY", "CLOUDFLARE_API_TOKEN", "GEMINI_API_KEY", "
                "CROWLLM_BASE_URL") + tuple(v[1] for v in OPENAI_COMPAT.values())
 
 
+# Queue P4: spend caps at the call site. These pools hold the owner's money (Cerebras $10, xAI $5, Gemini credit), so a
+# call to one is refused before any network traffic unless (1) the workflow lists the pool in ROUTER_PAID_POOLS, (2) the
+# call fits the per-call caps, and (3) ROUTER_USAGE_LOG names a log file and today's (UTC) calls to that pool in it are
+# under the daily cap. Worst case per pool per day is calls_per_day x (prompt_chars/4 + max_tokens) tokens; prices change,
+# so the caps are in calls and tokens and the log keeps each call's reported usage for the real figure.
+PAID = {"cerebras": dict(max_tokens=2000, prompt_chars=40000, calls_per_day=50),
+        "xai": dict(max_tokens=2000, prompt_chars=40000, calls_per_day=30),
+        "gemini": dict(max_tokens=2000, prompt_chars=40000, calls_per_day=50)}
+
+
+class Refused(ValueError):
+    """A paid call the caps don't allow; raised before any request is sent."""
+
+
+def _today_calls(log, engine):
+    day, n = time.strftime("%Y-%m-%d", time.gmtime()), 0
+    try:
+        with open(log) as f:
+            for line in f:
+                try:
+                    r = json.loads(line)
+                except ValueError:
+                    continue
+                n += r.get("engine") == engine and str(r.get("t", "")).startswith(day)
+    except FileNotFoundError:
+        pass
+    return n
+
+
+def paid_guard(engine, prompt_chars, max_tokens):
+    cap = PAID.get(engine)
+    if cap is None:
+        return
+    allowed = {x.strip() for x in os.environ.get("ROUTER_PAID_POOLS", "").split(",") if x.strip()}
+    if engine not in allowed:
+        raise Refused(f"{engine} is a paid pool; this workflow does not list it in ROUTER_PAID_POOLS")
+    if max_tokens > cap["max_tokens"] or prompt_chars > cap["prompt_chars"]:
+        raise Refused(f"{engine}: call over the per-call cap ({max_tokens} > {cap['max_tokens']} tokens or "
+                      f"{prompt_chars} > {cap['prompt_chars']} prompt chars)")
+    log = os.environ.get("ROUTER_USAGE_LOG")
+    if not log:
+        raise Refused(f"{engine}: no ROUTER_USAGE_LOG, so the daily cap can't be enforced")
+    if _today_calls(log, engine) >= cap["calls_per_day"]:
+        raise Refused(f"{engine}: daily cap of {cap['calls_per_day']} calls reached (UTC day)")
+
+
+def log_usage(rec):
+    """Queue Q22: one JSON line per call (no prompt, no reply text) when ROUTER_USAGE_LOG is set."""
+    log = os.environ.get("ROUTER_USAGE_LOG")
+    if log:
+        keep = {k: rec.get(k) for k in ("t", "engine", "model", "ok", "status", "latency_s", "usage")}
+        with open(log, "a") as f:
+            f.write(json.dumps(keep, sort_keys=True) + "\n")
+    return rec
+
+
 def compat_base(name):
     base = OPENAI_COMPAT[name][0]
     return os.environ.get(base[4:], "").rstrip("/") if base.startswith("env:") else base
@@ -87,6 +143,7 @@ def compat_models(name):
 
 
 def compat_call(name, model, prompt, max_tokens=64):
+    paid_guard(name, len(prompt), max_tokens)
     t0 = time.time()
     base, key = compat_base(name), os.environ[OPENAI_COMPAT[name][1]]
     req = urllib.request.Request(base + "/chat/completions", method="POST",
@@ -95,9 +152,9 @@ def compat_call(name, model, prompt, max_tokens=64):
                                  headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
     st, j, h = _send_h(req)
     text = (j.get("choices") or [{}])[0].get("message", {}).get("content") if st == 200 else None
-    return {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)), "engine": name, "model": model,
+    return log_usage({"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)), "engine": name, "model": model,
             "ok": text is not None, "status": st, "latency_s": round(time.time() - t0, 2), "usage": j.get("usage"),
-            "text": text, "limits": h, "error": None if text is not None else _scrub(j.get("_error") or json.dumps(j)[:600])}
+            "text": text, "limits": h, "error": None if text is not None else _scrub(j.get("_error") or json.dumps(j)[:600])})
 
 
 def _scrub(s):
@@ -110,6 +167,7 @@ def _scrub(s):
 
 
 def call(engine, model, prompt, system=None, max_tokens=2000):
+    paid_guard(engine, len(prompt) + len(system or ""), max_tokens)
     t0 = time.time()
     msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
     text, usage = None, None
@@ -142,9 +200,9 @@ def call(engine, model, prompt, system=None, max_tokens=2000):
             text, usage = "".join(p.get("text", "") for p in parts if not p.get("thought")), j.get("usageMetadata")
     else:
         raise ValueError(engine)
-    return {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)), "engine": engine, "model": model,
+    return log_usage({"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)), "engine": engine, "model": model,
             "ok": text is not None, "status": st, "latency_s": round(time.time() - t0, 2), "usage": usage,
-            "text": text, "error": None if text is not None else _scrub(j.get("_error") or json.dumps(j)[:600])}
+            "text": text, "error": None if text is not None else _scrub(j.get("_error") or json.dumps(j)[:600])})
 
 
 def gemini_models():
diff --git a/router/probe.py b/router/probe.py
index c0697e4..5fae5d6 100644
--- a/router/probe.py
+++ b/router/probe.py
@@ -6,7 +6,7 @@ Costs: 1 OpenRouter request (of 50/day), ~2 Cloudflare neurons, 1 Gemini request
 """
 import json, os, sys
 sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python3 -I drops the script dir
-from engines import call, gemini_models, OPENAI_COMPAT, compat_base, compat_models, compat_call
+from engines import call, gemini_models, OPENAI_COMPAT, compat_base, compat_models, compat_call, Refused
 
 PROMPT, WANT = "Reply with exactly this text and nothing else: PROBE-OK 7*6=42", "PROBE-OK 7*6=42"
 
@@ -33,7 +33,10 @@ def main(out):
             else:
                 recs.append({"engine": "gemini", "model": m, "ok": False, "error": "not in this key's model list"})
     for engine, model in plan:
-        r = call(engine, model, PROMPT, max_tokens=1500)
+        try:
+            r = call(engine, model, PROMPT, max_tokens=1500)
+        except Refused as e:   # queue P4: a capped paid pool is recorded, not fatal
+            recs.append({"engine": engine, "model": model, "ok": False, "error": f"refused by spend cap: {e}"}); continue
         r["exact"] = (r["text"] or "").strip() == WANT
         recs.append(r)
     recs += probe_compat()
@@ -120,7 +123,10 @@ def probe_compat():
             if m is None:
                 recs.append({"engine": name, "model": "none", "ok": False, "error": "no chat model chosen; set PROBE_%s_MODEL" % name.upper()})
                 continue
-            r = compat_call(name, m, PROMPT, max_tokens=400)   # reasoning models spend tokens before answering
+            try:
+                r = compat_call(name, m, PROMPT, max_tokens=400)   # reasoning models spend tokens before answering
+            except Refused as e:   # queue P4: a capped paid pool is recorded, not fatal
+                recs.append({"engine": name, "model": m, "ok": False, "error": f"refused by spend cap: {e}"}); continue
             r["exact"] = (r["text"] or "").strip() == WANT
             recs.append(r)
     return recs
diff --git a/runner/jail.py b/runner/jail.py
index b69d33d..6fc52b6 100644
--- a/runner/jail.py
+++ b/runner/jail.py
@@ -14,7 +14,7 @@ Gaps, stated: training writes to a host directory bound at /w, so its total disk
 file-size and open-file limits; and without --size (refused by check()), total disk use in
 bubblewrap's tmpfs is bounded only by the file-size and open-file limits and the host's memory.
 """
-import os, platform, subprocess, sys, tempfile
+import os, platform, shutil, subprocess, sys, tempfile
 
 HERE = os.path.dirname(os.path.abspath(__file__))
 sys.path.insert(0, os.path.join(os.path.dirname(HERE), "loop"))
@@ -70,7 +70,8 @@ def counter_confine(tool_dst):
 
 
 def jail(cmd, binds, timeout, cwd="/w", seccomp=True):
-    """Like loop.jail, with limits.c in front of cmd. seccomp=False only for the trusted timer."""
+    """Like loop.jail, with limits.c in front of cmd. seccomp=False only for trusted programs that fork: the timer,
+    and gcc (compile_candidate), whose input is untrusted but which cannot run under the fork-refusing filter."""
     build()
     binds = list(binds) + [(STATE["dir"], "/run/exagium", False)]
     pre = _limits_args(timeout, seccomp)
@@ -103,6 +104,43 @@ def jail(cmd, binds, timeout, cwd="/w", seccomp=True):
     return subprocess.run(a, capture_output=True, timeout=timeout, preexec_fn=L._fresh_keyring)
 
 
+def compile_candidate(src, out, timeout=300):
+    """Queue Q56: compile a candidate's infer.c inside the jail, never on the host. Only the source is copied in
+    (/w, the one writable place); gcc sees /usr, the Python roots and the jail's own files, so an #include of a host
+    path (a key file, the repo, the runner's home) fails to compile instead of copying the file into the binary and so
+    into the output. gcc forks (cc1, as, ld), so it runs with the rlimits and no_new_privs but without the seccomp
+    filter, as the timer does; the namespaces (bwrap) or Landlock (confine) still bound what it can read and write.
+    Gap, stated: a compiler bug triggered by hostile source would run with that reach, not the filtered one."""
+    build()
+    with tempfile.TemporaryDirectory(prefix="exagium-cc-") as w:
+        shutil.copy2(src, os.path.join(w, "infer.c"))
+        r = jail(["gcc", *L.CFLAGS, "-pipe", "-o", "/w/infer", "/w/infer.c", "-lm"], [(w, "/w", True)], timeout,
+                 seccomp=False)
+        if r.returncode or not os.path.exists(os.path.join(w, "infer")):
+            raise L.Failed(f"{src} does not compile in the jail:\n{r.stderr.decode(errors='replace')[-2000:]}")
+        shutil.copy2(os.path.join(w, "infer"), out)
+
+
+INCLUDE_OK = "#include <stdio.h>\nint main(void){puts(\"ok\");return 0;}\n"
+INCLUDE_HOST = "#include \"%s\"\nint main(void){return 0;}\n"
+
+
+def check_compile():
+    """Fail closed unless the jailed compiler builds a plain program and refuses to #include a host file."""
+    with tempfile.TemporaryDirectory(prefix="exagium-ccprobe-") as d:
+        secret = os.path.join(d, "host-secret.h")
+        open(secret, "w").write("int host_secret = 42;\n")
+        ok, bad = os.path.join(d, "ok.c"), os.path.join(d, "bad.c")
+        open(ok, "w").write(INCLUDE_OK)
+        open(bad, "w").write(INCLUDE_HOST % secret)
+        compile_candidate(ok, os.path.join(d, "ok"))
+        try:
+            compile_candidate(bad, os.path.join(d, "bad"))
+        except L.Failed:
+            return True
+        raise L.Failed("the jailed compiler read a host file through #include (Q56 probe)")
+
+
 PROBE = r"""
 import os, resource, socket, sys
 bad = []
@@ -183,6 +221,7 @@ def check():
             if r.returncode or out != "filtered":
                 raise L.Failed(f"confine inside the jail (the counter path) does not hold here (exit {r.returncode}, "
                                f"{out!r}): {r.stderr.decode(errors='replace')[-500:]}")
+    check_compile()   # Q56: candidate C is compiled in the jail, and a host #include must fail
     f = _bwrap_flags() if L.JAIL["mode"] == "bwrap" else {}
-    return dict(mode=L.JAIL["mode"], limits=LIMITS, seccomp_on_candidates=True, counter_filtered=True,
+    return dict(mode=L.JAIL["mode"], limits=LIMITS, seccomp_on_candidates=True, counter_filtered=True, compile_jailed=True,
                 disable_userns=f.get("disable_userns"), tmpfs_size=f.get("size") and TMPFS_BYTES)
diff --git a/runner/run.py b/runner/run.py
index 8152a49..7ffd3fe 100644
--- a/runner/run.py
+++ b/runner/run.py
@@ -245,7 +245,7 @@ def cmd_evaluate(a):
         c["weights_path"] = os.path.join(work, f"{c['name']}.weights")
         open(c["weights_path"], "wb").write(w)
         c["exe"] = os.path.join(work, f"{c['name']}.infer")
-        L.build(os.path.join(c["dir"], "infer.c"), c["exe"])
+        J.compile_candidate(os.path.join(c["dir"], "infer.c"), c["exe"])   # Q56: never compile candidate code on the host
         out, _ = timed(c, inp, n, timer, cpu)
         items = check(t, a.split, out)
         rec_dir = a.records or os.path.join(work, "records")
