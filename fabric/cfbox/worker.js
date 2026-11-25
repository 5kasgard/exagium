// forge-cfbox: one Durable Object with a Cloudflare Container, driven by fabric/sandbox/cf_eval.py to measure the
// sandbox criteria of audit/fabric/sandbox-decision-2026-11-25.md (#191), the same protocol E2B gets.
//
// Two ways a session could carry over between uses, both measured:
//   snap    after setup, snapshot the whole container disk and stop it; each check starts from the last snapshot
//           (the E2B pause/resume analogue; running processes cannot survive, files and installed tools should)
//   native  leave the container alone; it stops after IDLE_MS without requests; each check first records whether it
//           is still running, then whether files are still there
//
// POST /create?mode=snap|native, /check?mode=..., /kill?mode=...  with  Authorization: Bearer <BOX_TOKEN>.
// No GitHub token, no other secrets: a container here can run code but cannot touch the repo.
// Written by Claude Opus 5.5 for the Forge (2026-11-26). Kind of output: measured (the JSON each call returns).

const IDLE_MS = 10 * 60 * 1000;            // Cloudflare's own default sleepAfter is 10 min; E2B's test also used 10 min
const INSTANCE = "lite";                    // the decision doc: create the cheapest size
const MODES = new Set(["snap", "native"]);
const LANDLOCK = "import ctypes;l=ctypes.CDLL(None,use_errno=True);v=l.syscall(444,None,0,1);print(v if v>0 else 'none errno %d'%ctypes.get_errno())";
const BENCH = "import time;t=time.time();s=sum(i*i for i in range(3_000_000));print(round(time.time()-t,3))";
const TICK = "while true; do date +%s >> /workspace/ticks.log; sleep 10; done";

export function authorized(request, env) {
  const want = env.BOX_TOKEN || "";
  const got = (request.headers.get("authorization") || "").replace(/^Bearer\s+/i, "");
  if (want.length < 32 || got.length !== want.length) return false;
  let diff = 0;                                     // constant-time compare
  for (let i = 0; i < want.length; i++) diff |= want.charCodeAt(i) ^ got.charCodeAt(i);
  return diff === 0;
}

const dec = new TextDecoder();
const tail = (b, n) => dec.decode(b).trim().slice(-n);
const secs = (t) => Math.round((Date.now() - t) / 10) / 100;

export class EvalBox {
  constructor(ctx, env) { this.ctx = ctx; this.env = env; }

  get box() {
    if (!this.ctx.container) throw new Error("container binding missing");
    return this.ctx.container;
  }

  async sh(script, timeoutMs = 180000) {
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), timeoutMs);
    try {
      const p = await this.box.exec(["/bin/sh", "-c", script], { cwd: "/workspace", signal: ac.signal });
      const o = await p.output();
      return { rc: o.exitCode, out: tail(o.stdout, 400), err: tail(o.stderr, 200) };
    } catch (e) {
      return { rc: -1, out: "", err: String(e && e.message || e).slice(0, 300) };
    } finally { clearTimeout(timer); }
  }

  async startFresh(snapshotId) {
    const opts = { instance: INSTANCE, enableInternet: true, labels: { forge: "sandbox-eval", issue: "191" } };
    if (snapshotId) this.box.start({ ...opts, containerSnapshot: { id: snapshotId } });
    else this.box.start({ ...opts, image: this.box.images.sandbox });
    await this.box.setInactivityTimeout(IDLE_MS);
    // the first exec waits for the instance; time it as part of start
    return this.sh("echo up");
  }

  async startTicker() {
    const p = await this.box.exec(["/bin/sh", "-c", TICK], { cwd: "/workspace", stdout: "ignore", stderr: "ignore" });
    return p.pid;
  }

  // marker file, one installed package, a process that writes a timestamp every 10 s; throws if any of it failed
  async seed() {
    const setup = await this.sh("echo forge-$(date +%s) > /workspace/marker.txt && pip install -q --user --break-system-packages cowsay >/dev/null 2>&1; python3 -c 'import cowsay; print(\"ok\")'", 300000);
    if (setup.out !== "ok") throw new Error("setup failed: " + (setup.out || setup.err));
    const pid = await this.startTicker();
    await new Promise((r) => setTimeout(r, 25000));
    const ticks = (await this.sh("wc -l < /workspace/ticks.log")).out;
    if (!(Number(ticks) >= 2)) throw new Error("ticker not writing: " + ticks);
    await this.ctx.storage.put("pid", pid);
    return { pid, ticks_before_stop: ticks };
  }

  // a process survived only if the log kept growing during the wait and its newest line is under 20 s old;
  // a pid check alone cannot tell a survivor from a reused pid after a restart (review 2026-11-26)
  async tickerAlive() {
    const n = async () => Number((await this.sh("wc -l < /workspace/ticks.log 2>/dev/null || echo 0")).out) || 0;
    const before = await n();
    await new Promise((r) => setTimeout(r, 15000));
    const after = await n();
    const age = Number((await this.sh("echo $(( $(date +%s) - $(tail -n 1 /workspace/ticks.log 2>/dev/null || echo 0) ))")).out);
    return { ticks_before: before, ticks_after: after, last_tick_age_s: age, alive: after > before && age < 20 };
  }

  async snapshotAndStop() {
    let t = Date.now();
    const s = await this.box.snapshotContainer({ name: "forge-eval" });
    const snapshot_s = secs(t);
    t = Date.now(); await this.box.destroy();
    await this.ctx.storage.put("snapshot", s.id);
    return { snapshot_s, snapshot_mb: Math.round(s.size / 1e5) / 10, stop_s: secs(t) };
  }

  async create(mode) {
    if (this.box.running) await this.box.destroy();
    let t = Date.now();
    const up = await this.startFresh(null);
    if (up.rc !== 0) throw new Error("container did not start: " + up.err);
    const rec = { mode, instance: INSTANCE, create_s: secs(t), checks: {} };
    const c = rec.checks;
    c.uname = (await this.sh("uname -srm")).out;
    c.nproc = (await this.sh("nproc")).out;
    c.mem = (await this.sh("free -m | awk 'NR==2{print $2\" MB\"}'")).out;
    c.landlock_abi = (await this.sh(`python3 -c "${LANDLOCK}"`)).out;
    c.bwrap = await this.sh("bwrap --version");
    c.bwrap_proc = await this.sh("bwrap --ro-bind / / --proc /proc --dev /dev --unshare-all true && echo ok");
    c.cpu_bench_s = [];
    for (let i = 0; i < 5; i++) c.cpu_bench_s.push((await this.sh(`python3 -c "${BENCH}"`)).out);
    Object.assign(rec, await this.seed());
    if (mode === "snap") Object.assign(rec, await this.snapshotAndStop());
    rec.ok = true;
    return rec;
  }

  async check(mode) {
    const cyc = { mode, running_before: this.box.running };   // does not start the container
    const t = Date.now();
    if (mode === "snap") {
      const id = await this.ctx.storage.get("snapshot");
      if (!id) return { ...cyc, ok: false, error: "no snapshot stored" };
      if (this.box.running) await this.box.destroy();
      const up = await this.startFresh(id);
      cyc.resume_s = secs(t); cyc.up = up.out || up.err;
    } else if (!this.box.running) {
      const up = await this.startFresh(null);                 // native: a stopped container comes back from the image
      cyc.resume_s = secs(t); cyc.up = up.out || up.err;
    }
    if (cyc.up !== undefined && cyc.up !== "up") throw new Error("container did not start: " + cyc.up);
    cyc.ticker = await this.tickerAlive();                     // waits 15 s: long enough for a survivor to write
    cyc.file = (await this.sh("cat /workspace/marker.txt")).out;
    cyc.tool = (await this.sh("python3 -c 'import cowsay; print(\"ok\")'")).out;
    cyc.files_ok = cyc.file.startsWith("forge-") && cyc.tool === "ok";
    cyc.process_alive = cyc.ticker.alive;
    cyc.ok = cyc.files_ok && cyc.process_alive;
    // re-seed whatever was lost, so the next cycle measures a fresh gap instead of repeating this loss (review
    // 2026-11-26): snap loses the process every time; native after a stop loses everything
    if (!cyc.files_ok) await this.sh("rm -f /workspace/ticks.log");
    if (!cyc.files_ok || !cyc.process_alive) cyc.reseed = await this.seed();
    if (mode === "snap") Object.assign(cyc, await this.snapshotAndStop());
    return cyc;
  }

  async kill() {
    if (this.box.running) await this.box.destroy();
    await this.ctx.storage.deleteAll();
    return { killed: true };
  }

  async fetch(request) {
    const url = new URL(request.url);
    const mode = url.searchParams.get("mode");
    const op = url.pathname.slice(1);
    try {
      if (op === "create") return Response.json(await this.create(mode));
      if (op === "check") return Response.json(await this.check(mode));
      if (op === "kill") return Response.json(await this.kill());
      // snapshots: the Workers types (2026-11) have no delete call, so old ones are left; their size is recorded
      return new Response("not found", { status: 404 });
    } catch (e) {
      return Response.json({ ok: false, error: String(e && e.stack || e).slice(0, 600) }, { status: 500 });
    }
  }
}

export default {
  async fetch(request, env) {
    if (!authorized(request, env)) return new Response("unauthorized", { status: 401 });
    const url = new URL(request.url);
    const mode = url.searchParams.get("mode");
    if (request.method !== "POST" || !["/create", "/check", "/kill"].includes(url.pathname) || !MODES.has(mode)) {
      return new Response("POST /create|/check|/kill?mode=snap|native", { status: 400 });
    }
    return env.BOX.get(env.BOX.idFromName("eval-" + mode)).fetch(request);
  },
};
