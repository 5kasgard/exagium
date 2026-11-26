// Discord control for the owner (owner, 2026-11-26: "a heap of control from Discord"). Discord calls
// POST https://act.exagium.xyz/discord for each slash command; this answers within Discord's 3-second limit by
// starting GitHub workflows (the same path the scheduler and the MCP tools use) and replying only to the owner.
// Replies from agents arrive later through the webhook, as their persona (fabric/discord/post.py).
//
// Gates, in order: Discord's Ed25519 signature (DISCORD_PUBLIC_KEY), then the caller's user id must equal
// DISCORD_OWNER_ID (a Worker secret, not in the repo). Anyone else gets "not allowed". Only fixed workflows run.

export const AGENTS = ["lead", "free", "research", "coder"];
export const RUN = ["schedules.yml", "sandbox-eval.yml", "dash-snapshot.yml", "models-rank.yml", "keys-probe.yml",
  "box-sizes.yml", "workbench-manage.yml"];
const WATCH = ["worker", "schedules", "sandbox-eval", "tests", "notify", "dash-snapshot", "review", "propose"];
const NAMES = { lead: "Lead", free: "Scout", research: "Researcher", coder: "Coder" };

// Registered by act-deploy.yml (PUT /applications/{id}/commands). Owner-only by the id check; also hidden from
// members without Manage Server (default_member_permissions "32").
export const COMMANDS = [
  { name: "ask", description: "Ask a Forge agent; the reply arrives in this server as that agent",
    default_member_permissions: "32", options: [
      { type: 3, name: "agent", description: "Who", required: true, choices: AGENTS.map((a) => ({ name: NAMES[a], value: a })) },
      { type: 3, name: "prompt", description: "What to do", required: true, max_length: 1500 }] },
  { name: "run", description: "Start a Forge job now", default_member_permissions: "32", options: [
      { type: 3, name: "job", description: "Which job", required: true,
        choices: RUN.filter((w) => w !== "workbench-manage.yml").map((w) => ({ name: w.replace(".yml", ""), value: w })) }] },
  { name: "status", description: "Latest result of the main Forge jobs", default_member_permissions: "32" },
  { name: "prs", description: "Open pull requests", default_member_permissions: "32" },
];

const hex = (h) => new Uint8Array((h.match(/../g) || []).map((x) => parseInt(x, 16)));

export async function verify(request, body, publicKey) {
  const sig = request.headers.get("x-signature-ed25519"), ts = request.headers.get("x-signature-timestamp");
  if (!sig || !ts || !publicKey || sig.length !== 128) return false;
  // replay guard (review 2026-11-26): a captured signed request must not start work again later
  if (!(Math.abs(Date.now() / 1000 - Number(ts)) <= 300)) return false;
  try {
    const key = await crypto.subtle.importKey("raw", hex(publicKey), { name: "Ed25519" }, false, ["verify"]);
    return await crypto.subtle.verify("Ed25519", key, hex(sig), new TextEncoder().encode(ts + body));
  } catch { return false; }
}

const say = (content) => content;   // commands return text; handle() sends it as an edit of the deferred reply
const opt = (i, n) => (i.data.options || []).find((o) => o.name === n)?.value;
const RUNS = "https://github.com/5kasgard/exagium/actions";

export async function command(i, env, gh, repo) {
  const who = i.member?.user?.id || i.user?.id;
  if (!env.DISCORD_OWNER_ID || who !== env.DISCORD_OWNER_ID) return say("Not allowed.");
  const name = i.data?.name;
  if (name === "ask") {
    const agent = opt(i, "agent"), prompt = String(opt(i, "prompt") || "").trim();
    if (!AGENTS.includes(agent) || !prompt || prompt.length > 1500) return say("Need an agent and a prompt of at most 1,500 characters.");
    await gh(env, `/repos/${repo}/actions/workflows/workbench-manage.yml/dispatches`, "POST",
      { ref: "main", inputs: { mode: "run", agent, prompt } });
    return say(`Asked **${NAMES[agent]}**. The reply will be posted here as ${NAMES[agent]} (usually 1-3 min). Runs: <${RUNS}/workflows/workbench-manage.yml>`);
  }
  if (name === "run") {
    const job = opt(i, "job");
    if (!RUN.includes(job) || job === "workbench-manage.yml") return say("Unknown job.");
    await gh(env, `/repos/${repo}/actions/workflows/${job}/dispatches`, "POST", { ref: "main" });
    return say(`Started **${job.replace(".yml", "")}**. <${RUNS}/workflows/${job}>`);
  }
  if (name === "status") {
    const j = await gh(env, `/repos/${repo}/actions/runs?branch=main&per_page=60`);
    const seen = new Map();
    for (const r of j.workflow_runs || []) if (WATCH.includes(r.name) && !seen.has(r.name)) seen.set(r.name, r);
    const icon = (r) => r.status !== "completed" ? "⏳" : r.conclusion === "success" ? "✅" : r.conclusion === "skipped" ? "⏭️" : "❌";
    const lines = WATCH.filter((n) => seen.has(n)).map((n) => { const r = seen.get(n);
      return `${icon(r)} **${n}** ${r.status === "completed" ? r.conclusion : r.status} · ${r.updated_at.slice(5, 16).replace("T", " ")}Z`; });
    return say(lines.length ? lines.join("\n") : "No recent runs found.");
  }
  if (name === "prs") {
    const j = await gh(env, `/repos/${repo}/pulls?state=open&per_page=15`);
    const lines = (j || []).map((p) => `#${p.number} ${String(p.title).replace(/[\r\n@]/g, " ").slice(0, 90)} · ${p.user?.login || "?"} <${p.html_url}>`);
    return say(lines.length ? lines.join("\n") : "No open pull requests.");
  }
  return say("Unknown command.");
}

const reply = (content) => ({ content: String(content).slice(0, 1900), allowed_mentions: { parse: [] } });

// Discord allows 3 s for the first answer. Answer "thinking…" (type 5, only visible to the caller) at once, then
// do the work and edit that answer (review 2026-11-26: a slow GitHub call must not fail and tempt a retry).
export async function handle(request, env, gh, repo, ctx, post = fetch) {
  const body = await request.text();
  if (!(await verify(request, body, env.DISCORD_PUBLIC_KEY))) return new Response("bad signature", { status: 401 });
  let i;
  try { i = JSON.parse(body); } catch { return new Response("bad json", { status: 400 }); }
  if (i.type === 1) return Response.json({ type: 1 });            // Discord's PING when the endpoint is saved
  if (i.type !== 2) return Response.json({ type: 4, data: { ...reply("Not supported yet."), flags: 64 } });
  const work = (async () => {
    let text;
    try { text = await command(i, env, gh, repo); } catch (e) { text = "Failed: " + String(e && e.message || e).slice(0, 300); }
    await post(`https://discord.com/api/v10/webhooks/${i.application_id}/${i.token}/messages/@original`,
      { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify(reply(text)) });
  })();
  if (ctx?.waitUntil) ctx.waitUntil(work); else await work;
  return Response.json({ type: 5, data: { flags: 64 } });
}
