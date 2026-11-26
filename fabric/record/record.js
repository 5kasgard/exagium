// forge-record helpers (D1): the agent register and the activity log, shared by forge-agent, forge-act and the
// Discord commands. See schema.sql. Everything here is best-effort: a failed write never stops the work it records.

import MAKERS from "./makers.json" with { type: "json" };

export const CLASSES = ["main", "lead", "coder", "scout", "researcher", "worker", "chat", "job", "owner"];
const now = () => new Date().toISOString().slice(0, 19) + "Z";
const hex = (n) => [...crypto.getRandomValues(new Uint8Array(n))].map((b) => b.toString(16).padStart(2, "0")).join("");
export const slug = (s) => String(s).normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
export const redact = (t) => t == null ? t : String(t)
  .replace(/(bearer|authorization|api[_-]?key|token|secret|password)(["'\s:=]+)[A-Za-z0-9._\-]{12,}/gi, "$1$2<redacted>")
  .replace(/\b(sk|gh[pousr]|github_pat|csk|gsk|nvapi)[-_][A-Za-z0-9_\-]{16,}/g, "<redacted>")
  .replace(/https:\/\/discord(app)?\.com\/api\/webhooks\/\S+/g, "<redacted>");
export const clip = (s, n = 160) => (s == null ? null : String(s).replace(/\s+/g, " ").trim().slice(0, n));

// A name for a new instance: one no active instance has, at random; if all are in use, a name with a number.
export function pickName(active, pool = MAKERS.names, rand = Math.random) {
  const used = new Set(active);
  const free = pool.filter((n) => !used.has(n));
  if (free.length) return free[Math.floor(rand() * free.length)];
  const base = pool[Math.floor(rand() * pool.length)];
  let k = 2;
  while (used.has(`${base} ${k}`)) k++;
  return `${base} ${k}`;
}

export const RETIRE_DAYS = 7;   // an instance unused this long is retired and its name freed (review 2026-11-27)

export async function register(db, a) {
  if (!db) return null;
  try {
    await db.prepare("UPDATE agents SET retired_at = ? WHERE retired_at IS NULL AND verified = 1 AND last_at < ?")
      .bind(now(), new Date(Date.now() - RETIRE_DAYS * 864e5).toISOString().slice(0, 19) + "Z").run();
  } catch {}
  for (let attempt = 0; attempt < 3; attempt++) {   // two chats starting together may pick one name: the index refuses the second
  try {
    const active = (await db.prepare("SELECT name FROM agents WHERE retired_at IS NULL").all()).results.map((r) => r.name);
    const name = a.name || pickName(active);
    const id = `${slug(name)}-${hex(2)}`;
    await db.prepare(`INSERT INTO agents (id, name, class, model, provider, interface, parent, level, can, verified, created_at, last_at, note)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`).bind(id, name, CLASSES.includes(a.class) ? a.class : "chat", a.model || null,
      a.provider || null, a.interface || "unknown", a.parent || null, a.level || 0, a.can || "read", a.verified ? 1 : 0, now(), now(), clip(a.note, 300)).run();
    return { id, name };
  } catch (e) {
    if (/UNIQUE/i.test(String(e)) && !a.name) continue;
    console.log(JSON.stringify({ record: "register failed", e: String(e).slice(0, 200) })); return null;
  }
  }
  return null;
}

export async function touch(db, id, model) {
  if (!db || !id) return;
  try { await db.prepare("UPDATE agents SET last_at = ?, model = COALESCE(?, model) WHERE id = ?").bind(now(), model || null, id).run(); } catch {}
}

export async function log(db, e) {
  if (!db) return;
  try {
    await db.prepare(`INSERT INTO events (at, agent, actor, verified, interface, model, action, target, link, summary, chat)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`).bind(now(), e.agent || null, clip(e.actor, 120), e.verified ? 1 : 0, e.interface || null,
      e.model || null, clip(e.action, 40) || "note", clip(redact(e.target), 200), clip(e.link, 300), clip(redact(e.summary), 300), e.chat || null).run();
  } catch (x) { console.log(JSON.stringify({ record: "log failed", e: String(x).slice(0, 200) })); }
}

// Reads, for tools and commands.
export async function agents(db, { active = true, cls } = {}) {
  let q = "SELECT id, name, class, model, interface, parent, level, can, verified, created_at, last_at, retired_at FROM agents";
  const w = [], b = [];
  if (active) w.push("retired_at IS NULL");
  if (cls) { w.push("class = ?"); b.push(cls); }
  if (w.length) q += " WHERE " + w.join(" AND ");
  q += " ORDER BY last_at DESC LIMIT 100";
  return (await db.prepare(q).bind(...b).all()).results;
}

export async function activity(db, { agent, since, limit = 30 } = {}) {
  let q = "SELECT e.at, e.agent, a.name, a.class, e.actor, e.verified, e.interface, e.model, e.action, e.target, e.link, e.summary FROM events e LEFT JOIN agents a ON a.id = e.agent";
  const w = [], b = [];
  // a claimed actor name only matches events with no known instance, so a caller cannot pass as a registered agent
  if (agent) { w.push("(e.agent = ? OR a.name = ? OR (e.agent IS NULL AND e.actor = ?))"); b.push(agent, agent, agent); }
  if (since) { w.push("e.at >= ?"); b.push(since); }
  if (w.length) q += " WHERE " + w.join(" AND ");
  q += " ORDER BY e.id DESC LIMIT ?"; b.push(Math.min(200, Math.max(1, +limit || 30)));
  return (await db.prepare(q).bind(...b).all()).results;
}
