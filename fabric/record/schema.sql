-- forge-record (Cloudflare D1): who can act in the Forge, who has acted, when and what (owner, 2026-11-27).
-- One register of agent instances and one activity log, shared by every connector and agent. A first slice of
-- queue item P11 (one queryable store). Apply: fabric/record/apply.sh, or the D1 query API.

-- An agent instance: a unique name (a craftsperson), a class (its role), the model it runs on now, where it
-- lives, who started it. Instances come and go; the class and name keep many of one class apart.
CREATE TABLE IF NOT EXISTS agents (
  id          TEXT PRIMARY KEY,           -- e.g. "stradivari-3f2a"
  name        TEXT NOT NULL,              -- "Stradivari"
  class       TEXT NOT NULL,              -- main | lead | coder | scout | researcher | worker | chat | job | owner
  model       TEXT,                       -- current model, e.g. "gpt-oss-120b" (may change)
  provider    TEXT,
  interface   TEXT NOT NULL,              -- discord | lobehub | claude | chatgpt | gemini | glm | github-actions | ...
  parent      TEXT REFERENCES agents(id), -- the instance that started this one (a lead over its coders)
  level       INTEGER NOT NULL DEFAULT 0, -- depth under the owner
  can         TEXT NOT NULL DEFAULT 'read', -- comma list: read, issue, pr, workflow, merge, deploy, sandbox
  verified    INTEGER NOT NULL DEFAULT 0, -- 1: created by a token-holding Forge service; 0: self-described
  created_at  TEXT NOT NULL,
  last_at     TEXT,
  retired_at  TEXT,
  note        TEXT
);
CREATE INDEX IF NOT EXISTS agents_class ON agents(class, retired_at);
-- two active instances never share a name (register retries on a clash)
CREATE UNIQUE INDEX IF NOT EXISTS agents_active_name ON agents(name) WHERE retired_at IS NULL;

-- One line per action. Content is a short summary, never secrets or full outputs.
CREATE TABLE IF NOT EXISTS events (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  at        TEXT NOT NULL,
  agent     TEXT REFERENCES agents(id),   -- known instance, when there is one
  actor     TEXT,                         -- the name the actor gave (claimed), always kept
  verified  INTEGER NOT NULL DEFAULT 0,   -- 1: the actor's identity is known (a Forge service acting as a registered instance); 0: the name was claimed
  interface TEXT,
  model     TEXT,
  action    TEXT NOT NULL,                -- run, tool, issue, comment, pr, workflow, merge, deploy, note, ...
  target    TEXT,                         -- what it acted on: an item id, PR number, workflow, chat id
  link      TEXT,
  summary   TEXT,
  chat      TEXT
);
CREATE INDEX IF NOT EXISTS events_at ON events(at);
CREATE INDEX IF NOT EXISTS events_agent ON events(agent, at);
