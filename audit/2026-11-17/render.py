#!/usr/bin/env python3
"""One-queue checker and renderer. Usage: python3 render.py queue.json > QUEUE.md   (exit 1 if the file is invalid).
Checks: unique ids, known values, blockers that exist, no cycles, a next step and refs on every item.
Batches: related items done together (one checkout, one review, one CI run), ordered by the work they unlock.
Order: can run now (by optional priority, then how many open items it unlocks, directly or through others, then value, then cost), waiting on a blocker, needs the owner, scheduled, parked, done."""
import json, sys

KIND = {'test', 'build', 'fix', 'doc', 'infra', 'decision'}
STATUS = {'open', 'in progress', 'blocked', 'waiting on the owner', 'scheduled', 'parked', 'done'}
NEEDS = {'free', 'paid API', 'subscription usage', 'the owner', 'other people'}
VALUE = {'high': 0, 'medium': 1, 'low': 2}
SEV = {'critical', 'major', 'minor', 'none'}

q = json.load(open(sys.argv[1]))
items = q['items']
errs = []
ids = [i['id'] for i in items]
if len(ids) != len(set(ids)):
    errs.append('duplicate ids: ' + ', '.join(sorted({x for x in ids if ids.count(x) > 1})))
by = {i['id']: i for i in items}
for i in items:
    if 'origin' in i and i['origin'] not in ('follow-up', 'new'):   # follow-up: from finishing or checking work; new: a new direction
        errs.append(f"{i.get('id')}: origin must be follow-up or new")
    for f, allowed in (('kind', KIND), ('status', STATUS), ('needs', NEEDS), ('value', set(VALUE)), ('severity', SEV)):
        if i.get(f) not in allowed:
            errs.append(f"{i['id']}: bad {f} {i.get(f)!r}")
    if not i.get('next'):
        errs.append(f"{i['id']}: no next step")
    if not i.get('refs'):
        errs.append(f"{i['id']}: no refs")
    if 'priority' in i and not isinstance(i['priority'], int):
        errs.append(f"{i['id']}: priority must be an integer (lower first)")
    if not isinstance(i.get('cost'), (int, float)):
        errs.append(f"{i['id']}: cost must be a number (paid estimate, 0 if free)")
    for b in i.get('blocked_by', []):
        if b not in by:
            errs.append(f"{i['id']}: unknown blocker {b}")


def cyc(n, seen):
    if n in seen:
        return True
    return any(cyc(b, seen | {n}) for b in by.get(n, {}).get('blocked_by', []) if b in by)


for i in items:
    if cyc(i['id'], set()):
        errs.append(f"{i['id']}: blocker cycle")
LIVE = ('open', 'in progress', 'blocked', 'waiting on the owner', 'scheduled')
for i in items:
    if i['status'] in LIVE and not i.get('touches'):
        errs.append(f"{i['id']}: open items must say what they touch (files, paths, branches): add 'touches'")
bids = [b['id'] for b in q.get('batches', [])]
for b in q.get('batches', []):
    for x in b['items']:
        if x not in by:
            errs.append(f"batch {b['id']}: unknown item {x}")
        elif by[x].get('batch') != b['id']:
            errs.append(f"{x}: listed in batch {b['id']} but its batch field says {by[x].get('batch')!r}")
for i in items:
    if i.get('batch') and i['batch'] not in bids:
        errs.append(f"{i['id']}: unknown batch {i['batch']}")
if errs:
    print('\n'.join(errs), file=sys.stderr)
    sys.exit(1)


def open_blockers(i):
    return [b for b in i.get('blocked_by', []) if by[b]['status'] != 'done']


def unlocks(i, memo={}):
    """Open items that wait on i, directly or through other items: the critical path goes first."""
    if i['id'] not in memo:
        memo[i['id']] = {j['id'] for j in items if j['status'] != 'done' and i['id'] in j.get('blocked_by', [])}
        for d in list(memo[i['id']]):
            memo[i['id']] |= unlocks(by[d])
    return memo[i['id']]


groups = [('Can run now', lambda i: i['status'] in ('open', 'in progress') and not open_blockers(i) and i['needs'] != 'the owner'),
          ('Waiting on a blocker', lambda i: i['status'] in ('open', 'in progress', 'blocked') and open_blockers(i)),
          ('Needs the owner', lambda i: i['status'] == 'waiting on the owner' or (i['needs'] == 'the owner' and i['status'] not in ('done', 'parked', 'scheduled'))),
          ('Scheduled', lambda i: i['status'] == 'scheduled'),
          ('Parked', lambda i: i['status'] == 'parked'),
          ('Done', lambda i: i['status'] == 'done')]
def key(t):
    return t.split(' (')[0].strip().rstrip('/')


def overlap(a, b):
    """Shared touch points: the same path, or one path inside the other."""
    hits = set()
    for x in a.get('touches', []):
        for y in b.get('touches', []):
            kx, ky = key(x), key(y)
            if kx == ky or kx.startswith(ky + '/') or ky.startswith(kx + '/'):
                hits.add(kx if len(kx) >= len(ky) else ky)
    return hits


live = [i for i in items if i['status'] in LIVE]
shared = []
for n, a in enumerate(live):
    for b in live[n + 1:]:
        h = overlap(a, b)
        ordered = b['id'] in unlocks(a) or a['id'] in unlocks(b)  # a dependency already sequences them
        if h and not ordered and (a.get('batch') != b.get('batch') or not a.get('batch')):
            shared.append((a['id'], a.get('batch', '-'), b['id'], b.get('batch', '-'), sorted(h)))
for s_ in shared:
    print(f"note: {s_[0]} ({s_[1]}) and {s_[2]} ({s_[3]}) both touch {', '.join(s_[4])}", file=sys.stderr)

placed = set()
out = [f"# {q['title']}", '', q.get('intro', ''), '']
if shared:
    out += ['## Shared work across batches, not yet ordered (do together, or add a blocker)', '', '| item | batch | item | batch | both touch |', '|---|---|---|---|---|']
    out += [f"| {a} | {ba} | {b} | {bb} | {', '.join(h)} |" for a, ba, b, bb, h in shared]
    out.append('')
if q.get('batches'):
    def bkey(b):
        live = [by[x] for x in b['items'] if by[x]['status'] not in ('done', 'parked')]
        ready = any(not open_blockers(i) for i in live)
        reach = set().union(*[unlocks(i) for i in live]) - set(b['items']) if live else set()
        return (not ready, -len(reach), b['id'])
    out += ['## Batches (related items done together)', '', '| batch | items (open) | ready now | unlocks outside it | why together | worker |', '|---|---|---|---|---|---|']
    for b in sorted(q['batches'], key=bkey):
        live = [x for x in b['items'] if by[x]['status'] not in ('done', 'parked')]
        ready = [x for x in live if not open_blockers(by[x])]
        reach = set().union(*[unlocks(by[x]) for x in live]) - set(b['items']) if live else set()
        out.append(f"| {b['id']} {b['name']} | {', '.join(live) or 'all done'} | {', '.join(ready) or '-'} | {len(reach) or ''} | {b['why']} | {b['worker']} |")
    out.append('')
for name, pred in groups:
    rows = [i for i in items if i['id'] not in placed and pred(i)]
    rows.sort(key=lambda i: (i.get('priority', 99), -len(unlocks(i)), VALUE[i['value']], i['cost'], i['id']))
    if not rows:
        continue
    out += [f'## {name} ({len(rows)})', '', '| id | batch | unlocks | severity | value | kind | title | next | refs |', '|---|---|---|---|---|---|---|---|---|']
    for i in rows:
        placed.add(i['id'])
        nxt = i['next'] + (f" (blocked by {', '.join(open_blockers(i))})" if open_blockers(i) else '')
        out.append(f"| {i['id']} | {i.get('batch', '')} | {len(unlocks(i)) or ''} | {i['severity']} | {i['value']} | {i['kind']} | {i['title']} | {nxt} | {'; '.join(i['refs'])} |")
    out.append('')
left = [i['id'] for i in items if i['id'] not in placed]
if left:
    print('items fell through every group: ' + ', '.join(left), file=sys.stderr)
    sys.exit(1)
print('\n'.join(out))
