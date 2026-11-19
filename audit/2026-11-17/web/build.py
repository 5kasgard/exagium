"""Build a web view of the Forge queue from queue.json (the record). Usage: python3 build.py queue.json commit out.html"""
import json, re, sys, html
import networkx as nx

qpath, commit, out = sys.argv[1], sys.argv[2], sys.argv[3]
q = json.load(open(qpath))
items = q['items']
by = {i['id']: i for i in items}
GATE = re.compile(r'\bonly (if|when)\b|\bproceeds? only\b|go/no-go', re.I)

def key(t):
    return t.split(' (')[0].strip().rstrip('/')

G = nx.Graph()
nodes, edges = [], []
for i in items:
    nodes.append(dict(id=i['id'], type='item', status=i['status'], batch=i.get('batch', ''), kind=i['kind'],
                      needs=i['needs'], title=i['title'], next=i['next'], refs=i['refs'],
                      touches=i.get('touches', []), value=i.get('value', ''), gate=bool(GATE.search(i['next'])) and i['status'] != 'done'))
    G.add_node(i['id'])
    for b in i.get('blocked_by', []):
        if b in by:
            edges.append(dict(s=b, t=i['id'], kind='needs'))
            G.add_edge(b, i['id'], weight=3)
    for b in i.get('informs', []):
        if b in by:
            edges.append(dict(s=i['id'], t=b, kind='informs'))
            G.add_edge(i['id'], b, weight=1)
    if i['status'] not in ('done', 'parked'):
        if i['needs'] == 'the owner':   # a decision or action only the owner can take
            edges.append(dict(s='OWNER', t=i['id'], kind='owner'))
            G.add_edge('OWNER', i['id'], weight=1)
        elif i['status'] == 'waiting on the owner':   # sent to the owner after failed attempts (worker rule)
            edges.append(dict(s='OWNER', t=i['id'], kind='escalated'))
            G.add_edge('OWNER', i['id'], weight=1)
    if i['status'] != 'done':
        for t in i.get('touches', []):
            f = 'file:' + key(t)
            if f not in G:
                nodes.append(dict(id=f, type='file', title=key(t)))
                G.add_node(f)
            edges.append(dict(s=i['id'], t=f, kind='writes'))
            G.add_edge(i['id'], f, weight=0.5)
nodes.append(dict(id='OWNER', type='owner', title='The owner: keys, permissions, decisions'))
G.add_node('OWNER')
# batches pull their members together (layout only, not drawn)
for b in q.get('batches', []):
    m = [x for x in b['items'] if x in by]
    for a, c in zip(m, m[1:]):
        G.add_edge(a, c, weight=1.5)
live = [n for n in G if n == 'OWNER' or n.startswith('file:') or by[n]['status'] != 'done']
pos = {}
def pack(sub, x0, y0, width):
    """Lay out each connected piece on its own, biggest first, in rows; returns the next y."""
    comps = sorted(nx.connected_components(sub), key=len, reverse=True)
    x, y, rowh = x0, y0, 0
    for c in comps:
        n = len(c)
        size = 1300 if n > 40 else 520 if n > 8 else 200 if n > 1 else 70
        p = nx.spring_layout(sub.subgraph(c), seed=7, k=1.6 / max(n, 1) ** 0.5, iterations=400, weight='weight') if n > 1 else {next(iter(c)): (0, 0)}
        if x + size > x0 + width and x > x0:
            x, y, rowh = x0, y + rowh + 60, 0
        for k, (a, b2) in p.items():
            pos[k] = (x + size / 2 + a * size / 2.2, y + size / 2 + b2 * size / 2.2)
        x += size + 60; rowh = max(rowh, size)
    return y + rowh + 200
ynext = pack(G.subgraph(live), 0, 0, 1500)
pack(G.subgraph([n for n in G if n not in live]), 0, ynext, 1500)
COMPONENTS = [len(c) for c in sorted(nx.connected_components(G.subgraph(live)), key=len, reverse=True)]
for n in nodes:
    x, y = pos[n['id']]
    n['x'], n['y'] = round(float(x), 1), round(float(y), 1)
batches = {b['id']: b['name'] for b in q.get('batches', [])}
data = dict(nodes=nodes, edges=edges, batches=batches, commit=commit,
            source=f'https://github.com/5kasgard/exagium/blob/{commit}/audit/2026-11-17/queue.json')
tpl = open(sys.argv[0].replace('build.py', 'template.html')).read()
open(out, 'w').write(tpl.replace('/*DATA*/null', json.dumps(data, ensure_ascii=False).replace('</', '<\\/')))
print(len(nodes), 'nodes', len(edges), 'edges; live pieces by size:', COMPONENTS)
