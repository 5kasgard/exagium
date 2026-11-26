"""Successive-halving core with e-value elimination (queue K72)."""
def staged_select(scores, folds, lam=0.5, cut=0.1, keep=0.5):
    names = list(scores)
    st = {n: {'e': 1.0, 'alive': True, 'elim': None} for n in names}
    scored = 0
    prev = 0
    for f, end in enumerate(folds):
        for i in range(prev, end):
            for n in names:
                s = st[n]
                if not s['alive']:
                    continue
                d = scores[n][i] if i < len(scores[n]) else 0.0
                s['e'] *= 1 + lam * d
                scored += 1
                if s['e'] < cut:
                    s['alive'] = False
                    s['elim'] = f
        alive = [n for n in names if st[n]['alive']]
        if alive:
            k = max(1, int(len(alive) * keep + 0.5))
            ranked = sorted(alive, key=lambda n: st[n]['e'], reverse=True)
            keep_set = set(ranked[:k])
            for n in alive:
                if n not in keep_set:
                    st[n]['alive'] = False
                    st[n]['elim'] = f
        prev = end
        if not any(st[n]['alive'] for n in names):
            break
    survivors = [n for n in names if st[n]['alive']]
    elim = {n: st[n]['elim'] for n in names if st[n]['elim'] is not None}
    return {'survivors': survivors, 'scored_items': scored, 'eliminated_at': elim}


def scored_items(scores, folds, lam=0.5, cut=0.1, keep=0.5):
    """Number of (candidate, item) pairs staged_select reads before elimination and keep."""
    return staged_select(scores, folds, lam, cut, keep)['scored_items']
