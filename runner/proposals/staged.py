"""Successive-halving core with e-value elimination (queue K72). Pure function, no I/O."""


def staged_select(scores, folds=(60, 120, 240, 480), lam=0.5, cut=0.1, keep=0.5):
    """Run successive halving over the given dev folds, eliminating candidates on e-value.

    scores maps a name to per-item differences against label-linear (each in [-1, 1]).
    Each candidate's e-value is the running product of (1 + lam*d) over its items so far.
    A candidate is dropped when its e-value falls below cut. After each fold, only the top
    `keep` fraction (by e-value) survives.

    Returns {'survivors': [...], 'scored_items': int, 'eliminated_at': {name: fold_index}}.
    """
    alive = list(scores)
    evalues = {n: 1.0 for n in scores}
    eliminated_at = {}
    scored = 0
    prev = 0
    for fi, end in enumerate(folds):
        for name in alive:
            row = scores[name]
            for i in range(prev, min(end, len(row))):
                evalues[name] *= 1.0 + lam * row[i]
                scored += 1
        for name in list(alive):
            if evalues[name] < cut:
                eliminated_at[name] = fi
                alive.remove(name)
        if len(alive) > 1:
            ranked = sorted(alive, key=lambda n: -evalues[n])
            n_keep = max(1, int(len(alive) * keep))
            for name in ranked[n_keep:]:
                eliminated_at[name] = fi
            alive = ranked[:n_keep]
        prev = end
    return {'survivors': sorted(alive), 'scored_items': scored, 'eliminated_at': eliminated_at}
