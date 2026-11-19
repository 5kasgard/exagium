import json,sys
sys.path.insert(0,'runner/proposals'); from judge import paired
for t in sys.argv[1:]:
    d=json.load(open(f'runner/results/2026-11-20-p5a/dev-{t}.json'))['candidates']
    rep=json.load(open(f'runner/results/2026-11-20-p5a/reproduce-{t}.json'))
    repro=all(r.get('identical') for r in rep if r.get('candidate')=='label-mlp2l')
    m=d['label-mlp2l']; mm,ss=paired(m['items'],d['label-mlp16']['items'])
    print(f"== {t}: mlp2l {m['mean']:.4f} Ir {m.get('ir_per_item'):.0f}; vs mlp16 {mm:+.4f} ± {ss:.4f} -> depth helps: {mm>2*ss}; reproduces: {repro}")
    dom=[]
    for b,c in d.items():
        if b=='label-mlp2l' or 'items' not in c: continue
        x,s=paired(m['items'],c['items'])
        if (x<=s and c['ir_per_item']<m['ir_per_item']) or (x<-s and c['ir_per_item']<=m['ir_per_item']): dom.append(b)
        print(f"   {b:13s} {c['mean']:.4f} Ir {c['ir_per_item']:.0f}  mlp2l-minus {x:+.4f} ± {s:.4f}")
    print('   advance (not dominated):', not dom, dom)
