import sys, random
random.seed(1)
I,H,O=32,64,10
W1=[[random.random()-.5 for _ in range(I)] for _ in range(H)]
W2=[[random.random()-.5 for _ in range(H)] for _ in range(O)]
n=int(sys.argv[1]) if len(sys.argv)>1 else 50
c=0
for k in range(n):
    x=[random.random()-.5 for _ in range(I)]
    h=[max(0.0,sum(w*v for w,v in zip(r,x))) for r in W1]
    o=[sum(w*v for w,v in zip(r,h)) for r in W2]
    c+= o.index(max(o))==k%O
print(c)
