import sys, numpy as np
rng=np.random.default_rng(1)
W1=rng.standard_normal((128,64)).astype(np.float32); W2=rng.standard_normal((10,128)).astype(np.float32)
X=rng.standard_normal((int(sys.argv[1]) if len(sys.argv)>1 else 256,64)).astype(np.float32)
o=np.maximum(X@W1.T,0)@W2.T
print(int((o.argmax(1)==np.arange(len(X))%10).sum()))
