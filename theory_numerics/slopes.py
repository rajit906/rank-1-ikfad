import sys, numpy as np
d=np.loadtxt(sys.argv[1]); t,E=d[:,0],d[:,1]
out=[]
for w in sys.argv[2:]:
    lo,hi=(float(x) for x in w.split(':'))
    k=(t>=lo)&(t<=hi)&(E>0)
    out.append(f"[{lo:.0e},{hi:.0e}]={np.polyfit(np.log(t[k]),np.log(E[k]),1)[0]:+.3f}" if k.sum()>5 else "n/a")
print("  ".join(out))
