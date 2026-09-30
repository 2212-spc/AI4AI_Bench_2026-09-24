import numpy as np
# brackets (ok, diverged) at wu=0.02, qk=0
br={1e7:(0.01834,0.0189),3e7:(0.0099,0.011),1e8:(0.0057,0.006),3e8:(0.00346,0.00359)}
Ns=np.array(sorted(br)); mid=np.array([np.sqrt(br[n][0]*br[n][1]) for n in Ns])
x=np.log(Ns/1e8); y=np.log(mid)
A=np.vstack([np.ones_like(x),x]).T
(c,m),res,_,_=np.linalg.lstsq(A,y,rcond=None)
h0=np.exp(c); delta=-m
print("h0(wu=.02)",h0,"delta",delta,"resid",y-A@np.array([c,m]))
# pairwise slopes and bracket-extreme slopes 1e7 vs 3e8
lo_d=np.log(br[1e7][0]/br[3e8][1])/np.log(30); hi_d=np.log(br[1e7][1]/br[3e8][0])/np.log(30)
print("delta range from extreme brackets 1e7-3e8:",lo_d,hi_d)
Np=7e9
e02=h0*(Np/1e8)**-delta
print("prod edge wu=.02:",e02,"log10",np.log10(e02))
# bracket-based range for prod edge using 3e8 bracket and delta range
for d in (lo_d,delta,hi_d):
  for e in br[3e8]: print(" d",round(d,3),"e3e8",e,"->",e*(Np/3e8)**-d)
# wu=0.05: brackets 3e8 (0.0045,0.00484), 1e7 (0.0239,0.026)
m05_3e8=np.sqrt(0.0045*0.00484); m05_1e7=np.sqrt(0.0239*0.026)
d05=np.log(m05_1e7/m05_3e8)/np.log(30)
print("delta from wu=.05 pair",d05, "ratio wu.05/.02 at 3e8", m05_3e8/mid[-1], "at 1e7", m05_1e7/mid[0])
e05=m05_3e8*(Np/3e8)**-delta
print("prod edge wu=.05:",e05, "vs 0.000829", "range", 0.0045*(Np/3e8)**-hi_d, 0.00484*(Np/3e8)**-lo_d)
# wu=0 at 3e8: (0.0025,0.00274)
m0=np.sqrt(0.0025*0.00274)
print("wu0 3e8 edge ~",m0,"Qmin=0.05/edge:",0.05/0.00274,0.05/m0,0.05/0.0025)
# warmup factor fit: (1+wu/w0)^omega ratios r02=e02/e0, r05=e05/e0 at 3e8
r02=mid[-1]/m0; r05=m05_3e8/m0
print("r02",r02,"r05",r05)
best=None
for w0 in np.logspace(-4,0,400):
  om=np.log(r02)/np.log(1+0.02/w0); pred=(1+0.05/w0)**om
  err=abs(np.log(pred/r05))
  if best is None or err<best[0]: best=(err,w0,om)
print("w0,omega",best)
# q2
Qmin=0.05/0.00274
print("q2 lo/hi log10:",np.log10(e02*Qmin),np.log10(e02*60))
print("q4: edge_on min",e02*Qmin,"vs .00468 ; q5 edge_on range",e02*Qmin,e02*60,"vs .0241")
