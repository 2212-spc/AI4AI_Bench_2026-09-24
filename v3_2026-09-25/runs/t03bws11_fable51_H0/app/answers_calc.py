import numpy as np, json
E,A,a,B,b = 1.9057,636.0,0.3431,437.9,0.2703   # fresh law (joint fit)
def base(N,Deff): return E+A*N**-a+B*Deff**-b
def Drep(D,U,R):
    e=D/U
    return D if e<=1 else U*(1+R*(1-np.exp(-(e-1)/R)))
# fresh q=0.6 gain points at N=5e7
Dg=np.array([1e9,4e9,1.6e10]); G=np.array([4.9735-4.8283,4.4665-4.3886,4.1176-4.0831])
# model P (penalty): gain = c D^-b - pen  (least squares linear in c,pen)
X=np.c_[Dg**-b, -np.ones(3)]; c,pen=np.linalg.lstsq(X,G,rcond=None)[0]
gP=lambda D: c*D**-b-pen
# model X (exponent): filtered data term = B2 D^-b2 ; gain = B D^-b - B2 D^-b2
dt=np.array([4.8283,4.3886,4.0831])-(E+A*5e7**-a)
b2,lB2=np.polyfit(np.log(Dg),np.log(dt),1); b2=-b2; B2=np.exp(lB2)
gX=lambda D: B*D**-b-B2*D**-b2
print("penalty model c,pen",c,pen,"resid",gP(Dg)-G)
print("exponent model B2,b2",B2,b2,"resid",gX(Dg)-G)
for D in [1e9,1.6e10,4e10,1e11,2e11]: print(f"gain q=0.6 at D={D:.0e}: P={gP(D):.4f} X={gX(D):.4f}")
# shape in q from fresh D=4e9: s(q)=gain(q)/gain(0.6)
s={0:0.0,0.3:(4.4665-4.4384)/G[1],0.6:1.0,0.85:(4.4665-4.3236)/G[1]}
s[0.5]=s[0.3]+(s[0.6]-s[0.3])*(0.5-0.3)/0.3
print("shape s(q)",s)
def loss(N,D,q,sub,R,g):
    U=sub*(1-q); de=Drep(D,U,R); return base(N,de)-s[q]*g(de)
# checks vs repetition x filter runs
for R in [6.5,5.3]:
  for g,name in [(gP,'P'),(gX,'X')]:
    print(R,name,"run12 pred",round(loss(5e7,4e9,0.6,1e9,R,g),4),"obs 4.5425 | nb q0.6",round(loss(5e7,8e9,0.6,2.5e9,R,g),4),"obs 4.3047 | nb q0.3",round(loss(5e7,8e9,0.3,2.5e9,R,g),4),"obs 4.3168 | nb q0",round(loss(5e7,8e9,0,2.5e9,R,g),4),"obs 4.3094")
print("=== production answers")
Np,Dp=1e9,2e11
for R in [6.5,5.3]:
  for g,name in [(gP,'P'),(gX,'X')]:
    q1=loss(Np,Dp,0,4e10,R,g)-base(Np,Dp)
    q2=loss(Np,Dp,0.5,4e10,R,g)-loss(Np,Dp,0,4e10,R,g)
    q3={q:round(loss(Np,Dp,q,4e10,R,g),4) for q in [0,0.3,0.6,0.85]}
    q4=base(Np,1e11)-loss(Np,1e11,0.6,2e11,R,g)
    q5=[loss(Np,Dp,0.5,sub,R,g)-loss(Np,Dp,0,sub,R,g) for sub in [2e10,4e10,1e11]]
    q6=[loss(Np,Dp,0.6,sub,R,g)-loss(Np,Dp,0,sub,R,g) for sub in [2e10,1e11]]
    print(f"R={R} {name}: q1={q1:.4f} q2={q2:.4f} q3={q3} q4gain={q4:.4f} q5={np.round(q5,4)} q6diff={np.round(q6,4)}")
