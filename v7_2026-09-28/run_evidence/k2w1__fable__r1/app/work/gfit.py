import numpy as np, json, sys; from nm import fit; from load import load
DOM=['web','code','math','papers']; EV=['general','code','math']
U={'web':3e12,'code':4e10,'math':4e9,'papers':2e10}
def deff_tokens(D,mix,pool,Rstar):
    # returns per-domain effective tokens
    out=[]
    for i,d in enumerate(DOM):
        seen=D*mix[d]; u=(pool or U)[d]
        if seen<=u: out.append(seen); continue
        ep=seen/u; RD=ep-1; r=Rstar[i]
        out.append(u*(1+r*(1-np.exp(-RD/r))))
    return np.array(out)
def unpack(p):
    P={}
    k=0
    for e in EV:
        E,lA,al,lB,be,t1,t2,t3=p[k:k+8]; k+=8
        T=np.ones(4); 
        # reference domain =1: general->web, code->code, math->math
        ref={'general':0,'code':1,'math':2}[e]
        others=[i for i in range(4) if i!=ref]
        for i,lt in zip(others,[t1,t2,t3]): T[i]=np.exp(lt)
        P[e]=(E,np.exp(lA),al,np.exp(lB),be,T)
    P['R']=np.exp(p[k:k+4]); 
    return P
def predict(P,N,D,mix,pool=None):
    dt=deff_tokens(D,mix,pool,P['R'])
    out={}
    for e in EV:
        E,A,al,B,be,T=P[e]
        out[e]=E+A*N**(-al)+B*(T@dt)**(-be)
    out['composite']=0.34*out['general']+0.33*out['code']+0.33*out['math']
    return out
def fitall(rows,p0=None):
    def obj(p):
        P=unpack(p); s=0
        for r in rows:
            pr=predict(P,r['N'],r['D'],r['mix'],r['pool'])
            for e in EV: s+=(pr[e]-r['loss'][e])**2
        return s
    if p0 is None:
        p0=[]
        for e in EV: p0+= [1.0,np.log(150),0.3,np.log(600),0.3,-1.5,-1.5,-1.5]
        p0+=[0,0,0,0]
    p,v=fit(obj,np.array(p0),restarts=6,step=0.2,iters=20000)
    return p,v
if __name__=="__main__":
    rows=load()
    p0=None
    try: p0=json.load(open('params.json'))
    except: pass
    p,v=fitall(rows,p0)
    json.dump(list(p),open('params.json','w'))
    P=unpack(p); n=len(rows)*3
    print('rmse',np.sqrt(v/n))
    for e in EV:
        E,A,al,B,be,T=P[e]; print(e,'E=%.3f A=%.1f al=%.3f B=%.1f be=%.3f T='%(E,A,al,B,be),np.round(T,4))
    print('Rstar',np.round(P['R'],3))
    for r in rows:
        pr=predict(P,r['N'],r['D'],r['mix'],r['pool'])
        print('%.0e %.0e'%(r['N'],r['D']),[r['mix'][d] for d in DOM],'pool' if r['pool'] else '    ',' '.join('%s %.3f/%.3f'%(e[:1],r['loss'][e],pr[e]) for e in EV))
