import csv, glob, numpy as np
hist=list(csv.DictReader(open('/app/data/history.csv')))
base=np.zeros(24); 
R=np.zeros((14,24))
for r in hist: R[int(r['day'])-1,int(r['hour'])]=float(r['requests'])
base=R.mean(0)
def dayfactors():
    fs={}
    for d in range(14): fs[d+1]=(R[d]/base).mean()
    for f in sorted(glob.glob('/app/data/abtest_*.csv')):
        for r in csv.DictReader(open(f)):
            d=int(r['day']); h=int(r['hour']); n=float(r['requests_A'])+float(r['requests_B'])
            fs.setdefault(d,[]); 
            if isinstance(fs[d],list): fs[d].append(n/base[h])
    return {d:(np.mean(v) if isinstance(v,list) else v) for d,v in fs.items()}
def Bparams():
    """pooled per-hour B gen time & rating, weighted by requests"""
    g=np.zeros(24); gw=np.zeros(24); rt=np.zeros(24); ga=0; gaw=0; ra=0; raw=0
    for f in sorted(glob.glob('/app/data/abtest_*.csv')):
        for r in csv.DictReader(open(f)):
            h=int(r['hour']); nB=float(r['requests_B'])*(1-float(r['abandon_rate']))
            if nB>0:
                g[h]+=nB*float(r['gen_time_B_s']); rt[h]+=nB*float(r['rating_B']); gw[h]+=nB
            nA=float(r['requests_A'])*(1-float(r['abandon_rate']))
            if nA>0:
                ga+=nA*float(r['gen_time_A_s']); ra+=nA*float(r['rating_A']); gaw+=nA
    for r in hist:
        n=float(r['requests'])*(1-float(r['abandon_rate']))
        ga+=n*float(r['mean_gen_time_s']); ra+=n*float(r['mean_rating']); gaw+=n
    return g/gw, rt/gw, gw, ga/gaw, ra/gaw
if __name__=='__main__':
    fs=dayfactors(); print({d:round(v,3) for d,v in fs.items()})
    v=np.array(list(fs.values())); print('mean',v.mean(),'sd',v.std(ddof=1), 'logsd', np.log(v).std(ddof=1))
    g,rt,w,gA,rA=Bparams()
    for h in range(24): print(h, round(g[h],3), round(rt[h],4), int(w[h]))
    print('A', gA, rA)
    day=slice(9,18); night=[h for h in range(24) if not 9<=h<=17]
    print('B day gen', (g[day]*w[day]).sum()/w[day].sum(), 'rating', (rt[day]*w[day]).sum()/w[day].sum())
    print('B night gen', (g[night]*w[night]).sum()/w[night].sum(), 'rating', (rt[night]*w[night]).sum()/w[night].sum())
