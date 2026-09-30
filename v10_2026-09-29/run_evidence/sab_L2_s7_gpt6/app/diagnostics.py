exec(open('/app/analyze.py').read().split('# resample arrivals')[0])
T=pd.concat([H.assign(fraction_B=0),E],ignore_index=True)
# determine latent arrival rate from utilization; inversion monotonic
m=Ag+T.fraction_B.to_numpy()*(np.where(T.hour.between(9,17),params[1][1],params[0][1])-Ag)
c=T.replicas.to_numpy(); util=T.utilization.to_numpy()
lo=T.requests.to_numpy()/3600*.6;hi=T.requests.to_numpy()/3600*1.6
for _ in range(35):
 mid=(lo+hi)/2; w=queue(mid,m,c); u=mid*m*(1-th*w)/c
 lo=np.where(u<util,mid,lo);hi=np.where(u>=util,mid,hi)
l=(lo+hi)/2
pred=queue(l,m,c)
print('model wait residual inferred rates',np.mean(np.log(T.mean_queue_wait_s/pred)),np.std(np.log(T.mean_queue_wait_s/pred)))
print('poisson residual',np.mean((T.requests-3600*l)/np.sqrt(3600*l)),np.std((T.requests-3600*l)/np.sqrt(3600*l)))
print('replica matches',np.mean(np.clip(np.ceil(l*m/.75),4,16)==c))
for name,g in [('A',T[T.fraction_B==0]),('mix',T[T.fraction_B==.3]),('B',T[T.fraction_B==1])]:
 i=g.index;print(name,'wait obs/pred',np.average(g.mean_queue_wait_s,weights=g.requests)/np.average(pred[i],weights=g.requests))
T['inferred_rate']=l;T.to_csv('/app/data/inferred.csv',index=False)
