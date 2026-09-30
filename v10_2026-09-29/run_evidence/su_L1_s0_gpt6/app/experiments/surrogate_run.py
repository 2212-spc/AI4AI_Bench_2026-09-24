import sys
sys.path.insert(0,'/app/repo')
import train as m
import numpy as np,json,time,resource
D=np.load('/app/experiments/synthetic_corpus.npz');vx=np.load('/app/experiments/synthetic_test_X.npy');vy=np.load('/app/experiments/synthetic_gold.npy')
cfg=json.load(open('/app/repo/config.json'))
which=int(sys.argv[1]); power=cfg['decay_powers'][which];cfg['decay_powers']=[power]
t=time.process_time();models,norm=m.train(D['X'],D['y'],20000,cfg,0+1009*which)
p=[]
for i in range(0,len(vx),2048):
 z,_=m.forward(models[0],norm(vx[i:i+2048]));p0=np.exp(z-z.max(1,keepdims=True));p.append(p0/p0.sum(1,keepdims=True))
p=np.concatenate(p);np.save('/app/experiments/synth_prob%d.npy'%which,p)
print('member',which,'power',power,'dev',np.mean(p.argmax(1)==vy),'train',np.mean(m.predict(models,norm,D['X'])==D['y']),'cpu',time.process_time()-t,'rss_KB',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
