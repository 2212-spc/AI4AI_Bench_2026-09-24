from analyze import *

for wt in [0,.25,.5,1]:
 for kind in ['piece','quartic','cubic','quad','poscubic']:
  npn={'quad':1,'quartic':3}.get(kind,2)
  weights=(N/1e8)**wt
  def pred(p):
   z=x-(p[0]-p[1]*u-p[2]*v)
   base=p[3+npn]+p[4+npn]*np.exp(-p[6+npn]*u)+p[5+npn]*np.exp(-p[7+npn]*v)
   return base+penalty(z,p[3:3+npn],kind)
  init=[-5.45,.14,.2]+[.06,.06,.002][:npn]+[1.7,1.2,1.4,.35,.3]
  p,c=lm(lambda p:(pred(p)-y)*weights,init,500)
  npp=p[3+npn:];lr1=np.exp(p[0]-p[1]*np.log(10)-p[2]*np.log(10));lr2=np.exp(p[0]-p[1]*np.log(10)-p[2]*np.log(500));loss=npp[0]+npp[1]*10**-npp[3]+npp[2]*500**-npp[4]
  print(wt,kind,'rms',round(np.sqrt(c/len(y)),5),'p',np.round(p,6))
  print('ANS',np.log10(lr1),np.log10(lr2),penalty(np.log(.001453/lr2),p[3:3+npn],kind),loss)
  if wt==.5 and kind=='piece':
   print('residuals',np.round(y-pred(p),4))
   np.savez('/app/surface_fit.npz',p=p)
