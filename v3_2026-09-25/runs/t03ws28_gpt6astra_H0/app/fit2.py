exec(open('/app/analyze.py').read().split("if __name__")[0])
# q0 only model
ii=(x[:,2]==0)
for maskname,mask in [('q0',ii),('q05',(x[:,2]==0)|(x[:,2]==.5)),('all',(x[:,2]<=.5))]:
 xx=x[mask]; yy=y[mask]
 def fn(p): return predict(p,xx,rep='exp',quality='linear')
 p,rmse=fit(fn,[2,.8,.38,.36,.37,5,1],yy)
 print(maskname,p,rmse)
 def pp(z): return predict(p,z,rep='exp',quality='linear')
 for S in [4e10,2e10,1e11]:
  z=np.array([[1e9,2e11,.5,S],[1e9,2e11,0,S]])
  print(S, pp(z)[0]-pp(z)[1])
 # custom q .5 fit with base fixed? 
