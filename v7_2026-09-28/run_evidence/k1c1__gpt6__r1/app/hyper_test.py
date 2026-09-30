import math
counts=[(10,0),(16,1),(10,0),(10,1),(10,0),(10,0),(16,1),(10,0),(10,0),(10,2),(23,2),(10,0),(10,0),(10,0),(33,2),(10,0)]
states=[]
for i in range(30):
 m=math.exp(math.log(.001)+(i+.5)/30*math.log(400))
 for j in range(25):
  k=math.exp(math.log(1)+(j+.5)/25*math.log(500))
  a=m*k;b=(1-m)*k
  lp=math.log(m)+18*math.log(1-m)
  for n,e in counts:lp+=math.lgamma(a+e)-math.lgamma(a)+math.lgamma(b+n-e)-math.lgamma(b)+math.lgamma(k)-math.lgamma(k+n)
  states.append((lp,m,k))
z=max(x[0] for x in states);ps=[math.exp(x[0]-z) for x in states];tot=sum(ps)
print('m,k',*[sum(p*x[i] for x,p in zip(states,ps))/tot for i in [1,2]])
for n,e in counts:print(n,e,sum(p*(x[1]*x[2]+e)/(x[2]+n) for x,p in zip(states,ps))/tot)
