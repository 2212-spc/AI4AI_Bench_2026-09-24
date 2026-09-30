import contextlib,io,math
with contextlib.redirect_stdout(io.StringIO()):
 import replay
import policy
for mode in range(6):
 def priority(self):
  n=self.n;c=self.confidence
  r=[.07+.35*math.exp(-n/4)+.15*c, .4/(1+n/4)+.15*c, .4/(1+n/4)+.3*c, .4/(1+n/4)+.05*c, .07+.35*math.exp(-n/4)+.15*c, .4/(1+n/4)+.15*c][mode]
  if mode==4 and n>=30: r*=.01
  if mode==5 and n>=40: r*=.01
  return (1-c)*r
 policy._Question.priority=priority
 acc=[]; ns=[]
 for seed in range(30):
  e=replay.Env(seed);policy.run(e)
  acc.append(sum(e.out[q['id']]==q['correct_answer'] for q in replay.qs)/len(replay.qs));ns.append(max(e.used.values()))
 print(mode,sum(acc)/len(acc),min(acc),sum(ns)/len(ns))
