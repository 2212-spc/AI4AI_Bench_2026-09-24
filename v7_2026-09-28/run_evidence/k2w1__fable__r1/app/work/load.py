import json
def load():
    rows=[]
    for line in open('/app/lab_log.jsonl'):
        r=json.loads(line); a=r['args']; res=r['result']
        rows.append(dict(N=a['N'],D=a['D'],mix=a['mix'],pool=a.get('pool'),loss=res['eval_loss'],comp=res['composite'],epochs=res['epochs']))
    return rows
