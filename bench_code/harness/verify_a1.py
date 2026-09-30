import sys, json
sys.path.insert(0, '/tmp/bench/gen'); sys.path.insert(0, '/tmp/bench/lab')
import a1_verify_core as V
d = json.load(open(sys.argv[1]))
teacher = V.teacher_from(d['teacher'])
for rd in sys.argv[2:]:
    ok, rep = V.evaluate(rd + '/app', d['spec'] if 'spec' in d else d, teacher, verbose=False)
    print(rd, 'PASS' if ok else 'FAIL', json.dumps(rep)[:700])
