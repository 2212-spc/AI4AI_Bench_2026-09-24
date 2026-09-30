import json, math
# Estimates from power-law boundary constraints across notebook and new runs.
edge_prod = 0.0007365846245714808
# At maximum accessible N and minimum warmup, the off boundary is bracketed
# by a stable 0.002 run and a diverged 0.00265 run.
edge_min_lab = math.sqrt(0.002 * 0.00265)
factor_min = max(2.0, 0.05 / edge_min_lab)
answers = {
    'q1': {'lo': math.log10(edge_prod), 'hi': math.log10(edge_prod)},
    'q2': {'lo': math.log10(edge_prod * factor_min),
           'hi': math.log10(edge_prod * 60)},
    'q3': {'verdict': 'supported'},
    'q4': {'verdict': 'supported'},
    'q5': {'verdict': 'undetermined'},
    'q6': {'choice': 'C'},
    'q7': {'verdict': 'refuted'},
}
with open('/app/answers.json', 'w') as f:
    json.dump(answers, f, indent=2, allow_nan=False)
    f.write('\n')
print(json.dumps(answers, indent=2))
