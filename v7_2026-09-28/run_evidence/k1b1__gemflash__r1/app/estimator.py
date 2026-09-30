def estimate(env):
    keys = [k for k in dir(env) if not k.startswith('_')]
    mix = env.production_mix
    raise ValueError(f"keys={keys}, mix={mix}")
