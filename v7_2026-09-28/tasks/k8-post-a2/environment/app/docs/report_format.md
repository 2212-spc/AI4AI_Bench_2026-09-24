# report.json

    {
      "cause": "capacity",            // one of ['capacity', 'mixture', 'lr_scale']
      "fix": {"capacity": 0.74,       // the knob you blamed, moved as far as the release note allows
              "lr_scale": 1.2}        // and whatever else you had to re-tune around it
    }

Knobs you leave out keep their **release** value. The grader clamps `fix` to the same release constraint the
lab enforces, so asking for a value beyond the cap silently gets you the cap.
