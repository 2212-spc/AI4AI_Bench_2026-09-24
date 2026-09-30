# report.json

    {
      "cause": "<one of ['capacity', 'mixture', 'lr_scale']>",
      "fix": {"<knob>": 0.0,          // the knob you blamed, moved as far as the release note allows
              "<other knob>": 0.0}    // and whatever else you had to re-tune around it
    }

The values above are placeholders, not a worked example - fill in what you measured.

Knobs you leave out keep their **release** value. The grader clamps `fix` to the same release constraint the
lab enforces, so asking for a value beyond the cap silently gets you the cap.
