# Recipe change decision

Ship `c1`, `c2`, `c3`, `c5`, and `c6`.

The decision is based on 40 cluster runs across the proposed combinations and multiple training seeds. The shipped combination was evaluated on eight seeds with validation MSEs 0.042291, 0.040062, 0.041995, 0.042223, 0.040230, 0.042051, 0.041778, and 0.041033, for a sample mean of 0.041458. I therefore estimate the expected validation MSE as **0.04145**.

The main alternatives were weaker in the observed runs. Removing `c6` from the same clipped, decayed recipe averaged 0.043299 over five seeds; removing `c5` averaged 0.042427 over five seeds. Adding the wider hidden layer (`c4`) to all five shipped changes averaged 0.047248 over six seeds. The originally suggested `c1,c2,c4,c6` recipe also diverged on one of its two checked seeds, whereas clipping (`c3`) stabilized the high learning rate and momentum combination. The single change results in the teammate ablation were used as supporting evidence, but combination runs drove the final choice.
