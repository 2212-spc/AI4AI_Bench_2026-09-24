# Recipe change decision

Ship `c1`, `c2`, `c3`, `c5`, and `c6`. The predicted expected validation MSE is **0.0393**.

I used all 80 available training runs. The main evidence was a matched 17-seed comparison of the two strongest candidates:

| Recipe | Runs | Mean validation MSE | Sample SD |
|---|---:|---:|---:|
| c1+c2+c3+c5 | 17 | 0.04010 | 0.00136 |
| c1+c2+c3+c5+c6 | 17 | 0.03931 | 0.00086 |

The changes c1 and c2 were harmful alone and frequently diverged in un-clipped combinations, but the broader subset screen showed that c3 makes their combination stable and effective. The best single-seed screen result was c1+c2+c3+c5 at 0.038744 and the corresponding five-change result was 0.038781; the additional matched seeds favored adding c6 overall. c4 was excluded because its combinations were generally worse and some interactions with c1 or c2 diverged or produced much higher error. The final prediction is the mean of the 17 observed validation MSEs for the shipped subset, rounded to four decimal places.
