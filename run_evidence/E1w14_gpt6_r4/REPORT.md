# Recipe change decision

Ship `c1`, `c2`, `c3`, and `c5`. The predicted expected validation MSE is **0.0400684**.

I used all 40 available training runs. First, I screened every combination of c1–c4 with c5 on seed 17. The best screen result was c1+c2+c3+c5 at 0.040107; the next particularly relevant combinations were c1+c2+c3+c4+c5 at 0.045044 and c1+c2+c3+c5+c6 at 0.040104 on that seed. The screen also exposed divergence for several c1/c2 combinations without clipping, while clipping made the high learning rate and momentum combination viable.

I then evaluated the selected combination on nine seeds (17, 29, 43, and 101–106). Its MSEs were 0.040107, 0.040104, 0.040133, 0.039172, 0.039884, 0.039109, 0.042722, 0.038969, and 0.040416, giving a mean of 0.0400684. All nine runs completed without divergence.

For matched follow-up comparisons, adding c4 to the selected recipe was higher on all six shared seeds by an average of 0.006705. Adding c6 was higher on four of five shared seeds and increased the paired mean by 0.000825. The teammate's single-change results also support retaining c5 and dropping c4; the broader combination runs show that c1, c2, and c3 work well together when clipping and weight decay are present.
