# Model B launch memo

Recommend the schedule in `rollout.conf`: full B routing from 00:00 through 07:59 and from 22:00 through 23:59, with 30% B routing from 08:00 through 21:59. This respects the daytime guardrail and keeps the high traffic period below the point where B's longer generation time drives the shared pool into the 24-replica quota.

B raises answer ratings by about four to five hundredths, while its generation time is about 4.2 seconds overnight and 5.7 seconds during the daytime, compared with about 2.47 seconds for A. The queue model includes abandonment as a constant-hazard process and charges the observed 0.0098 score units per queue second. The schedule therefore takes all of the available B quality gain overnight and uses the maximum permitted daytime share where capacity remains valuable.

Q1: [-0.0053, +0.0007]
Q2: [+0.0137, +0.0177]
