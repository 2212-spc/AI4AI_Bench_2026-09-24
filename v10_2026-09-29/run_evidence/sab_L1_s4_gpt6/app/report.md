# Model B launch memo

Recommendation: launch the following hour-of-day schedule. It sends Model B at 100% in the quiet overnight and morning hours, then tapers B through the afternoon peak so the shared pool stays below the 16-replica quota. The schedule routes about 71% of requests to B in a typical week.

Q1: [-0.0250, -0.0195]
Q2: [+0.0180, +0.0215]

The all-A telemetry gives a production baseline of about 0.6251 score units per request. Model B raises served-answer ratings by about 0.025 points, but its measured generation time is about 7.24 seconds versus 3.07 seconds for A. With exponential patience, the experiments estimate a patience hazard of 0.0218 per second, or about 46 seconds mean patience. The resulting queue model predicts that a full B launch overloads the afternoon pool: the 16-replica ceiling makes abandonment and waiting costs larger than the rating improvement. It predicts a value of about -0.0226 for 100% B in every hour.

The chosen schedule uses 100% B in hours 00–09 and 21–23. It uses 87%, 70%, 55%, 45%, 38%, 35%, 36%, 41%, 52%, 69%, and 94% in hours 10–20 respectively. Those fractions align with the replica-count steps and preserve capacity where the traffic curve is highest. The estimated launch value is +0.0196 score units per request.

Validation included a five-day 100% B capacity drill during hours 01–05 capped at four replicas. The observed abandonment rate was 38.97% and mean queue wait was 17.85 seconds; the fitted queue model predicted 38.87% and 17.87 seconds. A one-percent increase in demand changed the schedule estimate by less than 0.0003 score units per request, while it made the all-B estimate more negative by about 0.0031.

Operational rollout: apply the 24 lines in `rollout.conf` as the repeating daily routing schedule, and monitor abandonment and queue wait around hours 10–20. The schedule is designed for the fixed 16-replica quota; an increase in demand or a change in Model B generation time should trigger a fresh capacity review.
