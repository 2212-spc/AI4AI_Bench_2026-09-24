Behavioural details of `pack_ref.py` that the fast implementation must preserve:
1. first fit in **bin creation order**, not best fit (leftmost open bin whose remaining capacity >= L);
2. a sequence with L > capacity consumes a bin id and that bin is never open;
3. the closure test is `remaining < close_below`, strictly;
4. the memory guard fires only when the number of open bins is strictly greater than max_open, and it
   closes the **fullest** open bin (smallest remaining), ties broken by smallest bin id;
5. rule order after a placement is: closure, then memory guard, then latency guard;
6. the latency guard uses the index of the sequence in the input (oversized sequences advance that index
   but are skipped by the guard because the loop body returns early for them);
7. the latency guard needs at least two open bins and closes the oldest one;
8. zero-length sequences take the leftmost open bin and open a new bin when none is open.
