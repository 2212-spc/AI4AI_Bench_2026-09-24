# Nightly classifier update

The production run makes 20,000 optimizer updates, while the tuning run made 1,000 updates. Both runs expose the model to the data for 32 passes, but AdamW applies decoupled weight decay on every update. I therefore reduced the tuned weight decay from `2.0` to `0.1`, preserving approximately the same cumulative decay over the training schedule while allowing the full corpus to provide its additional independent examples.

The network, learning rate, batch size, schedule, and loss remain unchanged. I also set the numerical libraries to one thread before importing NumPy so the job follows the single-core production constraint reliably.

I verified that the shipped command line still trains and writes an integer NumPy prediction array, and checked the training path with one-thread execution.
