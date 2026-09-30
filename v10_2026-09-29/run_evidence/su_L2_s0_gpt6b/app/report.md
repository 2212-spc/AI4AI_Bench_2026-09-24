# Nightly classifier

I replaced the raw-feature optimization setup with a data-dependent ZCA whitening transform, which removes the strong feature scale and correlation differences before the MLP sees a batch. The optimizer now uses generalized cross entropy: examples whose crowd label is repeatedly implausible receive a smaller gradient, reducing the effect of the noisy crowd labels. I also use mild feature noise and an exponential moving average of the weights for a more stable final predictor.

The production command line is unchanged. The implementation uses only numpy, computes all transforms from the supplied corpus at runtime, and writes integer class predictions to the requested output file.
