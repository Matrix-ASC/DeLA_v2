# Specialized Position Aware Max Pooling

Please read SPSE first.

## Prior

So we want feature max-pooling to be aware of spatial clues: f_out = pool{g(f_i, s_i)}.

What priors might be good?

1. To suppress noise, a distant point should not get too much attention.
2. To separate raw features from latent ones, multiplication should be better than addition.

These are my personal considerations, and are not tested.

## Solution

The most intuitive solution is to pool f * e^(-r^2). Of course others can be tried like f / (1+r^2), f * e^(-r^2) * (1+r^2), ...

It's not clear what's the key here in this feature-spatial fusion. But in my tests on S3DIS I didn't find notable gain. f * e^(-r^2) seems to work best.

Since edge (f_i - f_center) is actually pooled, the smallest output is 0. (f_center - f_center is always here)

If feature is pooled, there exists inconsistency when f_i < 0, distant points tend to be pooled in this case, contrary to when f_i > 0.

## Simplification

To reduce computation, only radius is considered here: s = r. Didn't test ellipsoid.

Besides, 4 channels share one r: f_1 - f_4 share r_1. 4 is chosen because in CUDA, 4 floats (128 bytes) are fetched together. Didn't test 8 as bfloat16 kernels are not implemented.

Its efficiency is very close to max-pooling. Its performance is very close to its full version.