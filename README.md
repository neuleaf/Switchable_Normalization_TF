# Tensorflow Switchable Normalization

The implementation of SN can synchronize `moving_mean` and `moving_variance` across multi GPUs.

## usage
 `from switch_norm import switch_norm`

if you use `tf.slim`, just replace `slim.batch_norm` with `switch_norm`. They share alomost the same parameters.


## Note: compatible with TF-1.9.0
