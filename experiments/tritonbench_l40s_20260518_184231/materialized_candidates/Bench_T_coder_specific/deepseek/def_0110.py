def exp_mean(input, dim=None, keepdim=False, dtype=None, out=None):
    import tensorflow as tf
    exp_input = tf.exp(input)
    mean_output = tf.reduce_mean(exp_input, axis=dim, keepdims=keepdim, out=out)
    if dtype is not None:
        mean_output = tf.cast(mean_output, dtype)
    return mean_output
