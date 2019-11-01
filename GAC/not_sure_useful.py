import tensorflow as tf

'''
This is for the code that shows up in their code, but not sure if we need in our implementations.
You can view it as a storage file.
'''


def normalize(x, stats):
    """
    Note: Not sure if this is correct

    Args:
        x (tf.Variable): tensorflow variable to normalize
        stats (utils.RunningMeanStd): Information regarding the mean and variance of data

    Returns:
        normalized value of x
    """
    if stats is None:
        return x
    mean, variance = tf.moments(stats, axes=1)
    return (x - tf.Variable(stats.mean)) / tf.Variable(tf.sqrt(stats.var))


def denormalize(x, stats):
    if stats is None:
        return x
    sigma = tf.sqrt(stats.var)
    return x * sigma + stats.mean