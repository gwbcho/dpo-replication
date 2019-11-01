# system dependencies
import tensorflow as tf


"""
File Description:

Helper functions for the policy class/file and may be used in other classes as well.
"""


def soft_update(target, source, tau):
    """
    Soft update function.

    Args:
        target (tf.Variable): Variable containing target information
        source (tf.Variable): Variable containing source information
    """
    for target_param, param in zip(target.trainable_parameters, source.trainable_parameters):
        target_param.assign(target_param * (1.0 - tau) + param * tau)


def hard_update(target, source):
    """
    Hard update function (converts all values directly instead of using a learning rate)

    Args:
        target (tf.Variable): Variable containing target information
        source (tf.Variable): Variable containing source information (delayed info)
    """
    for target_param, param in zip(target.trainable_parameters, source.trainable_parameters):
        target_param.assign(param)


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
