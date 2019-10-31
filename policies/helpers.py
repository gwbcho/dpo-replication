import random
from collections import namedtuple

import tensorflow as tf
"""
A transition is made of state, action, reward, state', is_terminal.
is_terminal flags if state is a terminal state, in which case this
transition should not be counted when calculating reward.
"""
Transition = namedtuple(
    'Transition',
    ['s', 'a', 'r', 'sp', 'is_terminal']
)


class ReplayMemory(object):
    """
    Holds Transition tuple (s,a,r,sp,is_terminal) and provides random sampling of them.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, n):
        return random.sample(self.buffer, n)

    def __len__(self):
        return len(self.buffer)


class ActionSampler():
    """
    Sampling actions from a given actor by feeding samples from a uniform distribution into the
    actor network.
    """

    def __init__(self, action_dim):
        self.dim = action_dim

    def get_actions(self, actor, states, actions=None):
        """
        Actions are obtained from the actor network.
        """
        if states.shape.rank > 1:
            batch_size = states.shape[0]
        else:
            batch_size = 1
        return actor(
            states,
            tf.random.uniform((batch_size, self.dim), minval=0.0, maxval=1.0),
            actions
        )

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
