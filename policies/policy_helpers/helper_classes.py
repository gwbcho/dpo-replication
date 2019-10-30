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


class HelperPolicyClass:

    def policy(self, actor, state):
        raise NotImplementedError

    def update_critic(self, state_batch, action_batch, reward_batch, mask_batch, next_state_batch):
        raise NotImplementedError

    def soft_update(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
