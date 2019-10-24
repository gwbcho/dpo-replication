# system dependencies
import os
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

"""
File Description:

Helper functions for the policy class/file and may be used in other classes as well.
"""

class Replay():
    """
    Holds Transition tuple (s,a,r,sp,is_terminal) and provides random sampling of them.
    """

    def __init__(self, size):
        self.size = size    # maximum number of items in this replay buffer
        self.buffer = []
        self.end = 0    # end is the index of last item

    def append(self, transition):
        if len(self) < self.size:
            # buffer is not full yet
            self.buffer.append(transition)
            self.end = len(self)-1
        else:
            # buffer is full, overwrite the oldest one
            self.end = (self.end+1) % self.size
            self.buffer[self.end] = transition

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

    def get_actions(self, actor, state, actions):
        """
        Actions are obtained from the actor network.
        """
        if state.shape.rank > 1:
            batch_size = state.shape[0]
        else:
            batch_size = 1
        return actor(
            state,
            tf.random.uniform((batch_size, self.dim), minval=0.0, maxval=1.0),
            actions
        )
