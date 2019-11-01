import random
import numpy as np
from collections import namedtuple

import tensorflow as tf



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


class ReplayBuffer:
    '''
    A simple FIFO experience replay buffer.
    Adapted from
    https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py (with The MIT License)
    keep the shape in __init__ in mind.
    '''

    def __init__(self, obs_dim, act_dim, size):
        self.transitions = namedtuple('transition', ['s', 'a', 'r', 'sp', 'it'])
        # (this_state, this_action, this_reward, next_state, this_is_terminal)

        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        self.transitions.s = self.obs1_buf[idxs]
        self.transitions.a = self.acts_buf[idxs]
        self.transitions.r = self.rews_buf[idxs]
        self.transitions.sp = self.obs2_buf[idxs]
        self.transitions.it = self.done_buf[idxs]
        return self.transitions



def update(target, source, tau):
    """
    update function.
    when tau = 1, then it's just assignment, i.e. hard update
    Args:
        target (tf.Variable): Variable containing target information
        source (tf.Variable): Variable containing source information
    """
    for target_param, param in zip(target.trainable_parameters, source.trainable_parameters):
        target_param.assign(target_param * (1.0 - tau) + param * tau)



