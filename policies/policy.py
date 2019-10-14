# external dependencies
import tensorflow as tf

# internal dependencies
import policies.policy_helpers.helpers as helpers

"""
File Description:

This file contains the Policy class which is the parent for all policies in this project. It's meant
to contain bare bones functionality to be modified after it's instantiation.
"""


class Policy:
    """
    Policy function for bare bones RL
    """

    def __init__(self, gamma, tau, num_inputs, action_space, replay_size, normalize_obs=True,
                 normalize_returns=False):
    # TODO
