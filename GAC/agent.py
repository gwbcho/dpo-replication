# import external dependencies
import numpy as np
import tensorflow as tf

# internal dependencies
import utils.utils as utils

# import local dependencies
import GAC.networks as networks
import GAC.helpers as helpers


"""
File Description:

Class: Generative Actor Critic (GAC) agent.
"""


class GACAgent:
    """
    GAC agent. 
    Action is alway from -1 to 1 in each dimension.
    Will not do normalization.

    """

    def __init__(self, state_dim, action_dim, 
                autoregressive=False, 
                gamma = 0.99, 
                soft_update_rate = 0.01,
                replay_size = 10000, 
                num_basis_functions=64, 
                target_mode='linear'):

        
    def 