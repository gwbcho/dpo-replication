import tensorflow as tf

import policies.gac.gac_helpers as gac_helpers
import policies.gac.gac_networks as gac_networks
import policies.policy as policy
import policies.policy_helpers as policy_helpers


"""
File Description:

This file hosts the Generator class which is a subclass of Policy. The purpose of this class is to
unify the networks constructed for GAC in gac_networks and construct a generator function which
follows an optimal stationary stochastic policy.
"""

class Actor():
    """
    The functionalities needed to interact with GAC are here, except the Neural Network part.
    nn is Neural Net of the actor.
    """

    def __init__(self, nn):
        self.nn = nn


class GenerativeActorCritic():
    """
    Class to construct a generative policy which samples from an optimally constructed
    distribution over continuous action space.
    """


    def __init__(self, 
