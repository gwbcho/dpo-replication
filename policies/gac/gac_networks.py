# import external dependencies
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
# import internal dependencies
import gac_helpers.helpers as helpers

"""
File Description:

This file is meant to contain all networks pertinent to the GAC algorithm. This includes the
Stochastic actors (AIQN and IQN), the value network, and the critic network. Note that, while they
maintain the same architecture, it is advisable that the value and critic netwok be DIFFERENT
classes, if for no other reason than legibility.
"""


class CosineBasisLinear(tf.keras.Model):
    def __init__(self, n_basis_functions, out_size):
        super(CosineBasisLinear, self).__init__()
        pass

    def forward(self, x):
        pass


class AutoRegressiveStochasticActor(tf.keras.Model):
    def __init__(self, num_inputs, action_dim, n_basis_functions):
        """
        the autoregressive stochastic actor is an implicit quantile network used to sample from a
        distribution over optimal actions. The model maintains it's autoregressive quality due to
        the recurrent network used.

        Class Args:
            num_inputs (int): number of inputs used for state embedding
            action_dim (int): the dimensionality of the action vector
            n_basis_functions (int): the number of basis functions
        """
        super(AutoRegressiveStochasticActor, self).__init__()

    def forward(self, state, taus, actions=None):
        pass

    def supervised_forward(self, state, taus, actions):
        pass


class StochasticActor(tf.keras.Model):
    def __init__(self, num_inputs, action_dim, n_basis_functions):
        super(StochasticActor, self).__init__()


    def forward(self, state, taus, actions):
        pass


class Critic(tf.keras.Model):
    '''
    The Critic class create one or two critic networks, which take states as input and return 
    the value of those states. The critic has two hidden layers and an output layer with size
    400, 300, and 1. All are fully connected layers.
    
    Class Args:
    num_inputs (int): number of states
    num_networks (int): number of critc networks need to be created
    '''
    def __init__(self, num_inputs, num_networks=1):
        super(Critic, self).__init__()        
        self.num_networks = num_networks
        self.num_inputs = num_inputs
        self.q1 = self.build()
        if self.num_networks == 2:
            self.q2 = self.build()
        elif self.num_networks > 2 or self.num_networks < 1:
            raise NotImplementedError
        
    def build(self):
    # A helper function for building the graph
        model = Sequential()
        model.add(Dense(units=400, input_shape=(self.num_inputs,), activation=tf.nn.leaky_relu))
        model.add(Dense(units=300, activation=tf.nn.leaky_relu))
        model.add(Dense(units=1))
        return model
    
    def __call__(self, x):
    # This function returns the value of the forward path given input x
        if self.num_networks == 1:
            return self.q1(x)
        else:
            return self.q1(x), self.q2(x)

class Value(tf.keras.Model):
    def __init__(self, num_inputs, num_networks=1):
        super(Value, self).__init__()
