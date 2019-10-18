# import external dependencies
import tensorflow as tf

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


class AutoRegressiveStochasticActor(tf.Module):
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
        # create all necessary class variables
        self.action_dim = action_dim
        self.state_embedding = tf.keras.layers.Dense(
            400,  # as specified by the architecture in the paper and in their code
            input_shape=(num_inputs,),
            activation=tf.nn.leaky_relu
        )
        # use the cosine basis linear classes to "embed" the inputted values to a set dimension
        # this is equivalent to the psi function specified in the Actor diagram
        self.noise_embedding = CosineBasisLinear(n_basis_functions, 400)
        self.action_embedding = CosineBasisLinear(n_basis_functions, 400)

        # construct the GRU to ensure autoregressive qualities of our samples
        self.rnn = tf.keras.layers.GRU(400, batch_first=True)
        # post processing linear layers
        self.dense_layer_1 = tf.keras.layers.Dense(400, activation=tf.nn.leaky_relu)
        # output layer (produces the sample from the implicit quantile function)
        # note the output is between [0, 1]
        self.dense_layer_2 = tf.keras.layers.Dense(1, activation=tf.nn.tanh)

    def __call__(self, state, taus, actions=None):
        """
        Analogous to the traditional call function in most models. This function conducts a single
        forward pass of the AIQN given the state.

        Args:
            state (tf.Variable): state vector containing a state with the format R^num_inputs
            taus (tf.Variable): randomly sampled noise vector for sampling purposes
            actions (tf.Variable): set of previous actions

        Returns:
            actions vector
        """
        if actions is not None:
            # if the actions are defined then we use the supervised forward method which generates
            # actions based on the provided sequence
            return self._supervised_forward(state, taus, actions)
        batch_size = state.shape[0]
        # batch x 1 x 400
        state_embedding = tf.expand_dims(self.state_embedding(state), 1)
        # batch x action dim x 400
        noise_embedding = self.noise_embedding(taus)

        action_list = []

        # allocate memory for the actions
        action = tf.zeros(batch_size, 1)
        hidden_state = None

        # If the prior actions are not provided then we generate the action vector dimension by
        # dimension. Note that the actions are in the domain [0, 1] (Why? I dunno).
        for idx in range(self.action_dim):
            # batch x 1 x 400
            action_embedding = self.action_embedding(tf.reshape(action, (batch_size, 1, 1)))
            rnn_input = tf.concat([state_embedding, action_embedding], dim=2)
            # Note that the RNN states encode the function approximation for the conditional
            # probability of the ordered sequence of vectors in d dimension space. Effectively,
            # the researchers claim that each variable in the d dimension vector are autocorrelated.
            gru_out, hidden_state = self.rnn(rnn_input, hidden_state)

            # batch x 400
            hadamard_product = tf.mul(tf.squeeze(gru_out, 1), noise_embedding[:, idx, :])
            action = self.dense_layer_2(self.dense_layer_1(hadamard_product))
            action_list.append(action)

        actions = tf.squeeze(tf.stack(action_list, dim=1), -1)
        return actions

    def _supervised_forward(self, state, taus, actions):
        """
        Private function to conduct a supervised forward call. This is relying on the assumption
        actions are not independent to each other. With this assumption of "autocorrelation" between
        action dimensions, this function creates a new action vector using prior actions as input.

        Args:
            state (tf.Variable(array)): state vector representation
            taus (tf.Variable(array)): noise vector
            actions (tf.Variable(array)): actions vector (batch x action dim)

        Returns:
            a action vector of size (batch x action dim)
        """
        # batch x action dim x 400
        state_embedding = tf.unsqueeze(tf.nn.leaky_relu(self.state_embedding(state)), 1)
        # batch x action dim x 400
        shifted_actions = tf.zeros_like(actions)
        shifted_actions[:, 1:] = actions[:, :-1]
        provided_action_embedding = self.action_embedding(shifted_actions)

        rnn_input = tf.concat([state_embedding, provided_action_embedding], dim=2)
        gru_out, _ = self.rnn(rnn_input)

        # batch x action dim x 400
        noise_embedding = self.noise_embedding(taus)
        # batch x action dim x 400
        # take the element wise product of these vectors
        hadamard_product = tf.mul(gru_out, noise_embedding)
        actions = self.dense_layer_2(self.dense_layer_1(hadamard_product))
        # batch x action dim
        return tf.squeeze(actions, -1)


class StochasticActor(tf.keras.Model):
    def __init__(self, num_inputs, action_dim, n_basis_functions):
        super(StochasticActor, self).__init__()


    def forward(self, state, taus, actions):
        pass


class Critic(tf.keras.Model):
    def __init__(self, num_inputs, num_networks=1):
        super(Critic, self).__init__()


class Value(tf.keras.Model):
    def __init__(self, num_inputs, num_networks=1):
        super(Value, self).__init__()
