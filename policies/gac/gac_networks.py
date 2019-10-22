# import external dependencies
import tensorflow as tf

# import internal dependencies
import policies.gac.gac_helpers.helpers as helpers

"""
File Description:

This file is meant to contain all networks pertinent to the GAC algorithm. This includes the
Stochastic actors (AIQN and IQN), the value network, and the critic network. Note that, while they
maintain the same architecture, it is advisable that the value and critic netwok be DIFFERENT
classes, if for no other reason than legibility.
"""


class CosineBasisLinear(tf.Module):
    def __init__(self, n_basis_functions, out_size, activation = None):
        """
        Parametrize the embeding function using Fourier series up to n_basis_functions terms
        Class Args:
            n_basis_functions (int): the number of basis functions
            out_size (int): the dimensionality of embedding
            activation: activation function before output
        """
        super(CosineBasisLinear, self).__init__()
        self.act_linear = tf.keras.layers.Dense(
            out_size, activation = activation,
            input_shape = (n_basis_functions,)
        )
        self.n_basis_functions = n_basis_functions
        self.out_size = out_size

    def __call__(self, x):
        """
        Args:
            x: tensor (batch_size_1, batch_size_2): one of the action components. batch_size_1
            and batch_size_2 correspond to the batch size of states (N) and actions per state (K)
        Return:
            out: tensor (batch_size_1, batch_size_2, out_size): the embedding vector phi(x).
        """
        batch_size = x.shape[0]
        h = helpers.cosine_basis_functions(x, self.n_basis_functions)
        out = self.act_linear(h)
        out = tf.reshape(out, (batch_size, -1, self.out_size))
        return out


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
            rnn_input = tf.concat([state_embedding, action_embedding], axis=2)
            # Note that the RNN states encode the function approximation for the conditional
            # probability of the ordered sequence of vectors in d dimension space. Effectively,
            # the researchers claim that each variable in the d dimension vector are autocorrelated.
            gru_out, hidden_state = self.rnn(rnn_input, hidden_state)

            # batch x 400
            hadamard_product = tf.squeeze(gru_out, 1) * noise_embedding[:, idx, :]
            action = self.dense_layer_2(self.dense_layer_1(hadamard_product))
            action_list.append(action)

        actions = tf.squeeze(tf.stack(action_list, axis=1), -1)
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
        state_embedding = tf.expand_dims(tf.nn.leaky_relu(self.state_embedding(state)), 1)
        # batch x action dim x 400
        shifted_actions = tf.zeros_like(actions)
        shifted_actions[:, 1:] = actions[:, :-1]
        provided_action_embedding = self.action_embedding(shifted_actions)

        rnn_input = tf.concat([state_embedding, provided_action_embedding], axis=2)
        gru_out, _ = self.rnn(rnn_input)

        # batch x action dim x 400
        noise_embedding = self.noise_embedding(taus)
        # batch x action dim x 400
        # take the element wise product of these vectors
        hadamard_product = gru_out * noise_embedding
        actions = self.dense_layer_2(self.dense_layer_1(hadamard_product))
        # batch x action dim
        return tf.squeeze(actions, -1)


class StochasticActor(tf.Module):
    def __init__(self, num_inputs, action_dim, n_basis_functions):
        """
        the stochasitc action generator.

        Class Args:
            num_inputs (int): number of inputs used for state embedding
            action_dim (int): the dimensionality of the action vector
            n_basis_functions (int): the number of basis functions
        """
        super(StochasticActor, self).__init__()
        hidden_size = int(400 / action_dim)
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.l1 = tf.keras.layers.Dense(
            self.hidden_size * self.action_dim,
            activation=tf.keras.layers.LeakyReLU(alpha=0.01),
            input_shape = (num_inputs,)
        )
        self.phi = CosineBasisLinear(
            n_basis_functions,
            self.hidden_size,
            activation= tf.keras.layers.LeakyReLU(alpha=0.01)
        )
        self.l2 = tf.keras.layers.Dense(
            200,
            activation= tf.keras.layers.LeakyReLU(alpha=0.01),
            input_shape = (self.hidden_size * self.action_dim,)
        )
        self.l3 = tf.keras.layers.Dense(
            self.action_dim,
            activation= tf.nn.tanh,
            input_shape = (200,)
        )

    def __call__(self, state, taus, actions):
        """
        TODO still not sure about the tensor size.
        Args:
            state: tensor (batch_size, num_inputs)
            taus:
            actions:
        Return:
            next_actions:
        """
        state_embedding = self.l1(state)
        noise_embedding = self.phi(taus)
        noise_embedding = tf.reshape(noise_embedding, (-1, self.hidden_size * self.action_dim))
        hadamard_product = state_embedding * noise_embedding
        l2 = self.l2(hadamard_product)
        next_actions = self.l3(l2)

        return next_actions


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
        self.input = tf.placeholder(tf.float32, [num_input])
        self.num_networks = num_networks
        self.q1 = self.build()
        if self.num_networks == 2:
            self.q2 = self.build()
        elif self.num_networks > 2 or self.num_networks < 1:
            raise NotImplementedError
        # create a session for execution of the graph
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def build(self):
    # A helper function for building the graph
        out1 = tf.keras.layers.dense(self.input, units=400, activation=tf.nn.leaky_relu)
        out2 = tf.keras.layers.dense(out1, units=300, activation=tf.nn.leaky_relu)
        return tf.keras.layers.dense(out2, units=1)

    def __call__(self, x):
    # This function returns the value of the forward path given input x
        if self.num_networks == 1:
            return self.session.run(self.q1, feed_dict={self.input:x})
        else:
            return self.session.run([self.q1, self.q2], feed_dict={self.input:x})

class Value(tf.keras.Model):
    def __init__(self, num_inputs, num_networks=1):
        super(Value, self).__init__()
