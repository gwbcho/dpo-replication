import unittest

import tensorflow as tf

import policies.gac.gac_networks as gac_networks


class TestGacNetworks(unittest.TestCase):

    def test_cosine_basis_linear(self):
        n_basis_functions = 64
        out_size = 400
        embedding = gac_networks.CosineBasisLinear(n_basis_functions, out_size)
        batch_size_1 = 10
        batch_size_2 = 10
        # random input
        x = tf.Variable(
            tf.random.normal(
                [batch_size_1, batch_size_2],
                stddev=.1,
                dtype=tf.float32
            )
        )
        # simply test if this is working
        out = embedding(x)
        self.assertEqual(out.shape[0], batch_size_1)
        self.assertEqual(out.shape[1], batch_size_2)
        self.assertEqual(out.shape[2], out_size)

    def test_autoregressive_stochastic_actor_no_action(self):
        batch_size_1 = 1
        num_inputs = 10
        action_dim = 5
        state_dim = 1
        n_basis_functions = 64
        # construct the autoregressive stochasitc actor for testing
        actor = gac_networks.AutoRegressiveStochasticActor(
            num_inputs,
            action_dim,
            n_basis_functions
        )
        state = tf.Variable(
            tf.random.normal(
                [num_inputs, state_dim],
                stddev=.1,
                dtype=tf.float32
            )
        )
        # WHY????
        taus = tf.Variable(
            tf.random.uniform(
                [batch_size_1, action_dim, 1]
            )
        )
        action = actor(state, taus)


if __name__ == '__main__':
    unittest.main()
