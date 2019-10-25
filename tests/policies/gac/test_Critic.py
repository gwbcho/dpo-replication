from policies.gac.gac_networks import Value, Critic
from policies.policy_helpers.helpers import Transition
import tensorflow as tf

critic = Critic(3+4, 2)
value = Value(3, 1)

transitions = Transition(
        tf.convert_to_tensor(
        [
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.],
        ]
        ),
        tf.convert_to_tensor(
        [
            [0., 0., 0., 0.],
            [0.1, 0.2, 0.3, 0.1],
            [0.4, 0.5, 0.6, 0.9],
        ]
        ),
        tf.convert_to_tensor(
        [
            1., 2., 3.
        ]
        ),
        tf.convert_to_tensor(
        [
            [4., 5., 6.],
            [7., 8., 9.],
            [1., 2., 3.],
        ]
        ),
        tf.convert_to_tensor(
        [
            False, False, False
        ]
        ),
    )

history1, history2 = critic.train(transitions, value, 0.99)
print(history1.history, history2.history)
