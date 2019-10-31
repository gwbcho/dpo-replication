from policies.gac.networks import Value, Critic, AutoRegressiveStochasticActor as Actor
from policies.policy_helpers.helper_classes import ActionSampler, Transition
import tensorflow as tf

actor = Actor(3, 4, 5)
critic1 = Critic(3, 4)
critic2 = Critic(3, 4)
value = Value(3)
sampler = ActionSampler(4)

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

train_history = value.train(transitions, sampler, actor, critic1, critic2, 2)
print(train_history.history)
