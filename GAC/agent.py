# import external dependencies
import numpy as np
import tensorflow as tf

# import local dependencies
from GAC.networks import StochasticActor, AutoRegressiveStochasticActor,  Critic, Value
from GAC.helpers import ReplayBuffer, update, ActionSampler


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
    def __init__(self, args):
        """
        Agent class to generate a stochastic policy.

        Args:
            args (class):
                Attributes:
                    TODO
        """
        self.args = args
        if args.actor == 'IQN':
            self.actor = StochasticActor(args.state_dim,args.action_dim)
            self.target_actor = StochasticActor(args.state_dim,args.action_dim)
        elif args.actor == 'AIQN':
            self.actor = AutoRegressiveStochasticActor(args.state_dim,args.action_dim)
            self.target_actor = AutoRegressiveStochasticActor(args.state_dim,args.action_dim)

        self.critics = Critic(args.state_dim, args.action_dim)
        self.target_critics = Critic(args.state_dim, args.action_dim)

        self.value = Value(args.state_dim)
        self.target_value = Value(args.state_dim)

        # initialize the target networks.
        update(self.target_actor, self.actor, 1.0)
        update(self.target_critics.model1, self.critics.model1, 1.0)
        update(self.target_critics.model2, self.critics.model2, 1.0)
        update(self.target_value.model, self.value.model, 1.0)

        self.replay = ReplayBuffer(args.state_dim, args.action_dim, args.buffer_size)
        self.action_sampler = ActionSampler(self.actor.action_dim)self.args = args

    def train_one_step(self):
        """
        Execute one update for each of the networks. Note that if no positive advantage elements
        are returned the algorithm doesn't update the actor parameters.

        Args:
            None

        Returns:
            None
        """
        # transitions is sampled from replay buffer
        transitions = self.replay.sample_batch(self.args.batch_size)
        # transitions is sampled from replay buffer
        critic_history = self.critics.train(transitions, self.target_value, self.args.gamma)
        value_history = self.value.train(
            transitions,
            # self.action_sampler # WE WILL DECIDE WHETHER WE NEED THIS LATER
            self.target_actor,
            self.target_critics,
            self.args.action_samples
        )
        # TODO: tile (states) (batch_size * K, state_dim)
        states, actions, advantages = self._sample_positive_advantage_actions(transitions.s)
        if advantages.shape[0]:
            self.actor.train(
                states,
                actions,
                advantages,
                self.args.mode,
                self.args.beta
            )

        update(self.target_actor, self.actor, self.args.tau)
        update(self.target_critics.model1, self.critics.model1, self.args.tau)
        update(self.target_critics.model2, self.critics.model2, self.args.tau)
        update(self.target_value.model, self.value.model, self.args.tau)

        return critic_history, value_history

    def _sample_positive_advantage_actions(self, states):
        """
        Sample from the target network and a uniform distribution.
        Then only keep the actions with positive advantage.
        Returning one action per state, if more needed, make states contain the
        same state multiple times.

        Args:
            states (tf.Variable): dimension (batch_size * K, state_dim)

        Returns:
            good_states (list): Set of positive advantage states
            good_actions (list): Set of positive advantage actions
            advantages (list[float]): set of positive advantage values (Q - V)
        """

        """ Sample actions """
        actions = self.action_sampler.get_actions(self.target_actor, states)
        actions = tf.concat([actions, tf.random.uniform(actions.shape, minval=-1.0, maxval=1.0)], 0)
        states = tf.concat([states, states], 0)

        """ compute Q and V dimensions (2 * batch_size * K, 1) """
        q = self.critics(states, actions)
        v = self.value(states)

        """ select s, a with positive advantage """
        indices = tf.squeeze(tf.where(q > v))
        good_states = tf.gather(states, indices)
        good_actions = tf.gather(actions, indices)
        advantages = tf.gather(q-v, indices)

        return good_states, good_actions, advantages

    def get_action(self, states):
        """
        Get a set of actions for a batch of states

        Args:
            states (tf.Variable): dimensions (TODO)

        Returns:
            sampled actions for the given state with dimension (batch_size, action_dim)
        """
        return self.action_sampler.get_actions(self.actor, states)

    def store_transitions(self, state, action, reward, next_state, is_done):
        self.replay.store(state, action, reward, next_state, is_done)

    def _tile(self, a, dim, n_tile):
        init_dim = a.shape[dim]
        num_dims = len(a.shape)
        repeat_idx = [1] * num_dims
        repeat_idx[dim] = n_tile
        tiled_results = tf.tile(a, repeat_idx)
        order_index = tf.Variable(
            np.concatenate(
                [init_dim * np.arange(n_tile) + i for i in range(init_dim)]
            ),
            dtype=tf.int64
        )
        return tf.gather_nd(tiled_results, order_index, dim)
