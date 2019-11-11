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
        self.action_sampler = ActionSampler(self.actor.action_dim)

    def train_one_step(self):
        """
        execute one update for each of the networks
        """

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
            actor_history = self.actor.train(
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

        return critic_history, value_history, actor_history

    def _sample_positive_advantage_actions(self, states):
        """
        Sample from the target network and a uniform distribution.
        Then only keep the actions with positive advantage.
        Returning one action per state, if more needed, make states contain the
        same state multiple times.

        Args:
            states (tf.Variable): dimension (batch_size * K, state_dim)
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
        return self.action_sampler.get_actions(self.actor, states)

    def store_transitions(self, state, action, reward, next_state, is_done):
        self.replay.store(state, action, reward, next_state, is_done)
