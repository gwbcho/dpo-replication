# import external dependencies
import numpy as np
import tensorflow as tf

# import local dependencies
from GAC.networks import StochasticActor, AutoRegressiveStochasticActor, VanillaActor, Critic, Value
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
        self.action_dim = args.action_dim
        self.state_dim = args.state_dim
        if args.actor == 'IQN':
            self.actor = StochasticActor(args.state_dim, args.action_dim)
            self.target_actor = StochasticActor(args.state_dim, args.action_dim)
        elif args.actor == 'AIQN':
            self.actor = AutoRegressiveStochasticActor(args.state_dim, args.action_dim)
            self.target_actor = AutoRegressiveStochasticActor(args.state_dim, args.action_dim)
        elif args.actor == 'Vanilla':
            self.actor = VanillaActor(args.state_dim, args.action_dim)
            self.target_actor = VanillaActor(args.state_dim, args.action_dim)
        else:
            raise NotImplementedError


        self.critics = Critic(args.state_dim, args.action_dim)
        self.target_critics = Critic(args.state_dim, args.action_dim)

        self.value = Value(args.state_dim)
        self.target_value = Value(args.state_dim)

        # initialize the target networks.
        update(self.target_actor, self.actor, 1.0)
        update(self.target_critics, self.critics, 1.0)
        update(self.target_value, self.value, 1.0)

        self.replay = ReplayBuffer(args.state_dim, args.action_dim, args.buffer_size)

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
        self.critics.train(transitions, self.target_value, self.args.gamma)
        self.value.train(transitions, self.target_actor, self.target_critics, self.args.action_samples)
        if self.args.actor in ['IQN', 'AIQN']:
            self.actor.train(transitions, self.target_actor, self.target_critics, self.target_value, 
                                    self.args.action_samples, self.args.mode, self.args.beta)
        elif self.args.actor in ['Vanilla']:
            self.actor.train(transitions, self.target_critics, self.args.action_samples)


        update(self.target_actor, self.actor, self.args.tau)
        update(self.target_critics, self.critics, self.args.tau)
        update(self.target_value, self.value, self.args.tau)


    def get_action(self, states):
        """
        Get a set of actions for a batch of states

        Args:
            states (tf.Variable): dimensions (batch_size, state_dim)

        Returns:
            sampled actions for the given state with dimension (batch_size, action_dim)
        """
        return self.actor.get_action(states)

    def store_transitions(self, state, action, reward, next_state, is_done):
        """
        Store the transition in the replay buffer.

        Args:
            # TODO
            state
            action
            reward
            next_state
            is_done
        """
        self.replay.store(state, action, reward, next_state, is_done)
