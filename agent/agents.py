# import external dependencies
import numpy as np
import tensorflow as tf

# import local dependencies
from agent.gac_networks import StochasticActor, AutoRegressiveStochasticActor, Critic, Value
from agent.sac_networks import SACActor, SACCritic
from agent.helpers import ReplayBuffer, update, ActionSampler


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
        self.target_entropy = - args.action_dim # For SAC training

        if args.actor == 'IQN':
            self.actor = StochasticActor(args.state_dim, args.action_dim)
            self.target_actor = StochasticActor(args.state_dim, args.action_dim)
        elif args.actor == 'AIQN':
            self.actor = AutoRegressiveStochasticActor(args.state_dim, args.action_dim)
            self.target_actor = AutoRegressiveStochasticActor(args.state_dim, args.action_dim)
        elif args.actor == 'SAC':
            self.actor = SACActor(args.state_dim, args.action_dim)
            self.target_actor = SACActor(args.state_dim, args.action_dim)
        else:
            raise NotImplementedError

        if args.actor in ['IQN', 'AIQN']:
            self.critics = Critic(args.state_dim, args.action_dim)
            self.target_critics = Critic(args.state_dim, args.action_dim)
        elif args.actor == 'SAC':
            self.critics = SACCritic(args.state_dim, args.action_dim)
            self.target_critics = SACCritic(args.state_dim, args.action_dim)
        else:
            raise NotImplementedError


        self.value = Value(args.state_dim)
        self.target_value = Value(args.state_dim)

        self.log_alpha = tf.Variable(0.0)
        self.optimizer = tf.keras.optimizers.Adam(0.0001)
        # initialize the target networks.
        update(self.target_actor, self.actor, 1.0)
        update(self.target_critics, self.critics, 1.0)
        update(self.target_value, self.value, 1.0)

        self.replay = ReplayBuffer(args.state_dim, args.action_dim, args.buffer_size)

    def train_one_step(self):
        if self.args.actor in ['AIQN', 'IQN']:
            # transitions is sampled from replay buffer
            transitions = self.replay.sample_batch(self.args.batch_size)
            self.critics.train(transitions, self.target_value, self.args.gamma)
            self.value.train(transitions, self.target_actor, self.target_critics, self.args.action_samples)
            self.actor.train(transitions, self.target_actor, self.target_critics, self.target_value, self.args)

            update(self.target_actor, self.actor, self.args.tau)
            update(self.target_critics, self.critics, self.args.tau)
            update(self.target_value, self.value, self.args.tau)

        elif self.args.actor == 'SAC':
            transitions = self.replay.sample_batch(self.args.batch_size)
            self.critics.train(transitions, self.actor, self.target_critics, self.args.gamma, self.log_alpha)
            self.actor.train(transitions, self.critics, 1, self.log_alpha)
            
            with tf.GradientTape() as tape:
                _, log_den = self.actor.get_action(transitions.s,den = True)
                alpha_loss = - tf.reduce_mean(self.log_alpha * tf.stop_gradient(log_den + self.target_entropy))
            gradients = tape.gradient(alpha_loss, [self.log_alpha])
            self.optimizer.apply_gradients(zip(gradients, [self.log_alpha]))

            update(self.target_critics, self.critics, self.args.tau)
        else:
            raise NotImplementedError


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
