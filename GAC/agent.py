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

        self.critic = Critic(args.state_dim, args.action_dim)
        self.target_critic = Critic(args.state_dim, args.action_dim)

        self.value = Value(args.state_dim)
        self.target_value = Value(args.state_dim)

        # initialize the target networks.
        update(self.target_actor, self.actor, 1.0)
        update(self.target_critic, self.critic, 1.0)
        update(self.target_value, self.value, 1.0)

        self.replay = ReplayBuffer(args.state_dim, args.action_dim, args.buffer_size) 
        


    
    def train_one_step(self):
        """
        execute one update for each of the networks
        """

        transitions = self.replay.sample_batch(self.args.batch_size) 
        # transitions is sampled from replay buffer
        critic_history = self.critic.train(transitions, self.target_value, self.args.gamma)
        value_history = self.value.train(
                transitions,
                # ActionSampler(self.actor.action_dim) # WE WILL DECIDE WHETHER WE NEED THIS LATER
                self.target_actor,
                self.target_critic,
                self.args.action_samples
                )
        states, actions, advantages = self._sample_positive_advantage_actions(transitions.s)
        actor_history = self.actor.train(states, actions, advantages, self.args.mode, self.args.beta)

        update(self.target_actor, self.actor, self.args.soft_rate)
        update(self.target_critic, self.critic, self.args.soft_rate)
        update(self.target_value, self.value, self.args.soft_rate)


        return critic_history, value_history, actor_history


    def _sample_positive_advantage_actions(self, states):
        """
        Sample from the target network and a uniform distribution.
        Then only keep the actions with positive advantage.
        Returning one action per state, if more needed, make states contain the
        same state multiple times.
        """

        """ Sample actions """
        actions = ActionSampler(self.actor.action_dim).get_actions(self.actor, states)
        actions = tf.concat([actions, tf.random.uniform(actions.shape, minval=-1.0, maxval=1.0)], 0)
        states = tf.concat([states, states], 0)
        
        """ compute Q and V """
        q = self.critic(states, actions)
        v = self.value(states)

        """ select s, a with positive advantage """
        indices = tf.squeeze(tf.where(q > v))
        good_states = tf.gather(states, indices)
        good_actions = tf.gather(actions, indices)
        advantages = tf.gather(q-v, indices)

        return good_states, good_actions, advantages   


    def get_action(self, states):
        return ActionSampler(self.actor.action_dim).get_actions(self.target_actor, states) 

    def store_transitions(self, state, action, reward, next_state, is_done):
        self.replay.store(state, action, reward, next_state, is_done)