import tensorflow as tf

from policies.gac.helpers import ActionSampler, ReplayMemory

class GenerativeActorCritic:
    """
    Implementing Algorithm 2 in the DPO paper.
    """

    def __init__(self, 
            actor, 
            critic, 
            value,
            target_actor,
            target_critic,
            target_value,
            args
            ):
        self.actor = actor
        self.critic = critic
        self.value = value
        self.target_actor = target_actor
        self.target_critic = target_critic
        self.target_value = target_value
        self.args = args


    def train_one_step(self, transitions):
        """
        execute one update for each of the networks
        transitions is sampled from replay buffer
        """

        critic_history = self.critic.train(transitions, self.target_value, self.args.gamma)
        value_history = self.value.train(
                transitions,
                ActionSampler(self.actor.action_dim),
                self.target_actor,
                self.target_critic,
                self.target_critic,
                self.args.action_samples
                )
        states, actions, advantages = self._sample_positive_advantage_actions(transitions.s)
        actor_history = self.actor.train(states, actions, advantages, self.args.mode)
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
