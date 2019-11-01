import tensorflow as tf

from helpers import ActionSampler, ReplayMemory

class GenerativeActorCritic:
    """
    Implementing Algorithm 2 in the DPO paper.
    """

    def __init__(self):
        ...


    def positive_advantage_actions(cls, states, actor, critic1, critic2, value):
        """
        Sample from the target network and a uniform distribution.
        Then only keep the actions with positive advantage.
        Returning one action per state, if more needed, make states contain the
        same state multiple times.
        """


        """ Sample actions """
        sampler = ActionSampler(actor.action_dim)
        actions = sampler.get_actions(actor, states)
        actions = tf.stack([actions, tf.random.uniform(actions.shape, minval=-1.0, maxval=1.0)])
        states = tf.stack([states, states])
        
        """ compute Q and V """
        q = tf.minimum(critic1(states, actions), critic2(states, actions))
        v = value(states)

        """ select s, a with positive advantage """
        indices = tf.squeeze(tf.where(q > v))
        good_states = tf.gather(states, indices)
        good_actions = tf.gather(actions, indices)

        return good_states, good_actions
