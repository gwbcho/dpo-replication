# external dependencies
import numpy as np
import tensorflow as tf

# internal dependencies
import policies.policy_helpers.helpers as helpers
import policies.policy_helpers.helper_classes as helper_classes
import utils.utils as utils

"""
File Description:

This file contains the Policy class which is the parent for all policies in this project. It's meant
to contain bare bones functionality to be modified after it's instantiation.
"""


class Policy(helper_classes.HelperPolicyClass):
    """
    Policy function for bare bones RL
    """

    def __init__(self, gamma, tau, num_inputs, action_space, replay_size, normalize_obs=True,
                 normalize_returns=False):
        super(Policy, self).__init__()
        self.num_inputs = num_inputs
        self.action_space = action_space

        self.gamma = gamma
        self.tau = tau
        self.normalize_observations = normalize_obs
        self.normalize_returns = normalize_returns

        if self.normalize_observations:
            self.obs_rms = utils.RunningMeanStd(shape=num_inputs)
        else:
            self.obs_rms = None

        if self.normalize_returns:
            self.ret_rms = utils.RunningMeanStd(shape=1)
            self.ret = 0
            self.cliprew = 10.0
        else:
            self.ret_rms = None

        self.memory = helper_classes.ReplayMemory(replay_size)
        self.actor = None
        self.actor_perturbed = None
        self.policy = helper_classes.ActionSampler(self.action_space)

    def select_action(self, state, action_noise=None, param_noise=None):
        state = helpers.normalize(
            tf.Variable(state),
            self.obs_rms
        )

        if param_noise is not None:
            action = self.policy.get_actions(self.actor_perturbed, state)[0]
        else:
            action = self.policy.get_actions(self.actor, state)[0]

        if action_noise is not None:
            action += tf.Variable(action_noise())

        action = tf.clip_by_value(action, -1, 1)

        return action

    def store_transition(self, state, action, mask, next_state, reward):
        B = state.shape[0]
        for b in range(B):
            self.memory.push(state[b], action[b], mask[b], next_state[b], reward[b])
            if self.normalize_observations:
                self.obs_rms.update(state[b])
            if self.normalize_returns:
                self.ret = self.ret * self.gamma + reward[b]
                self.ret_rms.update(np.array([self.ret]))
                if mask[b] == 0:  # if terminal is True
                    self.ret = 0

    def update_parameters(self, batch_size):
        transitions = self.memory.sample(batch_size)
        batch = helper_classes.Transition(*zip(*transitions))

        state_batch = helpers.normalize(
            tf.Variable(
                tf.stack(batch.state)
            ),
            self.obs_rms
        )

        action_batch = tf.Variable(tf.stack(batch.action))
        reward_batch = helpers.normalize(
            tf.Variable(
                tf.expand_dims(
                    tf.stack(batch.reward),
                    1
                )
            ),
            self.ret_rms
        )
        mask_batch = tf.Variable(
            tf.expand_dims(
                tf.stack(batch.mask),
                1
            )
        )
        next_state_batch = helpers.normalize(
            tf.Variable(
                tf.stack(batch.next_state)
            ),
            self.obs_rms
        )

        if self.normalize_returns:
            reward_batch = tf.clip_by_value(reward_batch, -self.cliprew, self.cliprew)

        value_loss = self.update_critic(
            state_batch,
            action_batch,
            reward_batch,
            mask_batch,
            next_state_batch
        )
        policy_loss = self.update_actor(state_batch)

        self.soft_update()

        return value_loss, policy_loss

    def perturb_actor_parameters(self, param_noise):
        """
        Apply parameter noise to actor model, for exploration
        """
        helpers.hard_update(self.actor_perturbed, self.actor)
        params = self.actor_perturbed.state_dict()
        for name in params:
            if 'ln' in name:
                pass
            param = params[name]
            param += tf.random.normal(param.shape) * param_noise.current_stddev

    def _tile(self, a, n_tile):
        return tf.tile(a, n_tile)
