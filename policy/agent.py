# import external dependencies
import numpy as np
import tensorflow as tf

# internal dependencies
import utils.utils as utils

# import local dependencies
import policy.networks as networks
import policy.policy_helpers.helper_functions as policy_helper_functions
import policy.policy_helpers.helper_classes as policy_helper_classes


"""
File Description:

This file hosts the GACAgent class which is a form of policy. The purpose of this class is to
unify the networks constructed for GAC in networks and construct a generator function which
follows an optimal stationary stochastic policy.
"""


class GACAgent:
    """
    Class to construct a generative policy which samples from an optimally constructed
    distribution over continuous action space.
    """

    def __init__(self, gamma, soft_update_rate, state_dim, action_space, replay_size,
                 normalize_obs=False, normalize_returns=False, num_basis_functions=64,
                 num_outputs=1, use_value=True, q_normalization=0.01, target_policy='linear',
                 target_policy_q='min', autoregressive=True, temp=1.0):

        # TODO: Construct function as specified in the paper
        self.state_dim = state_dim
        self.action_space = action_space
        self.num_outputs = num_outputs
        self.num_basis_functions = num_basis_functions
        self.action_dim = self.action_space.shape[0]
        self.use_value = use_value
        self.q_normalization = q_normalization
        self.target_policy = target_policy
        self.autoregressive = autoregressive
        self.temp = temp

        # initial Policy class initialized values
        self.gamma = gamma
        self.soft_update_rate = soft_update_rate
        self.normalize_observations = normalize_obs
        self.normalize_returns = normalize_returns

        if self.normalize_observations:
            self.obs_rms = utils.RunningMeanStd(shape=state_dim)
        else:
            self.obs_rms = None

        if self.normalize_returns:
            self.ret_rms = utils.RunningMeanStd(shape=1)
            self.ret = 0
            self.cliprew = 10.0
        else:
            self.ret_rms = None

        self.memory = policy_helper_classes.ReplayMemory(replay_size)
        self.actor = None
        self.actor_perturbed = None
        self.policy = policy_helper_classes.ActionSampler(self.action_dim)

        if target_policy_q == 'min':
            self.target_policy_q = lambda x, y: tf.math.minimum(x, y)
        elif target_policy_q == 'max':
            self.target_policy_q = lambda x, y: tf.math.minimum(x, y)
        else:
            self.target_policy_q = lambda x, y: (x + y / 2)

        if self.autoregressive:
            self.actor = networks.AutoRegressiveStochasticActor(
                self.state_dim,
                self.action_dim,
                self.num_basis_functions
            )
            self.actor_target = networks.AutoRegressiveStochasticActor(
                self.state_dim,
                self.action_dim,
                self.num_basis_functions
            )
            self.actor_perturbed = networks.AutoRegressiveStochasticActor(
                self.state_dim,
                self.action_dim,
                self.num_basis_functions
            )

        else:
            self.actor = networks.StochasticActor(
                self.state_dim,
                self.action_dim,
                self.num_basis_functions
            )
            self.actor_target = networks.StochasticActor(
                self.state_dim,
                self.action_dim,
                self.num_basis_functions
            )
            self.actor_perturbed = networks.StochasticActor(
                self.state_dim,
                self.action_dim,
                self.num_basis_functions
            )

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        self.critic = networks.Critic(self.state_dim + self.action_dim, num_networks=2)
        self.critic_target = networks.Critic(self.state_dim + self.action_dim, num_networks=2)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        self.value = networks.Value(self.state_dim)
        self.value_target = networks.Value(self.state_dim)
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def select_action(self, state, action_noise=None, param_noise=None):
        state = policy_helper_functions.normalize(
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
        batch = policy_helper_classes.Transition(*zip(*transitions))

        state_batch = policy_helper_functions.normalize(
            tf.Variable(
                tf.stack(batch.state)
            ),
            self.obs_rms
        )

        action_batch = tf.Variable(tf.stack(batch.action))
        reward_batch = policy_helper_functions.normalize(
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
        next_state_batch = policy_helper_functions.normalize(
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
        policy_helper_functions.hard_update(self.actor_perturbed, self.actor)
        params = self.actor_perturbed.state_dict()
        for name in params:
            if 'ln' in name:
                pass
            param = params[name]
            param += tf.random.normal(param.shape) * param_noise.current_stddev

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

    # TODO: implement the following functions
    def update_critic(self, state_batch, action_batch, reward_batch, mask_batch, next_state_batch):
        raise NotImplementedError

    def update_actor(self, state_batch):

    def soft_update(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError
