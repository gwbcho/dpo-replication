# import external dependencies
import tensorflow as tf

# import local dependencies
import policies.gac.gac_helpers as gac_helpers
import policies.gac.gac_networks as gac_networks
import policies.policy as policy
import policies.policy_helpers as policy_helpers


"""
File Description:

This file hosts the Generator class which is a subclass of Policy. The purpose of this class is to
unify the networks constructed for GAC in gac_networks and construct a generator function which
follows an optimal stationary stochastic policy.
"""


class Generative(policy.Policy):
    """
    Class to construct a generative policy which samples from an optimally constructed
    distribution over continuous action space.
    """

    def __init__(self, gamma, tau, num_inputs, action_space, replay_size, normalize_obs=False, normalize_returns=False,
                 num_basis_functions=64, num_outputs=1, use_value=True, q_normalization=0.01, target_policy='linear',
                 target_policy_q='min', autoregressive=True, temp=1.0):

        super(Generative, self).__init__(gamma=gamma, tau=tau, num_inputs=num_inputs, action_space=action_space,
                                         replay_size=replay_size, normalize_obs=normalize_obs,
                                         normalize_returns=normalize_returns)

        # TODO: Construct function as specified in the paper
        super(Generative, self).__init__()

        self.num_inputs = num_inputs
        self.action_space = action_space
        self.num_outputs = num_outputs
        self.num_basis_functions = num_basis_functions
        self.action_dim = self.action_space.shape[0]
        self.use_value = use_value
        self.q_normalization = q_normalization
        self.target_policy = target_policy
        self.autoregressive = autoregressive
        self.temp = temp

        if target_policy_q == 'min':
            self.target_policy_q = lambda x, y: tf.math.minimum(x, y)
        elif target_policy_q == 'max':
            self.target_policy_q = lambda x, y: tf.math.minimum(x, y)
        else:
            self.target_policy_q = lambda x, y: (x + y / 2)

        if self.autoregressive:
            self.actor = gac_networks.AutoRegressiveStochasticActor(self.num_inputs, self.action_dim, self.num_basis_functions)
            self.actor_target = gac_networks.AutoRegressiveStochasticActor(self.num_inputs, self.action_dim, self.num_basis_functions)
            self.actor_perturbed = gac_networks.AutoRegressiveStochasticActor(self.num_inputs, self.action_dim, self.num_basis_functions)

        else:
            self.actor = gac_networks.StochasticActor(self.num_inputs, self.action_dim, self.num_basis_functions)
            self.actor_target = gac_networks.StochasticActor(self.num_inputs, self.action_dim, self.num_basis_functions)
            self.actor_perturbed = gac_networks.StochasticActor(self.num_inputs, self.action_dim, self.num_basis_functions)


        # self.actor_optim = Adam(self.actor.parameters(), lr=1e-3)

        self.critic = gac_networks.Critic(self.num_inputs + self.action_dim, num_networks=2)
        self.critic_target = gac_networks.Critic(self.num_inputs + self.action_dim, num_networks=2)
        # self.critic_optim = Adam(self.critic.parameters(), lr=1e-3)

        self.value = gac_networks.Value(self.num_inputs)
        self.value_target = gac_networks.Value(self.num_inputs)
        # self.value_optim = Adam(self.value.parameters(), lr=1e-3)