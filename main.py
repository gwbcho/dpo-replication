import os
import argparse

import gym
import numpy as np
import tensorflow as tf

from GAC.networks import AutoRegressiveStochasticActor as AIQN
from GAC.networks import StochasticActor as IQN
from GAC.networks import Critic, Value
from GAC.agent import GACAgent


def create_argument_parser():
    parser = argparse.ArgumentParser(
            description='An implementation of the Distributional Policy Optimization paper.',
            )
    parser.add_argument('--environment', default="HalfCheetah-v2",
            help='name of the environment to run')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
            help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=5e-3, metavar='G',
            help='discount factor for model (default: 0.01)')
    parser.add_argument('--noise', default='ou', choices=['ou', 'param', 'normal'])
    parser.add_argument('--noise_scale', type=float, default=0.2, metavar='G',
            help='(default: 0.2)')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
            help='batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
            help='number of training epochs (default: None)')
    parser.add_argument('--epochs_cycles', type=int, default=20, metavar='N')
    parser.add_argument('--rollout_steps', type=int, default=100, metavar='N')
    parser.add_argument('--T', type=int, default=2000000, metavar='N',
            help='number of training steps (default: 2000000)')
    parser.add_argument('--model_path', type=str, default='/tmp/dpo/',
            help='trained model is saved to this location')
    parser.add_argument('--param_noise_interval', type=int, default=50, metavar='N')
    parser.add_argument('--start_timesteps', type=int, default=10000, metavar='N')
    parser.add_argument('--eval_freq', type=int, default=5000, metavar='N')
    parser.add_argument('--eval_episodes', type=int, default=10, metavar='N')
    parser.add_argument('--buffer_size', type=int, default=1000000, metavar='N',
            help='size of replay buffer (default: 1000000)')
    parser.add_argument('--action_samples', type=int, default=1)
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--experiment_name', default=None, type=str,
            help='For multiple different experiments, provide an informative experiment name')
    parser.add_argument('--print', default=False, action='store_true')
    parser.add_argument('--actor', default='IQN', choices=['IQN', 'AIQN'])
    parser.add_argument('--normalize_obs', default=False, action='store_true', help='Normalize observations')
    parser.add_argument('--normalize_rewards', default=False, action='store_true', help='Normalize rewards')
    parser.add_argument('--q_normalization', type=float, default=0.01,
            help='Uniformly smooth the Q function in this range.')
    parser.add_argument('--mode', type=str, default='linear', choices=['linear', 'max', 'boltzman', 'uniform'],
            help='Target policy is constructed based on this operator.')
    parser.add_argument('--beta', type=float, default=1.0,
            help='Boltzman Temperature for normalizing actions')
    return parser


def evaluate_policy(policy, env, episodes):
    """
    Run the environment env using policy for episodes number of times.
    Return: average rewards per episode.
    """
    total_reward = 0.0
    for _ in range(episodes):
        state = env.reset()
        while True:
            action = policy.act(state)
            state, reward, is_terminal, _ = env.step(action)
            total_reward += reward
            if is_terminal:
                break
    return total_reward / episodes


def main():
    args = create_argument_parser().parse_args()

    """
    Create Mujoco environment
    """
    env = gym.make(args.environment)
    args.action_dim = env.action_space.shape[0]
    args.state_dim = env.observation_space.shape[0]

    gac = GACAgent(args)

    state = env.reset()

    """
    training loop
    """
    for t in range(args.T):
        """
        Get an action from neural network and run it in the environment
        """
        action = gac.get_action(tf.convert_to_tensor([state]))
        next_state, reward, is_terminal, _ = env.step(action)
        gac.store_transitions(state, action, reward, next_state, is_terminal)
        state = env.reset() if is_terminal else next_state

        if gac.replay.size >= args.batch_size:
            gac.train_one_step()


if __name__ == '__main__':
    main()
