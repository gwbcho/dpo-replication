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
    parser.add_argument('--policy_type', default='ddpg', choices=['ddpg', 'generative'])
    parser.add_argument('--num_outputs', type=int, default=1)
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--experiment_name', default=None, type=str,
            help='For multiple different experiments, provide an informative experiment name')
    parser.add_argument('--print', default=False, action='store_true')
    parser.add_argument('--actor', default='IQN', choices=['IQN', 'AIQN'])
    parser.add_argument('--normalize_obs', default=False, action='store_true', help='Normalize observations')
    parser.add_argument('--normalize_rewards', default=False, action='store_true', help='Normalize rewards')
    parser.add_argument('--q_normalization', type=float, default=0.01,
            help='Uniformly smooth the Q function in this range.')
    parser.add_argument('--target_policy', type=str, default='linear', choices=['linear', 'max', 'boltzman', 'uniform'],
            help='Target policy is constructed based on this operator.')
    parser.add_argument('--target_policy_q', type=str, default='min', choices=['min', 'max', 'mean'],
            help='The Q value for each sample is determined based on this operator over the two Q networks.')
    parser.add_argument('--boltzman_temperature', type=float, default=1.0,
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
            action = policy.get_action(state)
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
    
    results_dict = {'train_rewards': []
                    'eval_rewards': [],
                    'actor_losses': [],
                    'value_losses': [],
                    'critic_losses': [],
                    
                    }
    episode_steps, episode_rewards = 0, 0 # total steps and rewards for each episode
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
        
        # check if game is terminated to decide how to update state, episode_steps, episode_rewards
        if is_terminal:
            state = env.reset()
            results_dict['train_rewards'].append((t, episode_rewards / episode_steps))
            episode_steps = 0
            episode_rewards = 0 
        else:
            state = next_state
            episode_steps += 1
            episode_rewards += reward
            
        # train
        if gac.replay.size >= args.batch_size:
            critic_history, value_history, actor_history = gac.train_one_step()
            # I don't know how to use History objects....
            # Somebody who knows please append the losses of critc, value and actor to results_dict
            
        # evaluate
        if t % args.eval_freq == 0:
            results_dict['eval_rewards'].append((t, evaluate_policy(gac, env, args.eval_episodes)))

if __name__ == '__main__':
    main()
