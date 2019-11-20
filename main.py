import os
import argparse

import gym
import numpy as np
import tensorflow as tf

from GAC.agent import GACAgent

from environment.rescale import normalize, denormalize

def create_argument_parser():

    parser = argparse.ArgumentParser(
            description='An implementation of the Distributional Policy Optimization paper.')
    parser.add_argument('--environment', default="HalfCheetah-v2",
            help='name of the environment to run. default="HalfCheetah-v2"')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
            help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=5e-3, metavar='G',
            help='discount factor for model (default: 0.005)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
            help='batch size (default: 64)')
    parser.add_argument('--buffer_size', type=int, default=1000000, metavar='N',
            help='size of replay buffer (default: 1000000)')
    parser.add_argument('--action_samples', type=int, default=16)
    parser.add_argument('--actor', default='IQN')
    parser.add_argument('--mode', type=str, default='linear', choices=['linear', 'max', 'boltzman', 'uniform'],
            help='Target policy is constructed based on this operator. default="linear" ')
    parser.add_argument('--beta', type=float, default=1.0,
            help='Boltzman Temperature for normalizing actions, default=1.0')

    parser.add_argument('--T', type=int, default=2000000, metavar='N',
            help='number of training steps (default: 2000000)')
    parser.add_argument('--eval_freq', type=int, default=5000, metavar='N')
    parser.add_argument('--eval_episodes', type=int, default=10, metavar='N')
    parser.add_argument('--norm_state', default=False, action='store_true',
            help='normalize the state to [-1,1]')

    parser.add_argument('--model_path', type=str, default='/tmp/dpo/',
            help='trained model is saved to this location, default="/tmp/dpo/"')

    return parser


def evaluate_policy(actor, env, args):
    """
    Run the environment env using policy for episodes number of times.
    Return: average rewards per episode.
    """
    total_reward = 0.0
    for _ in range(args.eval_episodes):
        state = env.reset()
        state = normalize(state, args.state_low, args.state_high)
        while True:
            action = actor.get_action(tf.convert_to_tensor([state]))
            action = tf.squeeze(action, [0]).numpy()
            action = denormalize(action, args.action_low, args.action_high)

            state, reward, is_terminal, _ = env.step(action)
            if args.norm_state:
                state = normalize(state, args.state_low, args.state_high)
            total_reward += reward
            if is_terminal:
                break
    return total_reward / args.eval_episodes


def main():
    args = create_argument_parser().parse_args()

    """
    Create Mujoco environment
    """
    env = gym.make(args.environment)
    env_eval = gym.make(args.environment)

    args.action_dim = env.action_space.shape[0]
    args.action_low = env.action_space.low
    args.action_high = env.action_space.high
    
    args.state_dim = env.observation_space.shape[0]
    args.state_low = env.observation_space.low
    args.state_high = env.observation_space.high

    gac = GACAgent(args)

    state = env.reset()
    if args.norm_state:
        state = normalize(state, args.state_low, args.state_high)

    results_dict = {
        'train_rewards': [],
        'eval_rewards': [],
        'actor_losses': [],
        'value_losses': [],
        'critic_losses': []
    }
    episode_steps, episode_rewards = 0, 0 # total steps and rewards for each episode
    """
    training loop
    """
    episode_count = 0
    for t in range(args.T):
        """
        Get an action from neural network and run it in the environment
        """

        action = gac.get_action(tf.convert_to_tensor([state]))
        action = tf.squeeze(action, [0]).numpy() 
        action = denormalize(action, args.action_low, args.action_high)

        next_state, reward, is_terminal, _ = env.step(action)
        if args.norm_state:
            next_state = normalize(next_state, args.state_low, args.state_high)

        if episode_count % 10 == 0 or episode_count > 100:
            env.render()
        gac.store_transitions(state, action, reward, next_state, is_terminal)
        
        episode_rewards += reward
        # check if game is terminated to decide how to update state
        if is_terminal:
            state = env.reset()
            if args.norm_state:
                state = normalize(state, args.state_low, args.state_high)
            episode_count += 1
            results_dict['train_rewards'].append((t, episode_rewards))
            print('training episode: {}, current interactions: {}, total interactions: {}, reward: {}'
                    .format(episode_count, episode_steps+1, t+1, episode_rewards))
            episode_steps = 0
            episode_rewards = 0
        else:
            state = next_state
            episode_steps += 1

        # train
        if gac.replay.size >= args.batch_size:
            gac.train_one_step()

        # evaluate
        if t % args.eval_freq == 0:
            eval_reward = evaluate_policy(gac, env_eval, args)
            print('eval_reward:', eval_reward)
            results_dict['eval_rewards'].append((t, eval_reward))

if __name__ == '__main__':
    main()
