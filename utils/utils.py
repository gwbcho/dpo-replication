import os

import tensorflow as tf
import numpy as np


def save_model(actor, basedir=None):
    if not os.path.exists('models/'):
        os.makedirs('models/')

    actor_path = "{}/ddpg_actor".format(basedir)

    # print('Saving models to {} {}'.format(actor_path, adversary_path))
    tf.save_model.save(actor, actor_path)


def load_model(basedir=None):
    actor_path = "{}/ddpg_actor".format(basedir)

    print('Loading model from {}'.format(actor_path))
    actor = tf.saved_model.load(actor_path)
    return actor


# Borrowed from openai baselines running_mean_std.py
class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


def moving_average(a, n=3):
    plot_data = np.zeros_like(a)
    for idx in range(len(a)):
        length = min(idx, n)
        plot_data[idx] = a[idx-length:idx+1].mean()
    return plot_data


def vis_plot(viz, log_dict):
    ma_length = 5
    if viz is not None:
        for field in log_dict:
            if len(log_dict[field]) > 0:
                _, values = zip(*log_dict[field])

                plot_data = np.array(log_dict[field])
                viz.line(X=plot_data[:, 0], Y=moving_average(plot_data[:, 1], ma_length), win=field,
                         opts=dict(title=field, legend=[field]))
