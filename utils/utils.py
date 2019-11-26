import os

import tensorflow as tf
import numpy as np


def save_model(actor, basedir=None):
    if not os.path.exists('models/'):
        os.makedirs('models/')

    actor_path = "{}/actor".format(basedir)

    # print('Saving models to {} {}'.format(actor_path, adversary_path))
    tf.saved_model.save(actor, actor_path)


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
