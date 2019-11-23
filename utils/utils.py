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

