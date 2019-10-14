import numpy as np
import tensorflow as tf


def cosine_basis_functions(x, n_basis_functions=64):
    """
    Cosine basis function
    """
    x = x.view(-1, 1)
    i_pi = np.tile(np.arange(1, n_basis_functions + 1, dtype=np.float32), (x.shape[0], 1)) * np.pi
    i_pi = tf.Variable(i_pi)
    if x.is_cuda:
        i_pi = i_pi.cuda()
    embedding = tf.math.cos(x * i_pi)
    return embedding
