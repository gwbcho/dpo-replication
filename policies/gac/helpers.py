import numpy as np
import tensorflow as tf


def cosine_basis_functions(x, n_basis_functions=64):
    """
    Cosine basis function (the function is denoted as psi in the paper). This is used to embed
    [0, 1] -> R^d. The i th component of output is cos(i*x).

    Args:
        x (tf.Variable)
        n_basis_functions (int): number of basis function for the
    """
    x = tf.reshape(x, (-1, 1))
    i_pi = np.tile(np.arange(1, n_basis_functions + 1, dtype=np.float32), (x.shape[0], 1)) * np.pi
    i_pi = tf.convert_to_tensor(i_pi)
    embedding = tf.math.cos(x * i_pi)
    return embedding
