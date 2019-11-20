import numpy as np


def normalize(x, x_min, x_max):
    '''
    x approximately in [min,max] in each dimension.
    y in [-1,1] 
    usually for state
    '''
    y = (x - x_min)/(x_max-x_min) * 2.0 - 1.0
    return np.clip(y, -1, 1)

def denormalize(x, y_min, y_max):
    '''
    x approximately in [-1,1] in each dimension.
    y in [min,max] 
    usually for action
    '''
    y = (x * 0.5 + 0.5) * (y_max - y_min) + y_min
    return np.clip(y, y_min, y_max)