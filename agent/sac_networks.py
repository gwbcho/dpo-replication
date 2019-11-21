
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from agent.helpers import FNN, ActionSampler

LOG_STD_MAX = 2
LOG_STD_MIN = -8
EPS = 1e-8


def gaussian_log_den(x, mean, log_std):
    pre_sum = - 0.5 * (((x-mean)/(tf.exp(log_std)+EPS))**2 + 2 * log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1, keepdims = True)

def squash(mean, action, log_den):
    new_mean = tf.tanh(mean) # actually mean is not tanh(mean)
    new_action = tf.tanh(action)
    new_log_den = log_den - tf.reduce_sum(tf.math.log(tf.clip_by_value(1 - new_action**2, EPS, 1.0)), 
                                        axis = 1, keepdims=True)
    return  new_mean, new_action, new_log_den

    
class SACActor(tf.Module):
    def __init__(self, state_dim, action_dim):
        super(SACActor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fnn = FNN([state_dim, 256, 256])
        self.mean_layer = Dense(action_dim, input_shape = (256,))
        self.std_layer = Dense(action_dim, input_shape = (256,))
        self.optimizer = tf.keras.optimizers.Adam(0.0001)

        self.get_action(tf.zeros([1, self.state_dim]))

    def get_action(self, states, den = False):
        raw = self.fnn(states)
        mean = self.mean_layer(tf.nn.relu(raw))
        log_std = tf.sigmoid(self.std_layer(tf.nn.relu(raw))) * (LOG_STD_MAX - LOG_STD_MIN) + LOG_STD_MIN 
        action = mean + tf.random.normal(tf.shape(mean)) * tf.exp(log_std)
        log_den = gaussian_log_den(action, mean, log_std)
        mean, action, log_den = squash(mean, action, log_den)
        if den:
            return action, log_den
        else:
            return action

    def train(self, transitions, target_critics, action_samples, log_alpha):
        tiled_states = tf.tile(transitions.s, [action_samples,1])
        with tf.GradientTape() as tape:
            actions, log_den = self.get_action(tiled_states, den=True)
            loss =  tf.reduce_mean(tf.exp(log_alpha) * log_den - target_critics(tiled_states, actions))
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))



class SACCritic(tf.Module):
    '''
    Critic for SAC
    '''
    def __init__(self, state_dim, action_dim):
        super(SACCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fnn1 = FNN([state_dim + action_dim, 256, 256, 1])
        self.fnn2 = FNN([state_dim + action_dim, 256, 256, 1])
        self.optimizer1 = tf.keras.optimizers.Adam(0.0001)
        self.optimizer2 = tf.keras.optimizers.Adam(0.0001)
        self(tf.zeros([1,self.state_dim]),tf.zeros([1, self.action_dim]))

    def __call__(self, states, actions):
        x = tf.concat([states, actions], -1)
        pred1 = self.fnn1(x)
        pred2 = self.fnn2(x)
        return tf.minimum(pred1, pred2)


    def train_use_value(self, transitions, value, gamma):
        yQ = transitions.r+gamma*(1-transitions.it)*value(transitions.sp)
        x = tf.concat([transitions.s, transitions.a], -1)

        with tf.GradientTape() as tape1:
            loss1 = tf.reduce_mean((self.fnn1(x) - yQ)**2)
        gradients1 = tape1.gradient(loss1, self.fnn1.trainable_variables)
        self.optimizer1.apply_gradients(zip(gradients1, self.fnn1.trainable_variables))

        with tf.GradientTape() as tape2:
            loss2 = tf.reduce_mean((self.fnn2(x) - yQ)**2)
        gradients2 = tape2.gradient(loss2, self.fnn2.trainable_variables)
        self.optimizer2.apply_gradients(zip(gradients2, self.fnn2.trainable_variables))


    def train_no_value(self, transitions, actor, target_critics, gamma, log_alpha):
        action, log_den = actor.get_action(transitions.sp, den = True)
        criticQ = target_critics(transitions.sp, action)
        yQ = transitions.r+gamma*(1-transitions.it)*(criticQ-tf.exp(log_alpha)*log_den)
        x = tf.concat([transitions.s, transitions.a], -1)

        with tf.GradientTape() as tape1:
            loss1 = tf.reduce_mean((self.fnn1(x) - yQ)**2)
        gradients1 = tape1.gradient(loss1, self.fnn1.trainable_variables)
        self.optimizer1.apply_gradients(zip(gradients1, self.fnn1.trainable_variables))

        with tf.GradientTape() as tape2:
            loss2 = tf.reduce_mean((self.fnn2(x) - yQ)**2)
        gradients2 = tape2.gradient(loss2, self.fnn2.trainable_variables)
        self.optimizer2.apply_gradients(zip(gradients2, self.fnn2.trainable_variables))



class SACValue(tf.Module):

    """
    Value network for SAC
    """

    def __init__(self, state_dim):
        super(SACValue, self).__init__()
        self.state_dim = state_dim
        self.fnn = FNN([state_dim, 128, 128, 1])
        self.optimizer = tf.keras.optimizers.Adam(0.0001)
        self(tf.zeros([1,self.state_dim]))

    def __call__(self, states):
        return self.fnn(states)

    def train(self, transitions, actor, critic, action_samples, log_alpha):

        tiled_states = tf.tile(transitions.s, [action_samples, 1])
        actions, log_den = actor.get_action(tiled_states, den = True)

        yV = critic(tiled_states, actions) - tf.exp(log_alpha) * log_den

        with tf.GradientTape() as tape:
            loss = tf.reduce_mean((self(tiled_states) - yV)**2)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
