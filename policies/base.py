
import numpy as np
import tensorflow as tf


class Policy(object):

    def __init__(self):
        pass

    def get_action(self, observation):
        pass

    def create_graph(self, inputs):
        pass


class MLPPolicy(Policy):

    def __init__(self, action_space):
        self.inputs = None
        self.action = None
        self.value = None
        self.action_space = action_space

    def get_action(self, observation):
        action_dist = self.action_prob.eval(feed_dict={self.inputs: np.reshape(observation, (1, -1))})[0]
        return np.random.choice(range(self.action_space), p=action_dist)

    def create_graph(self, inputs):
        self.inputs = inputs
        l1 = tf.layers.dense(inputs, 256, activation=tf.nn.relu)
        action_logits = tf.layers.dense(l1, self.action_space)
        value = tf.layers.dense(l1, 1)
        self.action = tf.argmax(action_logits)
        self.action_prob = tf.nn.softmax(action_logits)
        self.value = tf.squeeze(value, [1])
        return self.action_prob, self.value
