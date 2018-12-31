import numpy as np
import tensorflow as tf

class MLPActorCritic:
    def __init__(self, name, state_size, output_size):
        self.state_size = state_size
        self.output_size = output_size

        with tf.variable_scope(name):
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.state_size])
            self.dense_1 = tf.layers.dense(inputs=self.input, units=256, activation=tf.nn.relu)
            self.actor = tf.layers.dense(inputs=self.dense_1, units=self.output_size, activation=tf.nn.softmax)
            self.critic = tf.layers.dense(inputs=self.dense_1, units=1, activation=None)
    
            self.scope = tf.get_variable_scope().name

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)