import numpy as np
import tensorflow as tf

class MLPActorCritic:
    def __init__(self, name, state_size, output_size):
        self.state_size = state_size
        self.output_size = output_size

        with tf.variable_scope(name):
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.state_size])
            self.dense_1 = tf.layers.dense(inputs=self.input, units=256, activation=tf.nn.relu)
            self.dense_2 = tf.layers.dense(inputs=self.dense_1, units=256, activation=tf.nn.relu)
            self.dense_3 = tf.layers.dense(inputs=self.dense_2, units=256, activation=tf.nn.relu)
            self.actor = tf.layers.dense(inputs=self.dense_3, units=self.output_size, activation=tf.nn.softmax)
            self.critic = tf.layers.dense(inputs=self.dense_3, units=1, activation=None)
    
            self.scope = tf.get_variable_scope().name

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

class CNNActorCritic:
    def __init__(self, name, obs_stack, window_size, output_size):
        self.window_size = window_size
        self.obs_stack = obs_stack
        self.output_size = output_size

        with tf.variable_scope(name):
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, window_size, window_size, obs_stack])
            self.layer_1 = tf.layers.conv2d(inputs=self.input, filters=16, kernel_size=[3, 3], strides=[1, 1], padding='SAME', activation=tf.nn.relu)
            self.layer_2 = tf.layers.conv2d(inputs=self.layer_1, filters=64, kernel_size=[3, 3], strides=[1, 1], padding='SAME', activation=tf.nn.relu)
            self.reshape = tf.reshape(self.layer_2, [-1, window_size * window_size * 64])

            self.dense_1 = tf.layers.dense(inputs=self.reshape, units=256, activation=tf.nn.relu)
            self.actor = tf.layers.dense(inputs=self.dense_1, units=self.output_size, activation=tf.nn.softmax)
            self.critic = tf.layers.dense(inputs=self.dense_1, units=1, activation=None)

            self.scope = tf.get_variable_scope().name

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)