import tensorflow as tf

class CNNActorCritic:
    def __init__(self, name, window_size, obs_stack, output_size):
        self.window_size = window_size
        self.output_size = output_size
        self.obs_stack = obs_stack

        with tf.variable_scope(name):
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, window_size, window_size, obs_stack])
            self.conv1 = tf.layers.conv2d(inputs=self.input, filters=16, kernel_size=[3, 3], strides=[1, 1], padding='SAME', activation=tf.nn.relu)
            self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64, kernel_size=[3, 3], strides=[1, 1], padding='SAME', activation=tf.nn.relu)
            self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=128, kernel_size=[3, 3], strides=[1, 1], padding='SAME', activation=tf.nn.relu)

            self.reshape = tf.reshape(self.conv3, [-1, self.window_size * self.window_size * 128])
            self.dense_1 = tf.layers.dense(inputs=self.reshape, units=self.window_size * self.window_size, activation=tf.nn.relu)
            
            self.actor = tf.layers.dense(inputs=self.dense_1, units=self.output_size, activation=tf.nn.softmax)

            self.critic_layer_1 = tf.layers.dense(inputs=self.dense_1, units=64, activation=tf.nn.tanh)
            self.critic = tf.layers.dense(inputs=self.critic_layer_1, units=1, activation=None)

            self.scope = tf.get_variable_scope().name

    
    def get_action_prob(self, obs):
        return self.sess.run(self.act_probs, feed_dict={self.obs: obs})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)


class MLPActorCritic:
    def __init__(self, name, state_size, output_size):
        self.state_size = state_size
        self.output_size = output_size

        with tf.variable_scope(name):
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.state_size])
            self.dense_1 = tf.layers.dense(inputs=self.input, units=256, activation=tf.nn.relu)
            self.dense_2 = tf.layers.dense(inputs=self.dense_1, units=256, activation=tf.nn.relu)
            self.dense_3 = tf.layers.dense(inputs=self.dense_2, units=256, activation=tf.nn.relu)

            self.actor_layer_1 = tf.layers.dense(inputs=self.dense_3, units=256, activation=tf.nn.relu)
            self.actor_layer_2 = tf.layers.dense(inputs=self.actor_layer_1, units=256, activation=tf.nn.relu)
            self.actor = tf.layers.dense(inputs=self.actor_layer_2, units=self.output_size, activation=tf.nn.softmax)

            self.critic_layer_1 = tf.layers.dense(inputs=self.dense_3, units=256, activation=tf.nn.tanh)
            self.critic_layer_2 = tf.layers.dense(inputs=self.critic_layer_1, units=256, activation=None)
            self.critic = tf.layers.dense(inputs=self.critic_layer_2, units=1, activation=None)
    
            self.scope = tf.get_variable_scope().name

    def get_action_prob(self, obs):
        return self.sess.run(self.act_probs, feed_dict={self.obs: obs})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)