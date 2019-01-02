import numpy as np
import tensorflow as tf
from model import *
import copy

class PPO_MLP:
    def __init__(self, sess, state_size, output_size):
        self.state_size = state_size
        self.output_size = output_size
        self.sess = sess
        self.model = MLPActorCritic('network', state_size, output_size)
        self.old_model = MLPActorCritic('old_network', state_size, output_size)

        self.gamma = 0.99
        self.lamda = 0.99
        self.lr = 0.001
        self.batch_size = 4
        self.ppo_eps = 0.2

        self.pi_trainable = self.model.get_trainable_variables()
        self.old_pi_trainable = self.old_model.get_trainable_variables()

        with tf.variable_scope('assign_op'):
            self.assign_ops = []
            for v_old, v in zip(self.old_pi_trainable, self.pi_trainable):
                self.assign_ops.append(tf.assign(v_old, v))

        self.actions = tf.placeholder(dtype=tf.int32, shape=[None])
        self.targets = tf.placeholder(dtype=tf.float32, shape=[None])
        self.adv = tf.placeholder(dtype=tf.float32, shape=[None])

        act_probs = self.model.actor
        act_probs_old = self.old_model.actor

        act_probs = tf.reduce_sum(tf.multiply(act_probs, tf.one_hot(indices=self.actions, depth=self.output_size)), axis=1)
        act_probs_old = tf.reduce_sum(tf.multiply(act_probs_old, tf.one_hot(indices=self.actions, depth=self.output_size)), axis=1)

        act_probs = tf.clip_by_value(act_probs, 1e-10, 1.0)
        act_probs_old = tf.clip_by_value(act_probs_old, 1e-10, 1.0)

        ratios = tf.exp(tf.log(act_probs) - tf.log(act_probs_old))
        clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - self.ppo_eps, clip_value_max=1 + self.ppo_eps)
        actor_loss_minimum = tf.minimum(tf.multiply(self.adv, clipped_ratios), tf.multiply(self.adv, ratios))
        actor_loss = -tf.reduce_mean(actor_loss_minimum)

        values = self.model.critic
        critic_loss = tf.squared_difference(self.targets, tf.squeeze(values))
        critic_loss = tf.reduce_mean(critic_loss)

        total_loss = actor_loss + critic_loss
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(total_loss, var_list=self.pi_trainable)

    def train_model(self, state, action, targets, advs):
        self.sess.run(self.train_op, feed_dict={self.model.input: state,
                                                self.old_model.input: state,
                                                self.actions: action,
                                                self.adv: advs,
                                                self.targets: targets})

    def assign_policy_parameters(self):
        return self.sess.run(self.assign_ops)

    def get_action(self, state):
        action = self.sess.run(self.model.actor, feed_dict={self.model.input: state})
        action = [np.random.choice(self.output_size, p=i) for i in action]
        return np.stack(action)

    def get_value(self, state, next_state):
        value = self.sess.run(tf.squeeze(self.model.critic), feed_dict={self.model.input: state})
        next_value = self.sess.run(tf.squeeze(self.model.critic), feed_dict={self.model.input: next_state})
        return value, next_value

    def get_gaes(self, rewards, dones, values, next_values):
        deltas = [r + self.gamma * (1-d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1-dones[t]) * self.gamma * self.lamda * gaes[t + 1]
        
        target = gaes + values
        gaes = (gaes - gaes.mean())/(gaes.std() + 1e-30)
        return gaes, target

if __name__ == '__main__':
    
    sess = tf.Session()
    state_size, output_size = 8, 4
    
    ppo = PPO_MLP(sess, state_size, output_size)
    sess.run(tf.global_variables_initializer())
    ppo.assign_policy_parameters()

    state, next_state = np.random.rand(5, state_size), np.random.rand(5, state_size)

    
    state = np.random.rand(5, state_size)
    
    actions = [0, 3, 2, 1, 2]
    target = np.random.rand(5)
    adv = np.random.rand(5)
    ppo.train_model(state, actions, target, adv)
    


    '''    
    reward = [1, 0, 1, 1, 0]
    done = [False, False, True, False, True]
    value = [5, 3, 4, 2, 5]
    next_value = [3, 4, 2, 5, 1]

    result = ppo.get_gaes(reward, done, value, next_value)
    print(result)
    '''
    

    # use_gae == False
    # target : 0 + 0.99 * (1 - 1) * 1 = 0       adv : -5
    # target : 1 + 0.99 * (1 - 0) * 5 = 5.95    adv : 3.95
    # target : 1 + 0.99 * (1 - 1) * 2 = 1       adv : -3
    # target : 0 + 0.99 * (1 - 0) * 4 = 3.96    adv : 0.96
    # target : 1 + 0.99 * (1 - 0) * 3 = 3.97    adv : -1.03

    # use_gae == True
    # deltas : 0 + 0.99 * (1 - 1) * 1 = 0       adv : -5
    # deltas : 1 + 0.99 * (1 - 0) * 5 = 5.95    
    # deltas : 1 + 0.99 * (1 - 1) * 2 = 1       
    # deltas : 0 + 0.99 * (1 - 0) * 4 = 3.96    
    # deltas : 1 + 0.99 * (1 - 0) * 3 = 3.97    