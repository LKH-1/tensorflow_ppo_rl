from model import *
import copy

class MLPAgent:
    def __init__(self, sess, state_size, output_size):
        self.name = 'network'
        self.sess = sess
        self.state_size = state_size
        self.output_size = output_size
        self.ppo_eps = 0.2
        self.lr = 0.001
        self.epoch = 3
        self.batch_size = 32
        self.lamda = 0.99

        self.model = MLPActorCritic(self.name, self.state_size, self.output_size)
        self.gamma = 0.99

        pi_trainable = self.model.get_trainable_variables()

        self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
        self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
        self.next_value = tf.placeholder(dtype=tf.float32, shape=[None], name='v_preds_next')
        self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')
        self.old_policy = tf.placeholder(dtype=tf.float32, shape=[None, self.output_size], name='old_policy')
        self.target = tf.placeholder(dtype=tf.float32, shape=[None], name='target')

        self.act_probs = self.model.actor * tf.one_hot(indices=self.actions, depth=self.output_size)
        self.act_probs_ = tf.reduce_sum(self.act_probs, axis=1)
        self.clip_act_probs = tf.clip_by_value(self.act_probs_, clip_value_min=1e-10, clip_value_max=1.0)
        
        self.act_probs_old = self.old_policy * tf.one_hot(indices=self.actions, depth=self.output_size)
        self.act_probs_old_ = tf.reduce_sum(self.act_probs_old, axis=1)
        self.clip_act_probs_old = tf.clip_by_value(self.act_probs_old_, clip_value_min=1e-10, clip_value_max=1.0)

        self.ratio = tf.exp(tf.log(self.clip_act_probs) - tf.log(self.clip_act_probs_old))
        self.clip_ratio = tf.clip_by_value(self.ratio, clip_value_min=1 - self.ppo_eps, clip_value_max=1 + self.ppo_eps)

        loss_actor_clip = tf.minimum(tf.multiply(self.gaes, self.ratio), tf.multiply(self.gaes, self.clip_ratio))
        loss_actor_clip = tf.reduce_mean(loss_actor_clip)

        loss_vf = tf.squared_difference(self.target, self.model.critic)
        loss_vf = tf.reduce_mean(loss_vf)

        loss = -loss_actor_clip + 0.5 * loss_vf

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(loss, var_list=pi_trainable)

    def train_model(self, state, actions, targets, adv):
        old_policy = self.sess.run(self.model.actor, feed_dict={self.model.input: state})
        
        sample_range = np.arange(len(state))
        for i in range(self.epoch):
            np.random.shuffle(sample_range)
            for j in range(int(len(state)/self.batch_size)):
                sample_idx = sample_range[self.batch_size * j:self.batch_size * (j + 1)]
                state_batch = state[sample_idx]
                actions_batch = actions[sample_idx]
                targets_batch = targets[sample_idx]
                adv_batch = adv[sample_idx]
                old_policy_batch = old_policy[sample_idx]
                self.sess.run(self.train_op, feed_dict={self.model.input: state_batch,
                                                    self.actions: actions_batch,
                                                    self.target: targets_batch,
                                                    self.gaes: adv_batch,
                                                    self.old_policy: old_policy_batch})
        
    def get_action(self, state):
        policy, value = self.sess.run([self.model.actor, self.model.critic], feed_dict={self.model.input: state})
        action = [np.random.choice(self.output_size, p=i) for i in policy]
        policy = [i[j] for i, j in zip(policy, action)]
        return action, policy

    def get_value(self, done, state, next_state):
        value = self.sess.run(tf.squeeze(self.model.critic), feed_dict={self.model.input: state})
        next_value = self.sess.run(tf.squeeze(self.model.critic), feed_dict={self.model.input: next_state})
        return value, next_value

    def get_gaes(self, reward, done, value, next_value):
        num_step = len(reward)
        discounted_return = np.empty([num_step])

        gae = 0
        for t in range(num_step - 1, -1, -1):
            delta = reward[t] + self.gamma * \
                next_value[t] * (1 - done[t]) - value[t]
            gae = delta + self.gamma * self.lamda * (1 - done[t]) * gae

            discounted_return[t] = gae + value[t]

        # For Actor
        adv = discounted_return - value

        return adv, discounted_return