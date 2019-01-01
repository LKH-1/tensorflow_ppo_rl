
from ppo_agent import *
import tensorflow as tf
import numpy as np
from lunarLander_Environment import SubprocVecEnv
from tensorboardX import SummaryWriter

sess = tf.Session()
state_size, output_size = 8, 4
agent = MLPAgent(sess, state_size, output_size)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
num_worker = 8
num_step = 128
visualize = False
sample_idx = 0
ep = 0
score = 0
global_update = 0

writer = SummaryWriter()
sub = SubprocVecEnv(num_worker, visualize)
state = np.zeros([num_worker, state_size])

while True:
    total_state, total_next_state, total_reward, total_done, total_action = [], [], [], [], []
    total_prob = []
    global_update += 1
    for _ in range(num_step):
        action, prob = agent.get_action(state)
        next_state, reward, done = sub.step(action)

        total_state.append(state)
        total_next_state.append(next_state)
        total_action.append(action)
        total_reward.append(reward)
        total_done.append(done)
        total_prob.append(prob)

        score += reward[sample_idx]
        
        state = next_state

        if done[sample_idx]:
            ep += 1
            writer.add_scalar('data/reward_epi', score, ep)
            score = 0
    
    total_state = np.stack(total_state).transpose([1, 0, 2])
    total_next_state = np.stack(total_next_state).transpose([1, 0, 2])
    total_reward = np.stack(total_reward).transpose([1, 0])
    total_action = np.stack(total_action).transpose([1, 0])
    total_done = np.stack(total_done).transpose([1, 0])
    total_prob = np.stack(total_prob).transpose([1, 0])

    total_target = []
    total_adv = []
    for i in range(num_worker):
        value, next_value = agent.get_value(total_done[i], total_state[i], total_next_state[i])
        adv, target = agent.get_gaes(total_reward[i], total_done[i], value, next_value)
        
        total_adv.append(adv)
        total_target.append(target)

    total_adv = np.stack(total_adv).reshape([-1])
    total_target = np.stack(total_target).reshape([-1])
    total_state = np.stack(total_state).reshape([-1, state_size])
    total_action = np.stack(total_action).reshape([-1])
    total_reward = np.stack(total_reward).reshape([-1])
    total_prob = np.stack(total_prob).reshape([-1])

    agent.train_model(total_state, total_action, total_target, total_adv)
    
    writer.add_scalar('data/total_prob_per_rollout', sum(total_prob)/(num_worker * num_step), global_update)
    writer.add_scalar('data/reward_per_rollout', sum(total_reward)/num_worker, global_update)

    saver.save(sess, 'model/lunarlander')