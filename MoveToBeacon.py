from multiprocessing import Process, Pipe
import sys
from absl import flags
import gym
import matplotlib.pyplot as plt
import numpy as np
from ppo_agent import CNNAgent
from Environment import SubprocVecEnv
import tensorflow as tf
from tensorboardX import SummaryWriter

sess = tf.Session()
output_size = 256
num_worker = 4
num_step = 256
sample_idx = 0
ep = 0
window_size = 16
frame_skip = 8
visualize = False
obs_stack = 1
map_name = 'MoveToBeacon'
writer = SummaryWriter()
sub = SubprocVecEnv(num_worker, map_name, frame_skip, window_size, obs_stack, visualize)
agent = CNNAgent(sess, obs_stack, window_size, frame_skip, output_size)
sess.run(tf.global_variables_initializer())

state = np.zeros([num_worker, window_size, window_size, obs_stack])
score = 0
mean_prob = 0
step = 0
global_update = 0

while True:
    total_state, total_next_state, total_reward, total_done, total_action = [], [], [], [], []
    total_prob = []
    global_update += 1
    for _ in range(num_step):
        step += 1

        action, prob = agent.get_action(state)
        next_state, reward, done = sub.step(action)

        total_prob.append(prob)
        total_state.append(state)
        total_next_state.append(next_state)
        total_action.append(action)
        total_reward.append(reward)
        total_done.append(done)

        mean_prob += prob[sample_idx]
        score += reward[sample_idx]

        state = next_state

        if done[sample_idx]:
            ep += 1
            writer.add_scalar('data/prob_per_epi', mean_prob / step, ep)
            writer.add_scalar('data/reward_per_epi', score, ep)
            mean_prob = 0
            score = 0
            step = 0

    total_state = np.stack(total_state).transpose([1, 0, 2, 3, 4])
    total_next_state = np.stack(total_next_state).transpose([1, 0, 2, 3, 4])
    total_action = np.stack(total_action).transpose([1, 0])
    total_reward = np.stack(total_reward).transpose([1, 0])
    total_done = np.stack(total_done).transpose([1, 0])

    total_next_value = []
    total_adv = []
    for i in range(num_worker):
        value, next_value = agent.get_value(total_done[i], total_state[i], total_next_state[i])
        adv = agent.get_gaes(total_reward[i], total_done[i], value, next_value)
        total_adv.append(adv)
        total_next_value.append(next_value)

    total_adv = np.stack(total_adv)
    total_next_value = np.stack(total_next_value)
    
    total_state = total_state.reshape([-1, window_size, window_size, obs_stack])
    total_action = total_action.reshape([-1])
    total_reward = total_reward.reshape([-1])
    total_next_value = total_next_value.reshape([-1])
    total_adv = total_adv.reshape([-1])
    total_prob = np.stack(total_prob).reshape([-1])

    agent.train_model(total_state, total_action, total_reward, total_next_value, total_adv)

    writer.add_scalar('data/prob_per_rollout', sum(total_prob/(num_step * num_worker)), global_update)
    writer.add_scalar('data/reward_per_rollout', sum(total_reward)/num_worker, global_update)