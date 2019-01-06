from breakout_envorinment import Environment
from multiprocessing import Pipe
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from agent import PPO_CNN
from tensorboardX import SummaryWriter
import time

writer = SummaryWriter()
sess = tf.Session()
window_size, output_size, obs_stack = 84, 3, 4
agent = PPO_CNN(sess, window_size, obs_stack, output_size)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, 'breakout/model')

global_update = 0
sample_idx = 0
score = 0
episode = 0

num_worker = 2
num_step = 256
works = []
parent_conns = []
child_conns = []
visualize = True

for idx in range(num_worker):
    parent_conn, child_conn = Pipe()
    work = Environment(visualize, idx, child_conn)
    work.start()
    works.append(work)
    parent_conns.append(parent_conn)
    child_conns.append(child_conn)

states = np.zeros([num_worker, 84, 84, 4])

while True:

    total_state, total_reward, total_done, total_next_state, total_action = [], [], [], [], []
    global_update += 1

    for _ in range(num_step):
        actions = agent.get_action(states)

        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(action)

        next_states, rewards, dones, real_dones = [], [], [], []
        for parent_conn in parent_conns:
            s, r, d, rd = parent_conn.recv()
            next_states.append(s)
            rewards.append(r)
            dones.append(d)
            real_dones.append(rd)

        next_states = np.stack(next_states)
        rewards = np.hstack(rewards)
        dones = np.hstack(dones)
        real_dones = np.hstack(real_dones)

        score += rewards[sample_idx]

        total_state.append(states)
        total_next_state.append(next_states)
        total_done.append(dones)
        total_reward.append(rewards)
        total_action.append(actions)

        states = next_states

        if real_dones[sample_idx]:
            episode += 1
            writer.add_scalar('data/reward_per_episode', score, episode)
            print(episode, score)
            score = 0

        time.sleep(0.01)