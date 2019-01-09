from agent import PPO_MLP
import numpy as np
import tensorflow as tf
from tensorboardX import SummaryWriter
from lunarLander_environment import Environment
from multiprocessing import Process, Pipe

sess = tf.Session()
state_size, output_size = 8, 4
agent = PPO_MLP(sess, state_size, output_size)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, 'lunarlander/model')

num_worker = 2
num_step = 128
visualize = True
global_update = 0
sample_idx = 0
step = 0
score = 0
episode = 0

works = []
parent_conns = []
child_conns = []
for idx in range(num_worker):
    parent_conn, child_conn = Pipe()
    work = Environment(idx, child_conn, visualize)
    work.start()
    works.append(work)
    parent_conns.append(parent_conn)
    child_conns.append(child_conn)

states = np.zeros([num_worker, state_size])

while True:
    total_state, total_reward, total_done, total_next_state, total_action = [], [], [], [], []
    global_update += 1

    for _ in range(num_step):
        step += 1
        actions = agent.get_action(states)
        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(action)

        next_states, rewards, dones = [], [], []
        for parent_conn in parent_conns:
            s, r, d, _ = parent_conn.recv()
            next_states.append(s)
            rewards.append(r)
            dones.append(d)

        next_states = np.vstack(next_states)
        rewards = np.hstack(rewards)
        dones = np.hstack(dones)

        total_state.append(states)
        total_next_state.append(next_states)
        total_reward.append(rewards)
        total_done.append(dones)
        total_action.append(actions)

        score += rewards[sample_idx]
        
        if dones[sample_idx]:
            episode += 1
            print(episode, score)
            score = 0

        states = next_states