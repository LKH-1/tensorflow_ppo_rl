from agent import PPO_MLP
import numpy as np
import tensorflow as tf
from tensorboardX import SummaryWriter
from cartpole_environment import Environment
from multiprocessing import Process, Pipe

writer = SummaryWriter()
sess = tf.Session()
state_size, output_size = 4, 2
agent = PPO_MLP(sess, state_size, output_size)
sess.run(tf.global_variables_initializer())

num_worker = 4
num_step = 128
visualize = False
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
    work = Environment(idx, child_conn)
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

        states = next_states

    total_state = np.stack(total_state).transpose([1, 0, 2]).reshape([-1, state_size])
    total_next_state = np.stack(total_next_state).transpose([1, 0, 2]).reshape([-1, state_size])
    total_reward = np.stack(total_reward).transpose().reshape([-1])
    total_done = np.stack(total_done).transpose().reshape([-1])
    total_action = np.stack(total_action).transpose().reshape([-1])
    
    total_target, total_adv = [], []
    for idx in range(num_worker):
        value, next_value = agent.get_value(total_state[idx * num_step:(idx + 1) * num_step],
                                            total_next_state[idx * num_step:(idx + 1) * num_step])
        target, adv = agent.get_gaes(total_reward[idx * num_step:(idx + 1) * num_step],
                                    total_done[idx * num_step:(idx + 1) * num_step],
                                    value,
                                    next_value)
        total_target.append(target)
        total_adv.append(adv)

    agent.train_model(total_state, total_action, np.hstack(total_target), np.hstack(total_adv))