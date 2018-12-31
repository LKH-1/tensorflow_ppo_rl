
from ppo_agent import *
import tensorflow as tf
import numpy as np
from cartPole_Environment import SubprocVecEnv
from tensorboardX import SummaryWriter

sess = tf.Session()
state_size, output_size = 4, 2
agent = MLPAgent(sess, state_size, output_size)
sess.run(tf.global_variables_initializer())
num_worker = 4
num_step = 128
visualize = False
sample_idx = 0
ep = 0
score = 0

sub = SubprocVecEnv(num_worker, visualize)

state = np.zeros([num_worker, state_size])

while True:
    total_state, total_next_state, total_reward, total_done, total_action = [], [], [], [], []

    for _ in range(num_step):
        action, prob = agent.get_action(state)
        next_state, reward, done = sub.step(action)

        total_state.append(state)
        total_next_state.append(next_state)
        total_action.append(action)
        total_reward.append(reward)
        total_done.append(done)

        score += reward[sample_idx]
        
        state = next_state

        if done[sample_idx]:
            ep += 1
            print(ep, score)
            score = 0
    
    total_state = np.stack(total_state).transpose([1, 0, 2])
    total_next_state = np.stack(total_next_state).transpose([1, 0, 2])
    total_reward = np.stack(total_reward).transpose([1, 0])
    total_action = np.stack(total_action).transpose([1, 0])
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
    
    total_state = total_state.reshape([-1, state_size])
    total_action = total_action.reshape([-1])
    total_reward = total_reward.reshape([-1])
    total_next_value = total_next_value.reshape([-1])
    total_adv = total_adv.reshape([-1])

    agent.train_model(total_state, total_action, total_reward, total_next_value, total_adv)