from ppo_agent import *
import gym
import tensorflow as tf
import numpy as np
from tensorboardX import SummaryWriter

sess = tf.Session()
state_size, output_size = 4, 2
env = gym.make('CartPole-v1')

writer = SummaryWriter()

agent = MLPAgent(sess, state_size, output_size)
sess.run(tf.global_variables_initializer())

episode = 1000
rollout = 256
global_update = 0
state = env.reset()
ep = 0
step = 0
mean_prob = 0

while True:
    total_state, total_next_state, total_reward, total_done, total_action = [], [], [], [], []
    global_update += 1

    for _ in range(rollout):

        step += 1
        action, prob = agent.get_action(state)
        action, prob = action[0], prob[0]
        next_state, reward, done, _ = env.step(action)

        mean_prob += prob
        reward = 0

        if done:
            if step == 500:
                reward = 1
            else:
                reward = -1

        total_state.append(state)
        total_next_state.append(next_state)
        total_action.append(action)
        total_reward.append(reward)
        total_done.append(done)

        state = next_state

        if done:
            ep += 1

            writer.add_scalar('data/reward', reward, ep)
            writer.add_scalar('data/mean_prob', mean_prob / step, ep)
            writer.add_scalar('data/step', step, ep)

            print(step, reward, mean_prob / step, ep)
            
            state = env.reset()
            step = 0
            mean_prob = 0
    
    total_state = np.stack(total_state)
    total_next_state = np.stack(total_next_state)
    total_action = np.stack(total_action)
    total_reward = np.stack(total_reward)
    total_done = np.stack(total_done)

    value, next_value = agent.get_value(total_done, total_state, total_next_state)
    total_value, total_next_value = np.stack(value), np.stack(next_value)

    total_adv = agent.get_gaes(total_reward, total_done, total_value, total_next_value)
    
    agent.train_model(total_state, total_action, total_reward, total_next_value, total_adv)