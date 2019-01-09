from agent import PPO_MLP
import numpy as np
import tensorflow as tf
from tensorboardX import SummaryWriter
from lunarLander_environment import Environment
import gym

writer = SummaryWriter()
sess = tf.Session()
state_size, output_size = 8, 4
agent = PPO_MLP(sess, state_size, output_size)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

env = gym.make('LunarLander-v2')
episode = 0

while True:
    state = env.reset()
    done = False
    score = 0
    total_state, total_reward, total_done, total_next_state, total_action = [], [], [], [], []
    episode += 1

    while not done:
        action = agent.get_action([state])
        action = action[0]
        next_state, reward, done, _ = env.step(action)
        
        total_state.append(state)
        total_next_state.append(next_state)
        total_done.append(done)
        total_action.append(action)
        total_reward.append(reward)

        score += reward

        state = next_state

    total_state = np.stack(total_state)
    total_next_state = np.stack(total_next_state)
    total_reward = np.stack(total_reward)
    total_done = np.stack(total_done)
    total_action = np.stack(total_action)

    value, next_value = agent.get_value(total_state, total_next_state)
    adv, target = agent.get_gaes(total_reward, total_done, value, next_value)

    agent.train_model(total_state, total_action, np.hstack(target), np.hstack(adv))

    print(episode, score)
    writer.add_scalar('data/reward_per_rollout', score, episode)
    saver.save(sess, 'lunarlander/model')