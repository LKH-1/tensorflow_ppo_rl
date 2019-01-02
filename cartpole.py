import gym
from agent import PPO_MLP
import numpy as np
import tensorflow as tf
from tensorboardX import SummaryWriter

writer = SummaryWriter()
sess = tf.Session()
env = gym.make('CartPole-v0')
state_size, output_size = 4, 2
agent = PPO_MLP(sess, state_size, output_size)
sess.run(tf.global_variables_initializer())

ep = 0
rollout = 32

while True:
    total_state, total_reward, total_done, total_next_state, total_action = [], [], [], [], []
    ep += 1
    state = env.reset()
    done = False
    step = 0
    while not done:
        step += 1

        action = agent.get_action([state])
        action = action[0]

        next_state, reward, done, _ = env.step(action)

        reward = 0
        if done:
            if step == 200:
                reward = 1
            else:
                reward = -1

        total_state.append(state)
        total_next_state.append(next_state)
        total_reward.append(reward)
        total_done.append(done)
        total_action.append(action)

        state = next_state
    
    total_state = np.stack(total_state)
    total_next_state = np.stack(total_next_state)
    total_reward = np.stack(total_reward)
    total_done = np.stack(total_done)
    total_action = np.stack(total_action)

    total_value, total_next_value = agent.get_value(total_state, total_next_state)
    total_target, total_adv = agent.get_gaes(total_reward, total_done, total_value, total_next_value)

    agent.assign_policy_parameters()
    for i in range(3):
        agent.train_model(total_state, total_action, total_target, total_adv)
    print(ep, step)
    writer.add_scalar('data/reward', step, ep)