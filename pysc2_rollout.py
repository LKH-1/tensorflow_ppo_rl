from pysc2_environment import SubprocVecEnv
from agent import PPO_CNN
import tensorflow as tf
import numpy as np
from tensorboardX import SummaryWriter

writer = SummaryWriter()
map_name = 'MoveToBeacon'
num_step = 128
sample_idx, score, ep = 0, 0, 0
num_worker, window_size, frame_skip, visualize, obs_stack = 4, 16, 8, False, 1
sub = SubprocVecEnv(num_worker, map_name, window_size, frame_skip, visualize)
sess = tf.Session()
agent = PPO_CNN(sess, window_size, obs_stack, window_size * window_size)
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

state = np.zeros([num_worker, window_size, window_size, obs_stack])

global_update = 0

while True:
    total_state, total_reward, total_done, total_next_state, total_action = [], [], [], [], []
    global_update += 1
    for _ in range(num_step):
        actions = agent.get_action(state)
        
        next_state, reward, done, idx = sub.step(actions)
        
        next_state = np.stack(next_state)
        reward = np.hstack(reward)
        done = np.hstack(done)

        total_state.append(state)
        total_next_state.append(next_state)
        total_reward.append(reward)
        total_done.append(done)
        total_action.append(actions)

        score += reward[sample_idx]

        state = next_state

        if done[sample_idx]:
            ep += 1
            writer.add_scalar('data/reward_per_episode', score, ep)
            print(ep, score)
            score = 0
    
    total_state = np.stack(total_state).transpose(
            [1, 0, 2, 3, 4]).reshape([-1, window_size, window_size, obs_stack])
    total_next_state = np.stack(total_next_state).transpose(
            [1, 0, 2, 3, 4]).reshape([-1, window_size, window_size, obs_stack])
    total_reward = np.stack(total_reward).transpose().reshape([-1])
    total_action = np.stack(total_action).transpose().reshape([-1])
    total_done = np.stack(total_done).transpose().reshape([-1])

    
    total_target, total_adv = [], []

    for idx in range(num_worker):
        value, next_value = agent.get_value(total_state[idx * num_step:(idx + 1) * num_step],
                                            total_next_state[idx * num_step:(idx + 1) * num_step])
                                            
        adv, target = agent.get_gaes(total_reward[idx * num_step:(idx + 1) * num_step],
                                    total_done[idx * num_step:(idx + 1) * num_step],
                                    value,
                                    next_value)
        total_target.append(target)
        total_adv.append(adv)
    
    agent.train_model(total_state, total_action, np.hstack(total_target), np.hstack(total_adv))
    writer.add_scalar('data/reward_per_rollout', sum(total_reward)/(num_worker), global_update)
    saver.save(sess, 'MoveToBeacon/model')