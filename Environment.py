from multiprocessing import Process, Pipe
import sys
from absl import flags
from pysc2.env import sc2_env, environment
from pysc2.lib import actions, features
import matplotlib.pyplot as plt
import numpy as np
import config

def worker(remote, map_name, env_idx, frame_skip, window_size, obs_stack, visualize):
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)
    env =  sc2_env.SC2Env(
                    map_name=map_name,
                    agent_interface_format=sc2_env.parse_agent_interface_format(
                        feature_screen=window_size,
                        feature_minimap=window_size,
                        rgb_screen=None,
                        rgb_minimap=None,
                        action_space=None,
                        use_feature_units=False),
                    step_mul=frame_skip,
                    game_steps_per_episode=None,
                    disable_fog=False,
                visualize=visualize)

    history = np.zeros([window_size, window_size, 1])

    obs = env.reset()
    obs = env.step(actions=[actions.FunctionCall(config._SELECT_ARMY, [config._SELECT_ALL])])

    done = False
    score = 0
    episode = 0
    while True:
        action = remote.recv()
        if action == 'close':
            remote.close()
            
        x, y = int(action % window_size), int(action // window_size)
        obs = env.step(actions=[actions.FunctionCall(config._MOVE_SCREEN, [config._NOT_QUEUED, [x, y]])])
        
        observation = obs[0].observation.feature_screen.base[5]

        reward = obs[0].reward
        done = obs[0].step_type == environment.StepType.LAST
        score += reward

        if done:
            episode += 1
            print('Environment : ', env_idx, '| score : ', score)
            obs = env.reset()
            obs = env.step(actions=[actions.FunctionCall(config._SELECT_ARMY, [config._SELECT_ALL])])
            observation = obs[0].observation.feature_screen.base[5]
            score = 0

        history[:, :, :] = observation.reshape([window_size, window_size, obs_stack])

        remote.send([history, reward, done, env_idx])


class SubprocVecEnv:
    def __init__(self, n_proc, map_name, frame_skip, window_size, obs_stack, visualize):
        self.map_name = map_name
        self.obs_stack = obs_stack
        self.visualize = visualize
        self.n_proc = n_proc
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.n_proc)])
        self.ps = []
        self.frame_skip = frame_skip
        self.window_size = window_size
        
        for i, (work_remote,) in enumerate(zip(self.work_remotes, )):
            self.ps.append(
                Process(target=worker, args=(work_remote, self.map_name, i, self.frame_skip, self.window_size, self.obs_stack, self.visualize))
            )
        for p in self.ps:
            p.start()

    
    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(action)

        states = []
        rewards = []
        dones = []
        idxs = []
        for remote in self.remotes:
            s, r, d, idx = remote.recv()
            states.append(s)
            rewards.append(r)
            dones.append(d)
            idxs.append(idx)
        #print(idxs)
        return states, rewards, dones


    def close(self):
        for remote in self.remotes:
            remote.send('close')

        for p in self.ps:
            p.join()