from multiprocessing import Process, Pipe
import sys, config
from absl import flags
from pysc2.env import sc2_env, environment
from pysc2.lib import actions, features
import matplotlib.pyplot as plt
import numpy as np


def worker(remote, env_idx, map_name, frame_skip, window_size, visualize):
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

    obs = env.reset()
    obs = env.step(actions=[actions.FunctionCall(config._SELECT_ARMY, [config._SELECT_ALL])])
    score = 0
    ep = 0

    while True:
        action = remote.recv()
        if action == 'close':
            env.close()

        x, y = int(action // window_size), int(action % window_size)
        obs = env.step(actions=[actions.FunctionCall(config._MOVE_SCREEN, [config._NOT_QUEUED, [x, y]])])

        state = obs[0].observation.feature_screen.base[5].reshape([window_size, window_size, 1])
        reward = obs[0].reward
        done = obs[0].step_type == environment.StepType.LAST
        score += reward
        
        if done:
            ep += 1
            print(ep, score)
            score = 0
            obs = env.reset()
            obs = env.step(actions=[actions.FunctionCall(config._SELECT_ARMY, [config._SELECT_ALL])])
            state = obs[0].observation.feature_screen.base[5].reshape([window_size, window_size, 1])
        
        remote.send([state, reward, done, env_idx])

class SubprocVecEnv:
    def __init__(self, n_proc, map_name, window_size, frame_skip, visualize):
        self.map_name = map_name
        self.window_size = window_size
        self.frame_skip = frame_skip
        self.visualize = visualize
        self.n_proc = n_proc
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.n_proc)])
        self.ps = []

        for i, (work_remote,) in enumerate(zip(self.work_remotes, )):
            self.ps.append(
                Process(target=worker, args=(work_remote, i, self.map_name, self.frame_skip, self.window_size ,self.visualize))
            )
        for p in self.ps:
            p.start()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(action)

        state, reward, done, idxs = [], [], [], []
        for remote in self.remotes:
            s, r, d, idx = remote.recv()
            state.append(s)
            reward.append(r)
            done.append(d)
            idxs.append(idx)
        
        return state, reward, done, idxs

    def close(self):
        for remote in self.remotes:
            remote.send('close')

        for p in self.ps:
            p.join()
