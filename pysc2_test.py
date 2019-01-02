from multiprocessing import Process, Pipe
import sys
from absl import flags
from pysc2.env import sc2_env, environment
from pysc2.lib import actions, features
import matplotlib.pyplot as plt
import numpy as np
import config

class Environment(Process):
    def __init__(self, env_idx, map_name, window_size, frame_skip, visualize, child_conn):

        self.window_size = window_size
        self.frame_skip = frame_skip
        self.visualize = visualize
        FLAGS = flags.FLAGS
        FLAGS(sys.argv)
        self.env =  sc2_env.SC2Env(
                        map_name=map_name,
                        agent_interface_format=sc2_env.parse_agent_interface_format(
                            feature_screen=self.window_size,
                            feature_minimap=self.window_size,
                            rgb_screen=None,
                            rgb_minimap=None,
                            action_space=None,
                            use_feature_units=False),
                        step_mul=self.frame_skip,
                        game_steps_per_episode=None,
                        disable_fog=False,
                        visualize=self.visualize)
        self.env_idx = env_idx
        self.child_conn = child_conn
        self.env.reset()
        self.obs = self.env.step(actions=[actions.FunctionCall(config._SELECT_ARMY, [config._SELECT_ALL])])
        self.episode = 0
        self.score = 0

        def run(self):
            while True:
                print('aaaaaa')
                action = self.child_conn.recv()

                x, y = int(action // self.window_size), int(action % self.window_size)
                self.obs = self.env.step(actions=[actions.FunctionCall(config._MOVE_SCREEN, [config._NOT_QUEUED, [x, y]])])

                state = self.obs[0].observation.feature_screen.base[5].reshape([window_size, window_size, 1])
                reward = self.obs[0].reward
                done = self.obs[0].step_type == environment.StepType.LAST

                print(self.env_idx, reward, done)

                self.score += reward

                if done:
                    self.obs = self.reset()
                    print(self.episode, self.env_idx, self.score)
                    state = self.obs[0].observation.feature_screen.base[5].reshape([window_size, window_size, 1])
                    self.score = 0

                self.child_conn.send([state, reward, done])

        def reset(self):
            self.steps = 0
            self.episode += 1
            self.env.reset()
            obs = self.env.step(actions=[actions.FunctionCall(config._SELECT_ARMY, [config._SELECT_ALL])])
            return obs

