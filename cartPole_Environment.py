from multiprocessing import Process, Pipe
import sys
from absl import flags
import gym
import matplotlib.pyplot as plt
import numpy as np

def worker(remote, env_idx, visualize):
    env = gym.make('CartPole-v0')
    state = env.reset()
    done = False
    step = 0
    while True:
        if visualize:
            env.render()
        step += 1
        action = remote.recv()
        if action == 'close':
            remote.close()
            
        state, reward, done, _ = env.step(action)
        
        reward = 0

        if done:
            if step == 200:
                reward = 1
            else:
                reward = -1

        if done:
            print('Environment : ', env_idx, '| score : ', step)
            state = env.reset()
            step = 0

        remote.send([state, reward, done])


class SubprocVecEnv:
    def __init__(self, n_proc, visualize):
        self.visualize = visualize
        self.n_proc = n_proc
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.n_proc)])
        self.ps = []
        
        for i, (work_remote,) in enumerate(zip(self.work_remotes, )):
            self.ps.append(
                Process(target=worker, args=(work_remote, i, self.visualize))
            )
        for p in self.ps:
            p.start()

    
    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(action)

        states = []
        rewards = []
        dones = []
        for remote in self.remotes:
            s, r, d = remote.recv()
            states.append(s)
            rewards.append(r)
            dones.append(d)
        return states, rewards, dones


    def close(self):
        for remote in self.remotes:
            remote.send('close')

        for p in self.ps:
            p.join()
