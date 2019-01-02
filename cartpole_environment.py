from multiprocessing import Pipe, Process
import gym

class Environment(Process):
    def __init__(self, env_idx, child_conn):
        super(Environment, self).__init__()
        self.env = gym.make('CartPole-v0')
        self.is_render = False
        self.env_idx = env_idx
        self.child_conn = child_conn
        self.episode = 0
        self.step = 0
        self.env.reset()

    def run(self):
        super(Environment, self).run()
        while True:
            action = self.child_conn.recv()
            if self.is_render:
                self.env.render()

            state, reward, done, info = self.env.step(action)
            self.step += reward
            reward = 0
            if done:
                if self.step == 200:
                    reward = 1
                else:
                    reward = -1
                state = self.reset()
                done = False
                print(self.episode, self.env_idx, self.step)
                self.step = 0

            self.child_conn.send([state, reward, done, info])

    def reset(self):
        self.steps = 0
        self.episode += 1
        state = self.env.reset()
        return state