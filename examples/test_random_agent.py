import gym
import algo_selection
import time

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, env):
        self.env = env

    def act(self, observation, reward, done):
        return self.env.random_action()
    
if __name__ == '__main__':
    env = gym.make('algo_selection-v0')
    agent = RandomAgent(env)

    episode_count = 100
    reward = 0
    done = False
    
    for i in range(episode_count):
        ob = env.reset()
        print("New episode with probe with index #{}".format(env.index))
        while True:
            next_algo = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(next_algo)
            print('\n\treward: {}\n'.format(reward))
            time.sleep(1)
            if done:
                break
        print('\n')