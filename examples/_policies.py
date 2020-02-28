import numpy as np

class DiscreteActionAgent(object):
    def __init__(self, theta):
        self.w = theta[:,:-1]
        self.b = theta[:,-1]
    def act(self, ob):
        logits = np.matmul(ob, self.w) + self.b
        return np.argmax(logits)
    
#         logits = ob.matmul(self.w) + self.b
#         return logits.argmax()

class RandomAgent(object):
    def __init__(self, env):
        self.env = env

    def act(self, observation=None, reward=None, done=None):
        return self.env.random_action()