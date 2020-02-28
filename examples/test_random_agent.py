import gym
import algo_selection
import time
from _policies import RandomAgent
    
def do_rollout(agent, env, num_steps):
    total_rew = 0
    ob = env.reset()
    for t in range(num_steps):
        a = agent.act(ob)
        (ob, reward, done, _info) = env.step(a)
        total_rew += reward
        if done: break
    return total_rew, t+1
    
if __name__ == '__main__':
    env = gym.make('algo_selection-v0')
    agent = RandomAgent(env)

    episode_count = 100
    done = False
    num_steps = 20
    n_iter=10000
    
    global_rew = 0
    for i in range(n_iter):
        global_rew += do_rollout(agent, env, num_steps)[0]
    
    print('Episode mean reward: %7.3f'%(global_rew/n_iter))
        
    env.close()