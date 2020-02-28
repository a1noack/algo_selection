import gym
import numpy as np
import algo_selection
import numpy as np
from _policies import DiscreteActionAgent

def cem(f, th_mean, batch_size, n_iter, elite_frac, initial_std=1.0):
    """
    Generic implementation of the cross-entropy method for maximizing a black-box function

    f: a function mapping from vector -> scalar
    th_mean: initial mean over input distribution
    batch_size: number of samples of theta to evaluate per batch
    n_iter: number of batches
    elite_frac: each batch, select this fraction of the top-performing samples
    initial_std: initial standard deviation over parameter vectors
    """
    n_elite = int(np.round(batch_size*elite_frac))
    th_std = np.ones_like(th_mean) * initial_std

    for _ in range(n_iter):
        variation = np.random.randn(batch_size, th_mean.shape[0], th_mean.shape[1])
        ths = np.array([th_mean + dth for dth in th_std[None,:] * variation])
        ys = np.array([f(th) for th in ths])
        elite_inds = ys.argsort()[::-1][:n_elite]
        elite_ths = ths[elite_inds]
        th_mean = elite_ths.mean(axis=0)
        th_std = elite_ths.std(axis=0)
        yield {'ys' : ys, 'theta_mean' : th_mean, 'y_mean' : ys.mean()}
        
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
    params = dict(n_iter=100, batch_size=10000, elite_frac=0.01)
    num_steps = 20

    def noisy_evaluation(theta):
        agent = DiscreteActionAgent(theta)
        rew, T = do_rollout(agent, env, num_steps)
        return rew

    # train the agent
    for (i, iterdata) in enumerate(cem(f=noisy_evaluation, th_mean=np.zeros((env.num_algos, env.num_algos + 1)), **params)):
        print('Iteration %2i. Episode mean reward: %7.3f'%(i, iterdata['y_mean']))
        agent = DiscreteActionAgent(iterdata['theta_mean'])

    env.close()
