import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

def rargmax(vector):
    """ Argmax taht hooses randomly among eligible maximum indices. """
    m = np.amax(vector)
    indices = np.nonzero(vector == m) [0]
    return pr.choice(indices)

register(
    id='FrozenLake-v1',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False}
)

env = gym.make('FrozenLake-v1')

Q = np.zeros([env.observation_space.n, env.action_space.n])

dis = 0.99
num_episodes = 2000

rList = []

for i in range(num_episodes):
    state, info = env.reset()
    rAll = 0
    done = False

    while not done:
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i + 1))
        
        new_state , reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        Q[state, action] = reward + dis * np.max(Q[new_state, :])
        # Discount factor와 action을 고를때 add random noise만 첨가!
        # e-greedy는 아직!
        rAll += reward
        state = new_state
    rList.append(rAll)

print("Success rate: " + str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()