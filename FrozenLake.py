# Solution for https://gym.openai.com/envs/FrozenLake8x8-v0/
# ---------------------
import gym
import numpy as np
# ---------------------
#env = gym.make("FrozenLake8x8-v0")
env = gym.make("FrozenLake-v0")
#env = gym.make("Taxi-v2")
state = env.reset()
# ---------------------
# Random attempt
reward = None
done = None

G = 0
episodes = 0
rewardTracker = []

while reward!=1:
    state, reward, done, info = env.step(env.action_space.sample())
    G += reward
    if done == True:
        rewardTracker.append(G)
        state = env.reset()
        episodes += 1

print("Reached goal after {} episodes with a average return of {}".format(episodes, sum(rewardTracker)/len(rewardTracker)))
# ---------------------
# Q learning attempt
# Q learning doesn't work in this scenario since there is no negative reward for each action. 
# Therefore all the agent ever does is try action 0 and doesn't explore other actions
counter = 0
reward = None
Q = np.zeros([env.observation_space.n, env.action_space.n]) # q table
G = 0 # total accumalated reward per episode
alpha = 0.618 # learning rate

for episode in range(1, 1001):
    done = False
    G, reward = 0, 0
    state = env.reset()
    while done != True:
        action = np.argmax(Q[state])
        state2, reward, done, info = env.step(action)
        Q[state, action] += alpha * (reward + np.max(Q[state2]) - Q[state, action])
        G += reward
        state = state2
        #env.render()
    if episode % 50 == 0:
        print('Episode {} Total Reward: {}'.format(episode, G))
