# https://towardsdatascience.com/getting-started-with-reinforcement-q-learning-77499b1766b6
# https://www.oreilly.com/learning/introduction-to-reinforcement-learning-and-openai-gym <- better

# FOR Taxi-V2 environment in open ai gym
# https://gym.openai.com/envs/Taxi-v2/
# ----------------------
# Import libraries
import gym
import numpy as np
# ----------------------
# Environment setup
env = gym.make("Taxi-v2")
env.reset()
env.render()
# ----------------------
# get current state
state = env.reset()

# initialize counter and reward
counter = 0;
reward = None
# ----------------------
# change state
# total number of states: env.observation_space.n
#env.env.s = 114 # a single set state of many
#print(env.env.s)
#env.render()

# take an action
# down 0
# up 1
# right 2 
# left 3 
# pickup 4
# drop off 5
#state, reward, done, info = env.step(3)
#env.render()
# -----------------
# Loop that does random actions until environment is solved
"""
while reward != 20:
    state, reward, done, info = env.step(env.action_space.sample())
    counter += 1

print(counter)
"""
# ------------------
# Use Q table to solve environment
# 500 (number of states) by 6 (number of possible actions)
# table is initially all zeros, Q values for state action
# pairs are updated at every step
Q = np.zeros([env.observation_space.n, env.action_space.n])

# var that tracks total accumlated award for each episode
G = 0
# var for learning rate
alpha = 0.618

# implementation of basic Q learning algorithm
for episode in range(1, 1001):
    done = False
    G, reward = 0, 0
    state = env.reset()
    while done != True:
        # returns index of action of highest Q value for current state
        # argmax returns index of max element
        action = np.argmax(Q[state]) 
        # action above is taken and we store new state as state2
        state2, reward, done, info = env.step(action)
        # bellman step:
        Q[state, action] += alpha * (reward + np.max(Q[state2]) - Q[state, action])
        G += reward
        state = state2
    if episode % 50 == 0:
        print('Episode {} Total Reward: {}'.format(episode, G))









