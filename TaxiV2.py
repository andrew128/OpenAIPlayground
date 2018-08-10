# Import libraries
import gym
import numpy as np
from random import *
# ----------------------
# Environment setup
#env = gym.make("Taxi-v2")
env = gym.make("FrozenLake-v0")
env.reset()
# ----------------------
# Use Q table to solve environment
# 500 (number of states) by 6 (number of possible actions)
# table is initially all zeros, Q values for state action
# pairs are updated at every step
Q = np.zeros([env.observation_space.n, env.action_space.n])
# var that tracks total accumlated award for each episode
G = 0
# var for learning rate
alpha = 0.1 #0.618
# episilon
e = 0.1
# gamma
y = 0.95

avg = 0
for episode in range(1, 20001): # for each episode
    state, done, G, reward = env.reset(), False, 0, 0
    while done != True: # for each step of each episode
        prob = random()
        if prob > e:
            # returns index of action of highest Q value for current state
            action = np.argmax(Q[state]) 
        else:
            action = env.action_space.sample()
        # action above is taken and we store new state as state2
        state2, reward, done, info = env.step(action)
        # bellman step:
        Q[state, action] += alpha * (reward + y * np.max(Q[state2]) - Q[state, action])
        G += reward
        state = state2
    avg += G
    if episode % 50 == 0:
        #print('Episode {} Total Reward: {}'.format(episode, G))
        print(avg/50)
        avg = 0
