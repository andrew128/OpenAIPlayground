# Import libraries
import gym
import numpy as np
# ----------------------
# Environment setup
env = gym.make("Taxi-v2")
print(env.reset())
# ----------------------
# Use Q table to solve environment
# 500 (number of states) by 6 (number of possible actions)
# table is initially all zeros, Q values for state action
# pairs are updated at every step
Q = np.zeros([env.observation_space.n, env.action_space.n])
# var that tracks total accumlated award for each episode
G = 0
# var for learning rate
alpha = 0.1

avg = 0
for episode in range(1, 101): # for each episode
    state, done, G, reward = env.reset(), False, 0, 0
    while done != True: # for each step of each episode
        action = np.argmax(Q[state]) 
        state2, reward, done, info = env.step(action)
        print(action)
        Q[state, action] += alpha * (reward + np.max(Q[state2]) - Q[state, action])
        G += reward
        state = state2
    avg += G
    if episode % 50 == 0:
        print(avg/50)
        avg = 0
