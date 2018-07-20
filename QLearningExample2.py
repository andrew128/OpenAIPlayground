# https://towardsdatascience.com/getting-started-with-reinforcement-q-learning-77499b1766b6
# https://www.oreilly.com/learning/introduction-to-reinforcement-learning-and-openai-gym <- better
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
while reward != 20:
    state, reward, done, info = env.step(env.action_space.sample())
    counter += 1

print(counter)
# ------------------
# Use Q table to solve environment

