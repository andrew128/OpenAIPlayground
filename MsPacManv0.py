# Setup for Atari game Pacman in OpenAI gym
# https://gym.openai.com/envs/MsPacman-v0/
# ------------------------
import gym
# ------------------------
env = gym.make("MsPacman-v0")
state = env.reset()
# ------------------------
env.render()
