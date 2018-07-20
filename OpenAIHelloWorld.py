import gym

env = gym.make('CartPole-v0')
env.reset()

# run 20 tries
for i_episode in range(20):
    observation = env.reset()
    # for each trial try 100 random actions
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
