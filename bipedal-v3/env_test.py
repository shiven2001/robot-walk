import gymnasium as gym

env = gym.make("BipedalWalker-v3", render_mode="human")
observation, info = env.reset()
print(observation)
print(env.observation_space)
print(env.action_space)

episode_over = False
while not episode_over:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    print(action)
    observation, reward, terminated, truncated, info = env.step(action)
    episode_over = terminated or truncated

env.close()