import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()
print(observation)

episode_over = False
while not episode_over:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    print(action)
    observation, reward, terminated, truncated, info = env.step(action)
    episode_over = terminated or truncated

env.close()