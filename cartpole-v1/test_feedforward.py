import os
import pickle
import neat
import gym
import numpy as np

# load the winner
local_dir = os.path.dirname(__file__)
results_dir = os.path.join(local_dir, 'results')
with open(os.path.join(local_dir, results_dir, 'winner_feedforward'), 'rb') as f:
    c = pickle.load(f)

print('Loaded genome:')
print(c)

# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config_feedforward')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

net = neat.nn.FeedForwardNetwork.create(c, config)
env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()

episode_over = False
while not episode_over:
    action = np.argmax(net.activate(observation))
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    episode_over = terminated or truncated

env.close()