import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import os 
import pickle

env = gym.make("BipedalWalker-v3")

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=1000, log_interval=10)

# Save the model
local_dir = os.path.dirname(__file__)
results_dir = os.path.join(local_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

model.save(os.path.join(local_dir, results_dir, 'ddpg_BipedalWalker'))

vec_env = model.get_env()

model = DDPG.load(os.path.join(local_dir, results_dir, 'ddpg_BipedalWalker'))

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    env.render("human")