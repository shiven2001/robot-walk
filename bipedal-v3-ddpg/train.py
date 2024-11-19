import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
import os 

env = gym.make("BipedalWalker-v3")

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=100_000, log_interval=10, progress_bar=True)

# Save the model
local_dir = os.path.dirname(__file__)
results_dir = os.path.join(local_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

model.save(os.path.join(local_dir, results_dir, 'ddpg_BipedalWalker'))

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
